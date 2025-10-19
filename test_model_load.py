#!/usr/bin/env python3
# test_model_load.py - Test Llama 4 Scout loading before full training

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from config import cfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def test_model_loading():
    """Test if model loads and can do a forward pass"""

    model_name = cfg["model_name"]
    hf_token = cfg.get("hf_token", None)
    
    logger.info("="*70)
    logger.info("LLAMA 4 SCOUT - MODEL LOADING TEST")
    logger.info("="*70)
    
    # Check GPUs
    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        return False
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Load tokenizer
    logger.info("\n[1/5] Loading tokenizer...")
    try:
        tokenizer_kwargs = {"trust_remote_code": True}
        if hf_token:
            tokenizer_kwargs["token"] = hf_token
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✓ Tokenizer loaded")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False
    
    # Load model with 4-bit quantization
    logger.info("\n[2/5] Loading model with 4-bit quantization...")
    try:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        
        model_kwargs = {
            "quantization_config": bnb_cfg,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "use_cache": False,
        }
        if hf_token:
            model_kwargs["token"] = hf_token
        
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info("✓ Model loaded")
        
        # Show device map
        if hasattr(model, 'hf_device_map'):
            logger.info("\nDevice map:")
            for name, device in list(model.hf_device_map.items())[:10]:
                logger.info(f"  {name}: {device}")
            if len(model.hf_device_map) > 10:
                logger.info(f"  ... and {len(model.hf_device_map) - 10} more layers")
    
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Prepare for training
    logger.info("\n[3/5] Preparing for k-bit training...")
    try:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        logger.info("✓ Model prepared for training")
    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        return False
    
    # Apply LoRA
    logger.info("\n[4/5] Applying LoRA...")
    try:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("✓ LoRA applied")
    except Exception as e:
        logger.error(f"Failed to apply LoRA: {e}")
        return False
    
    # Test forward pass
    logger.info("\n[5/5] Testing forward pass...")
    try:
        model.train()
        
        # Create dummy input
        test_text = "This is a test input for the model."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        
        # Move inputs to first available GPU
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        
        logger.info("  Running forward pass...")
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(**inputs)
            loss = outputs.loss
        
        logger.info(f"  Loss: {loss.item():.4f}")
        logger.info("✓ Forward pass successful!")
        
        # Test backward pass
        logger.info("  Testing backward pass...")
        loss.backward()
        logger.info("✓ Backward pass successful!")
        
    except Exception as e:
        logger.error(f"Forward/backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Memory stats
    logger.info("\n" + "="*70)
    logger.info("MEMORY USAGE")
    logger.info("="*70)
    for i in range(gpu_count):
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free, mem_total = torch.cuda.mem_get_info(i)
        logger.info(f"GPU {i}:")
        logger.info(f"  Allocated: {mem_allocated:.2f} GB")
        logger.info(f"  Reserved:  {mem_reserved:.2f} GB")
        logger.info(f"  Free:      {mem_free/1024**3:.2f} GB / {mem_total/1024**3:.2f} GB")
    
    logger.info("\n" + "="*70)
    logger.info("✓ ALL TESTS PASSED!")
    logger.info("="*70)
    logger.info("\nModel is ready for training. You can now run:")
    logger.info("  python iomt_policy_generation.py")
    
    return True

if __name__ == "__main__":
    import sys
    success = test_model_loading()
    sys.exit(0 if success else 1)