#!/usr/bin/env python3
# model_trainer.py - Llama 4 Scout 17B 16E - FIXED for MoE architecture

import os
import logging
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import importlib
from config import cfg

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E", output_dir: str = "/home/bsindala/projects/llama4_finetuned_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.hf_token = cfg['hf_token'] if 'hf_token' in cfg else None

        local = os.path.exists(model_name)
        tokenizer_kwargs = {"trust_remote_code": True, "local_files_only": local}
        if self.hf_token:
            tokenizer_kwargs["token"] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = None
        self.quantized = False

    def _bnb_importable(self) -> bool:
        try:
            importlib.import_module("bitsandbytes")
            return True
        except Exception:
            return False

    def load_model(self):
        logger.info(f"Loading Llama 4 Scout 17B 16E from {self.model_name}")
        logger.info("NOTE: Gradient checkpointing DISABLED due to MoE architecture incompatibility")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local = os.path.exists(self.model_name)
        
        # Use device_map="auto" for automatic sharding
        device_map = "auto"
        logger.info(f"Using device_map='auto' - model will be sharded across {torch.cuda.device_count()} GPUs")

        hf_kwargs = {
            "trust_remote_code": True, 
            "local_files_only": local, 
            "low_cpu_mem_usage": True,
            "use_cache": False,  # Required for training
        }
        if self.hf_token:
            hf_kwargs["token"] = self.hf_token

        # Try 4-bit quantization (REQUIRED for MoE models with limited memory)
        if self._bnb_importable():
            try:
                logger.info("Loading with 4-bit quantization (REQUIRED for Llama 4 Scout MoE)...")
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_storage=torch.bfloat16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    **hf_kwargs
                )
                
                # Prepare for k-bit training WITHOUT gradient checkpointing
                self.model = prepare_model_for_kbit_training(
                    self.model, 
                    use_gradient_checkpointing=False  # CRITICAL: Disabled for MoE
                )
                self.quantized = True
                logger.info("✓ Model loaded with 4-bit quantization")
                
            except Exception as e:
                logger.error(f"4-bit loading failed: {e}")
                logger.error("Llama 4 Scout 17B MoE requires quantization to fit in memory")
                raise RuntimeError("4-bit quantization required but failed. Please ensure bitsandbytes is properly installed.")
        else:
            logger.error("bitsandbytes not available!")
            logger.error("Llama 4 Scout 17B MoE (16 experts) is too large without quantization")
            raise RuntimeError("4-bit quantization required. Please install: pip install bitsandbytes")

        # Apply LoRA after quantization
        self.model = self._apply_lora()
        logger.info("✓ LoRA applied")
        
        # Verify trainable params exist
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError("ERROR: No trainable parameters after LoRA! Training cannot proceed.")
        
        # Ensure model is in training mode
        self.model.train()
        
        # Disable any internal gradient checkpointing that may have been set
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()

    def _apply_lora(self):
        """Apply LoRA adapters for Llama 4 Scout MoE"""
        # For MoE models, be selective with target modules to avoid memory issues
        # Focus on attention layers, not all expert layers
        cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            # Only target attention projections, not MoE experts
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=None,
        )
        
        try:
            model = get_peft_model(self.model, cfg)
        except Exception as e:
            logger.error(f"Failed to apply LoRA: {e}")
            raise
        
        # Log trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")
        
        return model

    def train(self, train_ds, val_ds, num_epochs=3, lr=1e-5, batch_size=1, grad_accum=16):
        """Train Llama 4 Scout WITHOUT gradient checkpointing"""
        
        eval_strategy = "steps" if val_ds is not None else "no"
        eval_steps = 50 if val_ds is not None else None
        logging_steps = 10

        # Very conservative settings for MoE
        effective_batch = batch_size
        grad_accum = max(grad_accum, 32)  # Increase to compensate for no checkpointing
        
        logger.info(f"Effective batch size: {effective_batch * grad_accum}")
        logger.info(f"Per-device batch: {effective_batch}, Gradient accumulation: {grad_accum}")
        logger.info("WARNING: Gradient checkpointing DISABLED for MoE stability")

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=effective_batch,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=False,  # CRITICAL: Disabled for MoE
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
            optim="adamw_bnb_8bit" if self.quantized else "adamw_torch",
            bf16=True,
            bf16_full_eval=True,
            logging_steps=logging_steps,
            logging_first_step=True,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_steps is not None else 0,
            save_strategy="steps" if val_ds is not None else "no",
            save_steps=200 if val_ds is not None else None,
            save_total_limit=2,
            load_best_model_at_end=True if val_ds is not None else False,
            max_grad_norm=0.3,  # Lower for stability
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            report_to="none",  # Disable wandb/tensorboard for stability
            # Memory optimizations
            max_steps=-1,  # Train for full epochs
            eval_accumulation_steps=1,
        )

        # Custom data collator with proper padding
        data_collator = DataCollatorForLanguageModeling(
            self.tokenizer, 
            mlm=False,
            pad_to_multiple_of=8  # Efficient for GPU
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
        )

        logger.info("Starting training with Llama 4 Scout 17B 16E MoE...")
        logger.info(f"Training on {len(train_ds)} samples")
        
        try:
            trainer.train()
            logger.info("✓ Training completed successfully")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("OOM Error! Try:")
                logger.error("  1. Reduce max_seq_length in tokenization")
                logger.error("  2. Increase gradient_accumulation_steps")
                logger.error("  3. Use fewer samples for training")
            raise

        # Save model
        logger.info("Saving model...")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"✓ Model saved to {self.output_dir}")