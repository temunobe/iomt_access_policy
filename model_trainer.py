# model_trainer.py

import os
import logging
import torch
import gc
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train Llama with distributed multi-GPU support"""

    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct", output_dir: str = "./llama4_iomt_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.gpu_specs = self._detect_gpu_specs()
        logger.info(f"GPU Config: {self.gpu_specs}")
        
        # Get distributed training info
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        logger.info(f"Trainer initialized - rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}")
        
        # CRITICAL: Clear GPU cache before loading anything
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # Load tokenizer
        if os.path.exists(model_name):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = None

    def _detect_gpu_specs(self):
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Log memory usage
            for i in range(gpu_count):
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            return {"count": gpu_count, "name": gpu_name, "memory_gb": gpu_memory}
        return {"count": 0, "name": "CPU", "memory_gb": 0}

    def _apply_qlora(self):
        logger.info("Applying QLoRA adapters...")
        lora_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
        )
        model = get_peft_model(self.model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
        return model

    def load_model(self):
        logger.info(f"Loading model: {self.model_name}")
        
        # Aggressive memory cleanup before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("Cleared CUDA cache before model loading")

        # SOLUTION 1: Use 4-bit with CPU offloading to manage memory
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,  # Use fp16 instead of bf16
            bnb_4bit_use_double_quant=True,
        )

        try:
            logger.info("Loading model with 4-bit quantization...")
            
            # SOLUTION 2: Calculate safe max_memory (leave 10GB buffer)
            if self.gpu_specs['count'] > 0:
                safe_memory_gb = max(10, self.gpu_specs['memory_gb'] - 10)
                max_memory_config = {i: f"{safe_memory_gb}GB" for i in range(self.gpu_specs['count'])}
                # Add CPU offload
                max_memory_config["cpu"] = "50GB"
            else:
                max_memory_config = None
            
            logger.info(f"Memory config: {max_memory_config}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,  # Consistent with quantization
                local_files_only=os.path.exists(self.model_name),
                low_cpu_mem_usage=True,
                max_memory=max_memory_config,
                # SOLUTION 3: Offload some layers to CPU if needed
                offload_folder="./offload_tmp",
                offload_state_dict=True,
            )
            
            logger.info("Model loaded, preparing for k-bit training...")
            
            # SOLUTION 4: Don't call prepare_model_for_kbit_training if model is already quantized
            # The model is already prepared when loaded with quantization_config
            # Just apply LoRA directly
            
            # FIX: Llama 4 MoE doesn't support gradient checkpointing with the router
            # We'll handle this in TrainingArguments instead
            # self.model.gradient_checkpointing_enable()  # REMOVED - causes router unpacking error
            self.model = self._apply_qlora()
            
            logger.info("✓ Model loaded with QLoRA (4-bit)")
            
            # Log final memory usage
            if torch.cuda.is_available():
                for i in range(self.gpu_specs['count']):
                    allocated = torch.cuda.memory_allocated(i) / 1e9
                    reserved = torch.cuda.memory_reserved(i) / 1e9
                    logger.info(f"GPU {i} after load - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
            
            return self.model
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM Error during quantized load: {e}")
            logger.info("Attempting recovery: clearing cache and retrying with more aggressive offloading")
            
            # Emergency cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
            try:
                # SOLUTION 5: More aggressive CPU offloading
                max_memory_config = {}
                if self.gpu_specs['count'] > 0:
                    # Use only 60% of GPU memory
                    conservative_memory = int(self.gpu_specs['memory_gb'] * 0.6)
                    max_memory_config = {i: f"{conservative_memory}GB" for i in range(self.gpu_specs['count'])}
                max_memory_config["cpu"] = "100GB"
                
                logger.info(f"Retry with conservative memory config: {max_memory_config}")
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="balanced",  # Try balanced instead of auto
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    local_files_only=os.path.exists(self.model_name),
                    low_cpu_mem_usage=True,
                    max_memory=max_memory_config,
                    offload_folder="./offload_tmp",
                    offload_state_dict=True,
                )
                
                self.model.gradient_checkpointing_enable()
                self.model = self._apply_qlora()
                logger.info("✓ Model loaded with conservative settings")
                return self.model
                
            except Exception as e2:
                logger.error(f"Failed even with conservative settings: {e2}")
                raise RuntimeError(
                    "Cannot load model even with aggressive CPU offloading. "
                    "Suggestions:\n"
                    "1. Kill other GPU processes: nvidia-smi to identify, kill -9 <PID>\n"
                    "2. Use multiple GPUs with model parallelism\n"
                    "3. Use a smaller model or reduce LoRA rank\n"
                    "4. Try running on CPU (very slow): device_map='cpu'"
                ) from e2

    def train(self, train_dataset, eval_dataset, num_epochs=1, learning_rate=1e-5, batch_size=1, grad_accum=16):
        logger.info("STARTING DISTRIBUTED TRAINING")
        
        # Clear cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,  # Enable to save memory during training
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine_with_restarts",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            fp16=torch.cuda.is_available(),
            bf16=False,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            report_to=[],
            max_grad_norm=1.0,
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            ddp_backend="nccl" if torch.cuda.is_available() else "gloo",
            local_rank=self.local_rank,
            ddp_timeout=3600,
            # SOLUTION 6: More aggressive memory management during training
            max_steps=-1,
            logging_first_step=True,
            # Enable these to reduce memory usage
            eval_accumulation_steps=4,
            save_safetensors=True,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )

        try:
            trainer.train()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"OOM during training: {e}")
            logger.error("Try reducing batch_size to 1 and grad_accum to 4")
            raise
        except Exception as e:
            logger.error("Training failed with exception: %s", e)
            if "CUDA error" in str(e) or "cuda" in str(e).lower():
                logger.error("CUDA error detected. Debug steps:\n"
                             " 1) CUDA_LAUNCH_BLOCKING=1 python iomt_policy_generation.py\n"
                             " 2) Check nvidia-smi for other processes using GPU memory\n"
                             " 3) Reduce batch size and gradient accumulation\n"
                             " 4) Use fewer GPUs or single GPU mode")
            raise
        
        # Save only on rank 0
        if self.rank == 0:
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"✓ Training complete, model saved to {self.output_dir}")
        
        return trainer