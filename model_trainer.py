#!/usr/bin/env python3
# model_trainer.py - Single process with device_map="auto" sharding

import os
import logging
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                          BitsAndBytesConfig, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import importlib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        output_dir: str = "./mistral7b_model",
        *,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_bnb: bool = True,
    ):
        """Model trainer with safer, configurable defaults.

        Args:
            model_name: model path or HF id
            output_dir: dir to save checkpoints
            bf16: use bfloat16 during training (requires hardware support)
            gradient_checkpointing: enable gradient checkpointing (saves memory)
            lora_r, lora_alpha: LoRA hyperparameters
            use_bnb: attempt BitsAndBytes quantized loading when available
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.use_bnb = use_bnb

        local = os.path.exists(model_name)
        tokenizer_kwargs = {"trust_remote_code": True, "local_files_only": local}
        if self.hf_token:
            tokenizer_kwargs["use_auth_token"] = self.hf_token

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = None

    def _bnb_importable(self) -> bool:
        try:
            importlib.import_module("bitsandbytes")
            return True
        except Exception:
            return False

    def load_model(self):
        logger.info(f"Loading model from {self.model_name}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local = os.path.exists(self.model_name)
        
        # Use device_map="auto" to automatically shard across available GPUs
        device_map = "auto"
        logger.info("Using device_map='auto' - model will be sharded across available GPUs")

        hf_kwargs = {"trust_remote_code": True, "local_files_only": local, "low_cpu_mem_usage": True}
        if self.hf_token:
            hf_kwargs["use_auth_token"] = self.hf_token

        # Try quantization if bitsandbytes is available and allowed
        quantization_successful = False
        if self.use_bnb and self._bnb_importable():
            try:
                logger.info("Attempting 4-bit quantized load...")
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                    torch_dtype=(torch.bfloat16 if self.bf16 and torch.cuda.is_available() else None),
                    **hf_kwargs
                )
                self.model = prepare_model_for_kbit_training(self.model)
                logger.info("✓ Model loaded with 4-bit quantization")
                quantization_successful = True
            except Exception as e:
                logger.warning(f"4-bit loading failed: {e}")
                logger.info("Falling back to non-quantized load...")

        # Fallback to non-quantized if quantization failed or unavailable
        if not quantization_successful:
            dtype = torch.bfloat16 if (self.bf16 and torch.cuda.is_available()) else None
            logger.info(f"Loading model without quantization (dtype={dtype})")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=dtype,
                **hf_kwargs
            )
            logger.info("✓ Model loaded without quantization")

        # Always apply LoRA
        self.model = self._apply_lora()
        logger.info("✓ LoRA applied")
        
        # Verify trainable params exist
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError("ERROR: No trainable parameters after LoRA! Training cannot proceed.")

    def _apply_lora(self, r: int = 8, alpha: int = 16):
        """Apply LoRA adapters with configurable hyperparameters."""
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
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

    def train(self, train_ds, val_ds, num_epochs=3, lr=1e-5, batch_size=1, grad_accum=2):
        """Train with configurable options and safer defaults."""

        eval_strategy = "steps" if val_ds is not None else "no"
        eval_steps = 5000 if val_ds is not None else None
        logging_steps = 100

        effective_batch = batch_size * grad_accum
        logger.info(f"Training config: epochs={num_epochs}, per_device_batch={batch_size}, grad_accum={grad_accum}, effective_batch={effective_batch}, lr={lr}")

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=self.gradient_checkpointing,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine_with_restarts",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            bf16=self.bf16,
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_steps is not None else 0,
            save_strategy="steps" if val_ds is not None else "no",
            save_steps=10000 if val_ds is not None else None,
            save_total_limit=2,
            load_best_model_at_end=True if val_ds is not None else False,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            report_to=[],  # disable integrations by default
            # NO FSDP for single process
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        logger.info("Starting training...")
        try:
            trainer.train()
        except Exception as e:
            logger.error("Training failed: %s", e)
            # Helpful debug hints
            if "CUDA error" in str(e) or "cuda" in str(e).lower():
                logger.error("CUDA error during training. Try: 1) CUDA_LAUNCH_BLOCKING=1 for deterministic stack traces; 2) set bf16=False; 3) reduce batch size or disable quantization/offload.")
            raise

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"✓ Model saved to {self.output_dir}")