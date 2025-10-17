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
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", output_dir: str = "./mistral7b_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

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

        # Try quantization if bitsandbytes is available
        quantization_successful = False
        if self._bnb_importable():
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
                    torch_dtype=torch.bfloat16,
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
            logger.info("Loading model without quantization in bfloat16...")
            logger.info("With 2 GPUs available, model will be automatically sharded across both")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16,
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

    def _apply_lora(self):
        """Apply LoRA adapters - must succeed"""
        cfg = LoraConfig(
            r=8,  # Smaller rank for memory efficiency
            lora_alpha=16,
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
        """Train with gradient checkpointing (single process)"""
        
        eval_strategy = "steps" if val_ds is not None else "no"
        eval_steps = 50 if val_ds is not None else None
        logging_steps = 10

        args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,  # Enable for single-process training
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine_with_restarts",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            bf16=True,
            logging_steps=logging_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps if eval_steps is not None else 0,
            save_strategy="steps" if val_ds is not None else "no",
            save_steps=100 if val_ds is not None else None,
            save_total_limit=2,
            load_best_model_at_end=True if val_ds is not None else False,
            max_grad_norm=1.0,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
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
        trainer.train()

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"✓ Model saved to {self.output_dir}")