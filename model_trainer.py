# model_trainer.py

import os
import logging
import torch
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
        
        # Get distributed training info (torchrun sets these automatically)
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        logger.info(f"Trainer initialized - rank={self.rank}, local_rank={self.local_rank}, world_size={self.world_size}")

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

        try:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_8bit_compute_dtype=torch.bfloat16,
                bnb_8bit_use_double_quant=True,
            )
        except Exception as e:
            logger.warning(f"Could not create BitsAndBytesConfig: {e}")
            bnb_config = None

        if bnb_config is not None:
            try:
                logger.info("Loading model with 4-bit quantization (BitsAndBytes)...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    local_files_only=os.path.exists(self.model_name),
                    low_cpu_mem_usage=True
                )
                self.model = prepare_model_for_kbit_training(self.model)
                self.model = self._apply_qlora()
                logger.info("✓ Model loaded with QLoRA")
                return self.model
            except Exception as e:
                logger.warning(f"Quantized loading failed: {e}. Falling back to non-quantized.")

        logger.info("Loading model without quantization.")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
            local_files_only=os.path.exists(self.model_name),
            low_cpu_mem_usage=True
        )

        try:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = self._apply_qlora()
        except Exception:
            logger.info("Skipping QLoRA for non-quantized model.")

        logger.info("✓ Model loaded")
        return self.model

    def train(self, train_dataset, eval_dataset, num_epochs=1, learning_rate=1e-5, batch_size=1, grad_accum=16):
        logger.info("STARTING DISTRIBUTED TRAINING")
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accum,
            gradient_checkpointing=True,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.05,
            lr_scheduler_type="cosine_with_restarts",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            bf16=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            report_to="tensorboard",
            max_grad_norm=1.0,
            remove_unused_columns=False,
            # Distributed training settings
            ddp_find_unused_parameters=False,
            ddp_backend="nccl",
            local_rank=self.local_rank,
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

        trainer.train()
        
        # Save only on rank 0 to avoid conflicts
        if self.rank == 0:
            trainer.save_model(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            logger.info(f"✓ Training complete, model saved to {self.output_dir}")
        
        return trainer