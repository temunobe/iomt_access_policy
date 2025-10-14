# model_trainer.py

# model_trainer.py

import os
import logging
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
                         BitsAndBytesConfig, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import importlib

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct", output_dir: str = "./llama4_model"):
        self.model_name = model_name
        self.output_dir = output_dir
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        local = os.path.exists(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=local)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.model = None

    def _bnb_available(self) -> bool:
        """
        Return True if bitsandbytes is importable and provides the native 'nn' module
        (i.e., the compiled extension). If not available, quantized 4-bit loading
        will be skipped to avoid AttributeError.
        """
        try:
            bnb = importlib.import_module("bitsandbytes")
            available = hasattr(bnb, "nn")
            if not available:
                logger.warning("bitsandbytes module imported but does not expose 'nn' (probably not compiled for CUDA).")
            return available
        except Exception:
            logger.warning("bitsandbytes is not importable in this environment.")
            return False

    def load_model(self):
        logger.info(f"Loading model from {self.model_name}")

        # try to clear any leftover allocation and reduce fragmentation prior to loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        local = os.path.exists(self.model_name)

        # Choose device_map depending on whether we're in distributed (DDP) or single-process mode:
        if torch.cuda.is_available():
            if self.world_size > 1:
                # Running under torchrun/DDP: do NOT use device_map="auto"
                # Instead, load onto the local GPU for this process. Note: this requires
                # the model (or quantized model) to fit on a single GPU for each process.
                device_map = {"": f"cuda:{self.local_rank}"}
                logger.info(f"Distributed run detected (world_size={self.world_size}). Loading model onto cuda:{self.local_rank}.")
            else:
                device_map = "auto"
                logger.info("Single-process run detected. Using device_map='auto' to let HF dispatch weights over GPUs.")
        else:
            device_map = None

        # prefer 4-bit QLoRA if bitsandbytes is properly installed/compiled
        if self._bnb_available():
            try:
                logger.info("bitsandbytes available: attempting 4-bit QLoRA load")
                bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_cfg,
                    device_map=device_map,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    local_files_only=local,
                    low_cpu_mem_usage=True,
                )
                self.model = prepare_model_for_kbit_training(self.model)
                self.model = self._apply_lora()
                logger.info("✓ Model loaded with 4-bit QLoRA")
                return
            except Exception as e:
                logger.warning(f"4-bit loading with bitsandbytes failed: {e}. Falling back to non-quantized load.")

        # Fallback path (no bitsandbytes / quantization)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            logger.info("Loading model without 4-bit quantization (this requires more memory).")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map if device_map is not None else ( "auto" if torch.cuda.is_available() else None),
                torch_dtype=dtype,
                trust_remote_code=True,
                local_files_only=local,
                low_cpu_mem_usage=True,
            )
            try:
                self.model = self._apply_lora()
            except Exception:
                logger.info("Skipping LoRA for the fallback (CPU) path or non-compatible model.")
            logger.info("✓ Model loaded without quantization")
        except RuntimeError as re:
            # likely OOM or other runtime error while loading the full model
            logger.error("Failed to load the non-quantized model (likely OOM).")
            logger.error(str(re))
            # Provide a clearer actionable message
            if self.world_size > 1:
                logger.error("You are running in distributed mode. Loading a full (non-quantized) model onto each GPU may OOM.\n"
                             "Options:\n"
                             " - Install/compile bitsandbytes so 4-bit QLoRA works.\n"
                             " - Run as a single process and use device_map='auto' so the model is sharded across GPUs.\n"
                             " - Use accelerate/deepspeed/FSDP to shard parameters across processes.\n")
            else:
                logger.error("Consider installing bitsandbytes to enable 4-bit quantization or use machines with more GPU memory.")
            raise

    def _apply_lora(self):
        cfg = LoraConfig(r=64, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        return get_peft_model(self.model, cfg)

    def train(self, train_ds, val_ds, num_epochs=3, lr=1e-5, batch_size=1, grad_accum=16):
        # When running as a single-process (world_size == 1), some versions of accelerate
        # will try to call torch.distributed.get_world_size() while creating the PartialState
        # which raises if the default process group isn't initialized. To avoid that error,
        # initialize a minimal process group for the local single-process case so accelerate
        # can introspect safely. We'll destroy it afterwards.
        created_process_group = False
        try:
            if self.world_size == 1 and torch.distributed.is_available() and not torch.distributed.is_initialized():
                try:
                    # Use gloo for a lightweight single-process group (works without NCCL requirements).
                    torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
                    created_process_group = True
                    logger.info("Initialized lightweight torch.distributed process group (gloo) for single-process accelerate compatibility.")
                except Exception as e:
                    # If we can't init, we continue but note that accelerate may error in some environments.
                    logger.warning(f"Could not initialize torch distributed process group: {e}")

            args = TrainingArguments(
                output_dir=self.output_dir, num_train_epochs=num_epochs, per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=1, gradient_accumulation_steps=grad_accum, gradient_checkpointing=True,
                learning_rate=lr, weight_decay=0.01, warmup_ratio=0.05, lr_scheduler_type="cosine_with_restarts",
                optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch", bf16=True,
                logging_steps=10, eval_strategy="steps", eval_steps=100, save_strategy="steps",
                save_steps=200, save_total_limit=2, load_best_model_at_end=True, max_grad_norm=1.0,
                ddp_find_unused_parameters=False, ddp_backend="nccl", local_rank=self.local_rank
            )

            trainer = Trainer(
                model=self.model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                tokenizer=self.tokenizer, data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            )
            trainer.train()

            if self.rank == 0:
                trainer.save_model(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                logger.info(f"✓ Model saved to {self.output_dir}")
        finally:
            # Destroy the process group we created to avoid warnings on exit.
            if created_process_group and torch.distributed.is_initialized():
                try:
                    torch.distributed.destroy_process_group()
                    logger.info("Destroyed the temporary torch.distributed process group.")
                except Exception as e:
                    logger.warning(f"Failed to destroy process group: {e}")