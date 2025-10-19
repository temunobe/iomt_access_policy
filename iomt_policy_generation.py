#!/usr/bin/env python3
# iomt_policy_generation.py - Llama 4 Scout 17B 16E VERSION

import os
import logging
import datetime
import torch
import torch.distributed as dist
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_trainer import ModelTrainer
from policy_generator import PolicyGenerator
from evaluator import ModelEvaluator
from config import cfg
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_distributed(timeout_min=30):
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if not dist.is_available() or world_size == 1:
        return rank, local_rank, world_size

    if torch.cuda.is_available():
        ngpu = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % max(1, ngpu))

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo",
            init_method="env://", 
            timeout=datetime.timedelta(minutes=timeout_min)
        )
    return rank, local_rank, world_size

def main():
    # For multi-GPU with device_map="auto", run as single process
    # Don't use torchrun or distributed launch
    
    rank = 0  # Single process mode
    local_rank = 0
    world_size = 1

    logger.info("="*70)
    logger.info("IOMT Policy Generation Pipeline - Llama 4 Scout 17B 16E")
    logger.info(f"Running in single-process mode with device_map='auto'")
    logger.info(f"Available GPUs: {torch.cuda.device_count()}")
    logger.info("="*70)

    # ====== STAGE 1: LOAD DATA ======
    scenarios = None
    logger.info("\n[STAGE 1] Loading dataset...")
    try:
        loader = DataLoader(cfg.get('data_dir', 'clinical_access_control_scenarios.csv'))
        scenarios = loader.load()
        logger.info(f"✓ Loaded {len(scenarios)} scenarios")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    # ====== STAGE 2: FORMAT AND TOKENIZE ======
    logger.info("\n[STAGE 2] Formatting and tokenizing data...")
    try:
        formatter = DataFormatter(cfg["model_name"])
        logger.info("✓ Tokenizer loaded")
        
        # Format and split
        dataset = formatter.format_and_split(scenarios)
        logger.info("✓ Data formatted and split (train/val/test)")
        
        # Tokenize
        tokenized = formatter.prepare_tokenized_dataset(dataset, cfg['max_seq_length'])# max_seq_length=4096)
        logger.info("✓ Data tokenized")
        
        # Save to disk
        os.makedirs(cfg["llama_tokenized_cache"], exist_ok=True)
        tokenized.save_to_disk(cfg["llama_tokenized_cache"])
        logger.info(f"✓ Tokenized dataset saved to {cfg['llama_tokenized_cache']}")
        
    except Exception as e:
        logger.error(f"Failed in formatting/tokenization: {e}")
        raise

    # ====== STAGE 3: TRAIN MODEL ======
    logger.info(f"\n[STAGE 3] Training model...")
    logger.info(f"Model will be automatically sharded across {torch.cuda.device_count()} GPUs")
    
    try:
        trainer = ModelTrainer(
            model_name=cfg["model_name"],
            output_dir=cfg["model_output"]
        )
        trainer.load_model()
        logger.info("Model loaded and sharded across GPUs")
        
        # Train with automatic multi-GPU sharding
        trainer.train(
            train_ds=tokenized["train"],
            val_ds=tokenized["validation"],
            num_epochs=cfg["epochs"],
            lr=cfg["lr"],
            batch_size=cfg["batch"],
            grad_accum=cfg["grad_accum"]
        )
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ====== STAGE 4: EVALUATE ======
    logger.info(f"\n[STAGE 4] Evaluating model...")
    try:
        # Reload scenarios for evaluation
        loader = DataLoader(cfg.get('data_dir', 'clinical_access_control_scenarios.csv'))
        scenarios = loader.load()

        gen = PolicyGenerator(cfg["model_output"])

        # Sample policies
        logger.info("Generating sample policies...")
        for i, s in enumerate(scenarios[:min(3, len(scenarios))]):
            try:
                policy = gen.generate(
                    s.description,
                    {"device_type": s.device_type, "criticality": s.criticality}
                )
                val = gen.validate_policy(policy)
                logger.info(f"  Sample {i+1}: Valid XML={val['is_valid_xml']}, "
                           f"Target={val['has_target']}, Rules={val['has_rules']}")
            except Exception as e:
                logger.warning(f"  Sample {i+1} generation failed: {e}")
        
        # Full evaluation
        logger.info(f"Running full evaluation on {cfg['eval_size']} scenarios...")
        evaluator = ModelEvaluator(gen)
        metrics = evaluator.evaluate(scenarios, sample_size=cfg['eval_size'])
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Valid XML Rate: {metrics['valid_xml_rate']:.1f}%")
        logger.info(f"Has Target Rate: {metrics['has_target_rate']:.1f}%")
        logger.info(f"Has Rules Rate: {metrics['has_rules_rate']:.1f}%")
        
        if 'avg_time' in metrics:
            logger.info(f"Avg Generation Time: {metrics['avg_time']:.2f}s")
        if 'median_time' in metrics:
            logger.info(f"Median Generation Time: {metrics['median_time']:.2f}s")
        
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    logger.info("\n✓ Pipeline complete!")

if __name__ == "__main__":
    main()