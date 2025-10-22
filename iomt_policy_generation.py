#!/usr/bin/env python3
# iomt_policy_generation.py - FIXED VERSION

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
from config import config
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
    rank, local_rank, world_size = setup_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    # cfg = {
    #     "model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    #     "model_output": "./mistral7b_model_v3",
    #     "tokenized_cache": "./tokenized_dataset_cache",
    #     "epochs": 1,  # Reduced for memory
    #     "lr": 1e-5,
    #     "batch": 1,
    #     "grad_accum": 8,  # Reduced from 16
    #     "eval_size": 50   # Reduced from 100
    # }

    if rank == 0:
        logger.info("="*70)
        logger.info("IOMT Policy Generation Pipeline - Multi-GPU Training")
        logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
        logger.info("="*70)

    # ====== STAGE 1: LOAD DATA (RANK 0 ONLY) ======
    scenarios = None
    if rank == 0:
        logger.info("\n[STAGE 1] Loading dataset...")
        try:
            loader = DataLoader(config.get('data_dir', '/home/bsindala/projects/datasets/clinical_access_control_scenarios_1M.csv'))
            scenarios = loader.load()
            logger.info(f"✓ Loaded {len(scenarios)} scenarios")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    if world_size > 1:
        dist.barrier()

    # ====== STAGE 2: FORMAT AND TOKENIZE (RANK 0 ONLY) ======
    if rank == 0:
        logger.info("\n[STAGE 2] Formatting and tokenizing data...")
        try:
            formatter = DataFormatter(config["mistral_model_name"])
            logger.info("✓ Tokenizer loaded")
            
            # Format and split
            dataset = formatter.format_and_split(scenarios)
            logger.info("✓ Data formatted and split (train/val/test)")
            
            # Tokenize
            tokenized = formatter.prepare_tokenized_dataset(dataset, max_seq_length=4096)
            logger.info("✓ Data tokenized")
            
            # Save to disk for other ranks
            os.makedirs(config["tokenized_cache"], exist_ok=True)
            tokenized.save_to_disk(config["tokenized_cache"])
            logger.info(f"✓ Tokenized dataset saved to {config['tokenized_cache']}")

        except Exception as e:
            logger.error(f"Failed in formatting/tokenization: {e}")
            raise

    # ====== WAIT FOR RANK 0 ======
    if world_size > 1:
        dist.barrier()
        logger.info(f"Rank {rank}: Barrier reached after rank 0 tokenization")

    # ====== STAGE 3: LOAD TOKENIZED DATA (ALL RANKS) ======
    logger.info(f"\n[STAGE 3] Rank {rank} loading tokenized dataset...")
    try:
        tokenized = load_from_disk(config["tokenized_cache"])
        logger.info(f"✓ Rank {rank} loaded tokenized dataset")
        logger.info(f"  Train samples: {len(tokenized['train'])}")
        logger.info(f"  Val samples: {len(tokenized['validation'])}")
        logger.info(f"  Test samples: {len(tokenized['test'])}")
    except Exception as e:
        logger.error(f"Rank {rank} failed to load tokenized dataset: {e}")
        raise

    # ====== WAIT FOR ALL RANKS ======
    if world_size > 1:
        dist.barrier()

    # ====== STAGE 4: TRAIN MODEL (ALL RANKS) ======
    if rank == 0:
        logger.info(f"\n[STAGE 4] Training model...")
        logger.info(f"Using FSDP to shard model across {world_size} GPUs")
    
    try:
        trainer = ModelTrainer(
            model_name=config["mistral_model_name"],
            output_dir=config["mistral_model_output"]
        )
        trainer.load_model()
        logger.info(f"Rank {rank}: Model loaded")
        
        # Train with FSDP (set via accelerate config)
        trainer.train(
            train_ds=tokenized["train"],
            val_ds=tokenized["validation"],
            num_epochs=config["epochs"],
            lr=config["lr"],
            batch_size=config["batch"],
            grad_accum=config["grad_accum"]
        )
        
    except Exception as e:
        logger.error(f"Rank {rank}: Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ====== STAGE 5: EVALUATE (RANK 0 ONLY) ======
    if rank == 0:
        logger.info(f"\n[STAGE 5] Evaluating model...")
        try:
            # Reload scenarios for evaluation
            loader = DataLoader(config.get('data_dir', '/home/bsindala/projects/datasets/clinical_access_control_scenarios_1M.csv'))
            scenarios = loader.load()

            gen = PolicyGenerator(config.get("mistral_model_output", './mistral7b_model_v3'))

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
            
            # Full evaluation with comprehensive metrics
            logger.info(f"Running full evaluation on {config['eval_size']} scenarios...")
            evaluator = ModelEvaluator(gen)
            metrics = evaluator.evaluate(scenarios, sample_size=config['eval_size'])
            
            # Print detailed report
            evaluator.print_detailed_report(metrics)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()

    # ====== CLEANUP ======
    if world_size > 1:
        dist.destroy_process_group()
        logger.info("Destroyed process group")

    logger.info("\n✓ Pipeline complete!")

if __name__ == "__main__":
    main()