# iomt_policy_generation.py

import logging
import os
import system
import traceback
import datetime
import torch
import subprocess
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_trainer import ModelTrainer
from policy_generator import PolicyGenerator
from evaluator import ModelEvaluator
from config import cfg
import torch.distributed as dist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_distributed(timeout_minutes: int = 30):
    """
    Minimal, torchrun-compatible distributed init.
    - Uses init_method="env://" so torchrun --standalone works without explicit MASTER_ADDR/PORT handling.
    - Sets CUDA device from LOCAL_RANK before init so NCCL knows the local GPU mapping.
    Returns: (rank, local_rank, world_size)
    """
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))

    # If distributed support isn't available, return early.
    if not torch.distributed.is_available():
        return rank, local_rank, world_size

    # Set CUDA device early so NCCL picks the right local device.
    if torch.cuda.is_available():
        ngpus = torch.cuda.device_count()
        device_id = local_rank % max(1, ngpus)
        try:
            torch.cuda.set_device(device_id)
            print(f"[rank {rank}] set CUDA device to {device_id}")
        except Exception as e:
            print(f"[rank {rank}] warning: failed to set CUDA device {device_id}: {e}")

    timeout = datetime.timedelta(minutes=timeout_minutes)

    if not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        # Use the environment-based init (torchrun --standalone or torchrun launched by Slurm sets the env)
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )

    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = setup_distributed()
    
    # Only rank 0 logs fully
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    
    if rank == 0:
        logger.info("="*60)
        logger.info("IOMT Access Policy Generation - Distributed Training")
        logger.info("="*60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {cfg}")
    
    if rank == 0:
        logger.info("1/5 LOADING DATASET")
    scenarios = None
    if rank == 0:
        loader = DataLoader(cfg["dataset_csv"])
        scenarios = loader.load()
    
    # Broadcast scenarios to all ranks (simplified - in production use proper serialization)
    if world_size > 1:
        torch.distributed.barrier()
    
    if rank == 0:
        logger.info("2/5 FORMATTING DATA")
    formatter = DataFormatter(cfg["model_name"])
    if rank == 0:
        dataset = formatter.format_and_split(scenarios)
    else:
        # Other ranks wait
        dataset = None
    
    if rank == 0:
        tokenized_dataset = formatter.prepare_tokenized_dataset(
            dataset, 
            max_seq_length=4096,
            num_proc=1
        )
    else:
        tokenized_dataset = None
    
    if world_size > 1:
        torch.distributed.barrier()
    
    if rank == 0:
        logger.info("3/5 TRAINING MODEL")

    trainer_obj = ModelTrainer(cfg["model_name"], cfg["model_output"])
    trainer_obj.load_model()
    
    if rank == 0:
        trainer_obj.train(
            tokenized_dataset["train"],
            tokenized_dataset["validation"],
            num_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accumulation"]
        )
        
        logger.info("4/5 GENERATING SAMPLE POLICIES")
        generator = PolicyGenerator(cfg["model_output"])
        for scenario in scenarios[:3]:
            logger.info(f"Generating policy for: {scenario.scenario_id}")
            device_metadata = {
                "device_type": scenario.device_type,
                "criticality": scenario.criticality
            }
            policy = generator.generate(scenario.description, device_metadata)
            validation = generator.validate_policy(policy)
            logger.info(f"Valid XML: {validation['is_valid_xml']}, Has Target: {validation['has_target']}, Has Rules: {validation['has_rules']}")
        
        logger.info("5/5 EVALUATING MODEL")
        evaluator = ModelEvaluator(generator)
        metrics = evaluator.evaluate(scenarios, sample_size=cfg["eval_sample_size"])
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"Scenarios processed: {len(scenarios)}")
        logger.info(f"Model saved to: {cfg['model_output']}")
        logger.info(f"Training epochs: {cfg['epochs']}")
        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("="*60)
    else:
        # Other ranks participate in training but don't generate policies
        trainer_obj.train(
            None,  # These will be loaded by distributed data loading
            None,
            num_epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            grad_accum=cfg["grad_accumulation"]
        )
    
    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # Try to determine rank early so logs are per-process
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))

    try:
        main()
    except Exception:
        # Write full traceback to a per-rank file for easy debugging
        log_path = f"/tmp/iomt_failure_rank{rank}.log"
        with open(log_path, "w") as f:
            traceback.print_exc(file=f)
        # Also print to stderr/stdout so Slurm output captures it
        traceback.print_exc()
        # Re-raise so torchrun sees the child failed (keeps same behavior)
        raise