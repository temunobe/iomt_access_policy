# iomt_policy_generation.py

import logging
import os
#import system
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
# New name for this config was introduced in newer PyTorch; set both so older/newer runtimes behave the same.
if "PYTORCH_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]

def setup_distributed(timeout_minutes: int = 30):
    """
    Robust distributed init that works with torchrun, Slurm, or standalone execution.
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

    # Check if already initialized (can happen in some environments)
    if torch.distributed.is_initialized():
        return rank, local_rank, world_size

    # For single-process execution (world_size == 1), we still initialize
    # a (local) process group. Some libraries (accelerate/transformers)
    # call torch.distributed.get_world_size() and expect a default
    # process group to exist even in single-process mode. Initializing
    # a single-process group avoids "Default process group has not been
    # initialized" errors while being a no-op for multi-process runs
    # where the group is already created by torchrun.
    # Note: if the process group is already initialized we return early above.

    # Set MASTER_ADDR and MASTER_PORT if not set (for single-node multi-GPU)
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        print(f"[rank {rank}] MASTER_ADDR not set, defaulting to localhost")
    
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
        print(f"[rank {rank}] MASTER_PORT not set, defaulting to 29500")

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    
    # If single-process, try to create a local process group so libraries
    # that expect a default group (accelerate/transformers) won't fail.
    if world_size == 1:
        try:
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=1,
                rank=rank,
                timeout=timeout,
            )
            print(f"[rank {rank}] Initialized single-process distributed group with {backend}")
            return rank, local_rank, world_size
        except Exception as e:
            print(f"[rank {rank}] env:// single-process init failed: {e}; trying file-based init")
            # Try a file-based init as a last resort for single-node single-process
            try:
                import tempfile
                tmpf = tempfile.mktemp()
                file_url = f"file://{tmpf}"
                torch.distributed.init_process_group(
                    backend=backend,
                    init_method=file_url,
                    world_size=1,
                    rank=rank,
                    timeout=timeout,
                )
                print(f"[rank {rank}] Initialized single-process distributed group with file:// init and {backend}")
                return rank, local_rank, world_size
            except Exception as e2:
                print(f"[rank {rank}] file:// single-process init failed: {e2}; continuing without process group")

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=timeout,
        )
        print(f"[rank {rank}] Successfully initialized distributed process group with {backend}")
    except Exception as e:
        print(f"[rank {rank}] Failed to initialize distributed: {e}")
        # Fall back to non-distributed mode
        return rank, local_rank, 1

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
    
    # Load and format data on rank 0, then broadcast or save/load
    if rank == 0:
        logger.info("1/5 LOADING DATASET")
        loader = DataLoader(cfg["dataset_csv"])
        scenarios = loader.load()
        
        logger.info("2/5 FORMATTING DATA")
        formatter = DataFormatter(cfg["model_name"])
        dataset = formatter.format_and_split(scenarios)
        
        tokenized_dataset = formatter.prepare_tokenized_dataset(
            dataset, 
            max_seq_length=4096,
            num_proc=1
        )
        
        # Save tokenized dataset to disk for other ranks to load
        if world_size > 1:
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="iomt_dataset_")
            tokenized_dataset.save_to_disk(temp_dir)
            dataset_path = temp_dir
            logger.info(f"Saved tokenized dataset to {dataset_path}")
        else:
            dataset_path = None
    else:
        scenarios = None
        tokenized_dataset = None
        dataset_path = None
    
    # Broadcast dataset path to all ranks
    if world_size > 1:
        import torch.distributed as dist
        
        # Rank 0 broadcasts the path length and then the path
        if rank == 0:
            path_bytes = dataset_path.encode('utf-8')
            path_len = torch.tensor([len(path_bytes)], dtype=torch.long).cuda()
        else:
            path_len = torch.tensor([0], dtype=torch.long).cuda()
        
        dist.broadcast(path_len, src=0)
        
        if rank == 0:
            path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8).cuda()
        else:
            path_tensor = torch.zeros(path_len.item(), dtype=torch.uint8).cuda()
        
        dist.broadcast(path_tensor, src=0)
        
        if rank != 0:
            dataset_path = bytes(path_tensor.cpu().numpy()).decode('utf-8')
            from datasets import load_from_disk
            tokenized_dataset = load_from_disk(dataset_path)
            logger.info(f"[rank {rank}] Loaded tokenized dataset from {dataset_path}")
        
        dist.barrier()
    
    if rank == 0:
        logger.info("3/5 TRAINING MODEL")

    trainer_obj = ModelTrainer(cfg["model_name"], cfg["model_output"])
    trainer_obj.load_model()
    
    # All ranks participate in training
    trainer_obj.train(
        tokenized_dataset["train"] if tokenized_dataset else None,
        tokenized_dataset["validation"] if tokenized_dataset else None,
        num_epochs=cfg["epochs"],
        learning_rate=cfg["lr"],
        batch_size=cfg["batch"],
        grad_accum=cfg["grad_accum"]
    )
    
    # Only rank 0 does policy generation and evaluation
    if rank == 0:
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
        
        # Cleanup temporary dataset directory
        if world_size > 1 and dataset_path:
            import shutil
            shutil.rmtree(dataset_path)
    
    if world_size > 1:
        torch.distributed.destroy_process_group()

    # If we initialized a single-process group earlier (world_size==1),
    # try to destroy it to avoid resource-leak warnings. Guard with
    # is_initialized check so this is a no-op when nothing was created.
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        # Best-effort cleanup; ignore failures during shutdown
        pass

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