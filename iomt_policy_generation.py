# iomt_policy_generation.py

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
            init_method="env://", timeout=datetime.timedelta(minutes=timeout_min)
        )
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = setup_distributed()
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)

    if rank == 0:
        logger.info("="*50)
        logger.info("IOMT Policy Generation - 2 GPU Training")
        logger.info(f"World size: {world_size}")
        logger.info("="*50)

    scenarios = None
    if rank == 0:
        logger.info("Loading dataset...")
        scenarios = DataLoader(cfg["dataset_csv"]).load()

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info("Formatting data...")
        fmt = DataFormatter(cfg["model_name"])
        dataset = fmt.format_and_split(scenarios)
        tokenized = fmt.prepare_tokenized_dataset(dataset, max_seq_length=4096)
    else:
        tokenized = None

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        logger.info("Training model...")
    trainer = ModelTrainer(model_name=cfg["model_name"], output_dir=cfg["model_output"])
    trainer.load_model()
    trainer.train(
        tokenized["train"] if rank == 0 else None,
        tokenized["validation"] if rank == 0 else None,
        num_epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch"], grad_accum=cfg["grad_accum"]
    )

    if rank == 0:
        logger.info("Generating policies...")
        gen = PolicyGenerator(cfg["model_output"])
        for s in scenarios[:3]:
            policy = gen.generate(s.description, {"device_type": s.device_type, "criticality": s.criticality})
            val = gen.validate_policy(policy)
            logger.info(f"Policy valid: {val['is_valid_xml']}, target: {val['has_target']}, rules: {val['has_rules']}")

        logger.info("Evaluating...")
        metrics = ModelEvaluator(gen).evaluate(scenarios, sample_size=cfg["eval_size"])
        logger.info(f"Valid XML: {metrics['valid_xml_rate']:.1f}%")
        logger.info(f"Has Target: {metrics['has_target_rate']:.1f}%")
        logger.info(f"Has Rules: {metrics['has_rules_rate']:.1f}%")
        logger.info(f"Avg Time: {metrics['avg_time']:.2f}s")
        logger.info("="*50)

    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()