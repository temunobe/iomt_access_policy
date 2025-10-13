# iomt_policy_generation.py

import logging
import os
import torch
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_trainer import ModelTrainer
from policy_generator import PolicyGenerator
from evaluator import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def setup_distributed():
    """Initialize distributed training environment"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Process info - rank={rank}, local_rank={local_rank}, world_size={world_size}")
    
    if world_size > 1:
        try:
            torch.distributed.init_process_group(
                backend="nccl",
                timeout=torch.distributed.timedelta(minutes=30)
            )
            logger.info(f"✓ Initialized distributed training: rank={rank}, local_rank={local_rank}, world_size={world_size}")
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    torch.cuda.set_device(local_rank)
    logger.info(f"Process rank={rank} using GPU device_id={local_rank}")
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = setup_distributed()
    
    # Only rank 0 logs fully
    if rank != 0:
        logging.getLogger().setLevel(logging.WARNING)
    
    config = {
        "dataset_csv": "/data/user/bsindala/PhD/Research/DataSets/clinical_access_control_scenarios.csv",
        "model_name": "/data/user/bsindala/PhD/Research/LLM_models/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "model_output": "./llama4_iomt_model",
        "epochs": 3,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "grad_accumulation": 32,
        "eval_sample_size": 50,
    }
    
    if rank == 0:
        logger.info("="*60)
        logger.info("IOMT Access Policy Generation - Distributed Training")
        logger.info("="*60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Config: {config}")
    
    if rank == 0:
        logger.info("1/5 LOADING DATASET")
    scenarios = None
    if rank == 0:
        loader = DataLoader(config["dataset_csv"])
        scenarios = loader.load()
    
    # Broadcast scenarios to all ranks (simplified - in production use proper serialization)
    if world_size > 1:
        torch.distributed.barrier()
    
    if rank == 0:
        logger.info("2/5 FORMATTING DATA")
    formatter = DataFormatter(config["model_name"])
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
    
    trainer_obj = ModelTrainer(config["model_name"], config["model_output"])
    trainer_obj.load_model()
    
    if rank == 0:
        trainer_obj.train(
            tokenized_dataset["train"],
            tokenized_dataset["validation"],
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            grad_accum=config["grad_accumulation"]
        )
        
        logger.info("4/5 GENERATING SAMPLE POLICIES")
        generator = PolicyGenerator(config["model_output"])
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
        metrics = evaluator.evaluate(scenarios, sample_size=config["eval_sample_size"])
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETE - SUMMARY")
        logger.info("="*60)
        logger.info(f"Scenarios processed: {len(scenarios)}")
        logger.info(f"Model saved to: {config['model_output']}")
        logger.info(f"Training epochs: {config['epochs']}")
        logger.info(f"Evaluation metrics: {metrics}")
        logger.info("="*60)
    else:
        # Other ranks participate in training but don't generate policies
        trainer_obj.train(
            None,  # These will be loaded by distributed data loading
            None,
            num_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            grad_accum=config["grad_accumulation"]
        )
    
    if world_size > 1:
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()