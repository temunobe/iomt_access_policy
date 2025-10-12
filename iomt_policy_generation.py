# iomt_policy_generation.py

import logging
import os
from data_loader import DataLoader
from data_formatter import DataFormatter
from model_trainer import ModelTrainer
from policy_generator import PolicyGenerator
from evaluator import ModelEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = 'true'

def main():
    config = {
        # <-- UPDATE THESE PATHS TO YOUR ENVIRONMENT
        "dataset_csv": "/data/user/bsindala/PhD/Research/DataSets/clinical_access_control_scenarios.csv",
        "model_name": "/data/user/bsindala/PhD/Research/LLM_models/meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "model_output": "./llama4_iomt_model",
        "epochs": 1, #3, Testing
        "learning_rate": 1e-5,
        "batch_size": 1, # 4, For memory efficiency
        "grad_accumulation": 16, # 8, Smaller batch compensation
        "eval_sample_size": 100, #None
    }

    logger.info("1/5 LOADING DATASET")
    loader = DataLoader(config["dataset_csv"])
    scenarios = loader.load()

    logger.info("2/5 FORMATTING DATA")
    formatter = DataFormatter(config["model_name"])
    dataset = formatter.format_and_split(scenarios)
    tokenized_dataset = formatter.prepare_tokenized_dataset(dataset, max_seq_length=8192, num_proc=1)

    logger.info("3/5 TRAINING MODEL")
    trainer_obj = ModelTrainer(config["model_name"], config["model_output"])
    trainer_obj.load_model()
    trainer_obj.train(
        tokenized_dataset["train"],
        tokenized_dataset["validation"],
        num_epochs=config["epochs"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        grad_accum= config["grad_accumulation"]
    )

    logger.info("4/5 GENERATING SAMPLE POLICIES")
    generator = PolicyGenerator(config["model_output"])
    for scenario in scenarios[:3]:
        logger.info(f"Generating policy for: {scenario.scenario_id}")
        device_metadata = {"device_type": scenario.device_type, "criticality": scenario.criticality}
        policy = generator.generate(scenario.description, device_metadata)
        validation = generator.validate_policy(policy)
        logger.info(f"Valid XML: {validation['is_valid_xml']}, Has Target: {validation['has_target']}, Has Rules: {validation['has_rules']}")

    logger.info("5/5 EVALUATING MODEL")
    evaluator = ModelEvaluator(generator)
    metrics = evaluator.evaluate(scenarios, sample_size=config["eval_sample_size"])

    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info(f"Scenarios processed: {len(scenarios)}")
    logger.info(f"Model saved to: {config['model_output']}")
    logger.info(f"Training epochs: {config['epochs']}")
    logger.info(f"Evaluation metrics: {metrics}")

if __name__ == "__main__":
    main()