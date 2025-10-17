# evaluator.py
import time
import numpy as np
import logging
from tqdm import tqdm
logger = logging.getLogger(__name__)
class ModelEvaluator:
    """Evaluate model"""
    def __init__(self, generator):
        self.generator = generator
    def evaluate(self, test_scenarios, sample_size=None):
        logger.info("EVALUATING MODEL")
        if sample_size:
            test_scenarios = test_scenarios[:sample_size]
        results = {"valid_xml": 0, "with_target": 0, "with_rules": 0, "generation_times": []}
        for scenario in tqdm(test_scenarios, desc="Evaluating"):
            start = time.time()
            device_metadata = {"device_type": scenario.device_type, "criticality": scenario.criticality}
            generated = self.generator.generate(scenario.description, device_metadata)
            generation_time = time.time() - start
            results["generation_times"].append(generation_time)
            validation = self.generator.validate_policy(generated)
            if validation["is_valid_xml"]:
                results["valid_xml"] += 1
            if validation["has_target"]:
                results["with_target"] += 1
            if validation["has_rules"]:
                results["with_rules"] += 1
        total = len(test_scenarios)
        metrics = {
            "total_evaluated": total,
            "valid_xml_rate": results["valid_xml"] / total * 100 if total else 0.0,
            "has_target_rate": results["with_target"] / total * 100 if total else 0.0,
            "has_rules_rate": results["with_rules"] / total * 100 if total else 0.0,
            "avg_generation_time": np.mean(results["generation_times"]) if results["generation_times"] else 0.0,
            "median_generation_time": np.median(results["generation_times"]) if results["generation_times"] else 0.0,
        }
        logger.info(f"Valid XML: {metrics['valid_xml_rate']:.1f}%")
        logger.info(f"Has Target: {metrics['has_target_rate']:.1f}%")
        logger.info(f"Has Rules: {metrics['has_rules_rate']:.1f}%")
        logger.info(f"Avg Time: {metrics['avg_generation_time']:.2f}s")
        logger.info(f"Median Time: {metrics['median_generation_time']:.2f}s")
        return metrics