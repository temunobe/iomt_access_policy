# evaluator.py

import logging
import time
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, generator):
        self.generator = generator

    def evaluate(self, scenarios, sample_size=None):
        test = scenarios[:sample_size] if sample_size else scenarios
        stats = {"valid": 0, "target": 0, "rules": 0, "times": []}

        for s in tqdm(test, desc="Evaluating"):
            t0 = time.time()
            policy = self.generator.generate(s.description, {"device_type": s.device_type, "criticality": s.criticality})
            stats["times"].append(time.time() - t0)
            val = self.generator.validate_policy(policy)
            if val["is_valid_xml"]: stats["valid"] += 1
            if val["has_target"]: stats["target"] += 1
            if val["has_rules"]: stats["rules"] += 1

        n = len(test)
        return {
            "total": n,
            "valid_xml_rate": stats["valid"] / n * 100 if n else 0,
            "has_target_rate": stats["target"] / n * 100 if n else 0,
            "has_rules_rate": stats["rules"] / n * 100 if n else 0,
            "avg_time": np.mean(stats["times"]) if stats["times"] else 0,
            "median_time": np.median(stats["times"]) if stats["times"] else 0,
        }