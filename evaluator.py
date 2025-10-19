# evaluator.py

import logging
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xml.etree.ElementTree import ET
import re

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, generator):
        self.generator = generator

    def evaluate(self, scenarios, sample_size=None):
        """Comprehensive evaluation with multiple metrics"""
        test = scenarios[:sample_size] if sample_size else scenarios
        
        # Basic structure metrics
        stats = {
            "valid": 0,
            "target": 0, 
            "rules": 0,
            "times": []
        }
        
        # Binary classification arrays for sklearn metrics
        y_true_valid_xml = []
        y_pred_valid_xml = []
        
        y_true_has_target = []
        y_pred_has_target = []
        
        y_true_has_rules = []
        y_pred_has_rules = []
        
        # Element-level matching
        element_matches = []
        
        # Semantic similarity
        semantic_scores = []
        
        logger.info(f"Evaluating {len(test)} scenarios...")
        
        for s in tqdm(test, desc="Evaluating"):
            t0 = time.time()
            
            # Generate policy
            generated_policy = self.generator.generate(
                s.description,
                {"device_type": s.device_type, "criticality": s.criticality}
            )
            stats["times"].append(time.time() - t0)
            
            # Validate generated policy
            gen_val = self.generator.validate_policy(generated_policy)
            
            # Validate ground truth policy
            gt_val = self.generator.validate_policy(s.access_policy)
            
            # Basic structure metrics
            if gen_val["is_valid_xml"]:
                stats["valid"] += 1
            if gen_val["has_target"]:
                stats["target"] += 1
            if gen_val["has_rules"]:
                stats["rules"] += 1
            
            # Binary classification: Valid XML
            y_true_valid_xml.append(1 if gt_val["is_valid_xml"] else 0)
            y_pred_valid_xml.append(1 if gen_val["is_valid_xml"] else 0)
            
            # Binary classification: Has Target
            y_true_has_target.append(1 if gt_val["has_target"] else 0)
            y_pred_has_target.append(1 if gen_val["has_target"] else 0)
            
            # Binary classification: Has Rules
            y_true_has_rules.append(1 if gt_val["has_rules"] else 0)
            y_pred_has_rules.append(1 if gen_val["has_rules"] else 0)
            
            # Element-level comparison (if both are valid XML)
            if gen_val["is_valid_xml"] and gt_val["is_valid_xml"]:
                element_score = self._compare_xml_elements(generated_policy, s.access_policy)
                element_matches.append(element_score)
            else:
                element_matches.append(0.0)
            
            # Semantic similarity
            semantic_score = self._semantic_similarity(generated_policy, s.access_policy)
            semantic_scores.append(semantic_score)
        
        n = len(test)
        
        # Calculate all metrics
        metrics = {
            # Basic rates
            "total": n,
            "valid_xml_rate": stats["valid"] / n * 100 if n else 0,
            "has_target_rate": stats["target"] / n * 100 if n else 0,
            "has_rules_rate": stats["rules"] / n * 100 if n else 0,
            
            # Timing
            "avg_time": np.mean(stats["times"]) if stats["times"] else 0,
            "median_time": np.median(stats["times"]) if stats["times"] else 0,
            
            # Classification metrics for Valid XML
            "valid_xml_precision": precision_score(y_true_valid_xml, y_pred_valid_xml, zero_division=0) * 100,
            "valid_xml_recall": recall_score(y_true_valid_xml, y_pred_valid_xml, zero_division=0) * 100,
            "valid_xml_f1": f1_score(y_true_valid_xml, y_pred_valid_xml, zero_division=0) * 100,
            "valid_xml_accuracy": accuracy_score(y_true_valid_xml, y_pred_valid_xml) * 100,
            
            # Classification metrics for Has Target
            "has_target_precision": precision_score(y_true_has_target, y_pred_has_target, zero_division=0) * 100,
            "has_target_recall": recall_score(y_true_has_target, y_pred_has_target, zero_division=0) * 100,
            "has_target_f1": f1_score(y_true_has_target, y_pred_has_target, zero_division=0) * 100,
            "has_target_accuracy": accuracy_score(y_true_has_target, y_pred_has_target) * 100,
            
            # Classification metrics for Has Rules
            "has_rules_precision": precision_score(y_true_has_rules, y_pred_has_rules, zero_division=0) * 100,
            "has_rules_recall": recall_score(y_true_has_rules, y_pred_has_rules, zero_division=0) * 100,
            "has_rules_f1": f1_score(y_true_has_rules, y_pred_has_rules, zero_division=0) * 100,
            "has_rules_accuracy": accuracy_score(y_true_has_rules, y_pred_has_rules) * 100,
            
            # Element-level similarity
            "avg_element_similarity": np.mean(element_matches) * 100 if element_matches else 0,
            "median_element_similarity": np.median(element_matches) * 100 if element_matches else 0,
            
            # Semantic similarity
            "avg_semantic_similarity": np.mean(semantic_scores) * 100 if semantic_scores else 0,
            "median_semantic_similarity": np.median(semantic_scores) * 100 if semantic_scores else 0,
        }
        
        return metrics
    
    def _compare_xml_elements(self, generated_xml: str, ground_truth_xml: str) -> float:
        """
        Compare XML elements between generated and ground truth.
        Returns similarity score between 0 and 1.
        """
        try:
            gen_root = ET.fromstring(generated_xml)
            gt_root = ET.fromstring(ground_truth_xml)
            
            # Extract all element tags
            gen_elements = self._get_all_tags(gen_root)
            gt_elements = self._get_all_tags(gt_root)
            
            if not gt_elements:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(set(gen_elements) & set(gt_elements))
            union = len(set(gen_elements) | set(gt_elements))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"XML comparison failed: {e}")
            return 0.0
    
    def _get_all_tags(self, root) -> list:
        """Recursively get all XML tags from a tree"""
        tags = [root.tag.split('}')[-1]]  # Remove namespace
        for child in root:
            tags.extend(self._get_all_tags(child))
        return tags
    
    def _semantic_similarity(self, generated: str, ground_truth: str) -> float:
        """
        Simple semantic similarity based on key XACML terms.
        Returns score between 0 and 1.
        """
        # Key XACML terms to look for
        key_terms = [
            'Policy', 'PolicySet', 'Rule', 'Target', 'Condition',
            'Subject', 'Resource', 'Action', 'Environment',
            'Permit', 'Deny', 'Match', 'Apply', 'AttributeValue'
        ]
        
        gen_lower = generated.lower()
        gt_lower = ground_truth.lower()
        
        # Count matching terms
        matches = 0
        total = 0
        
        for term in key_terms:
            term_lower = term.lower()
            gt_has = term_lower in gt_lower
            gen_has = term_lower in gen_lower
            
            if gt_has:
                total += 1
                if gen_has:
                    matches += 1
        
        return matches / total if total > 0 else 0.0
    
    def print_detailed_report(self, metrics: dict):
        """Print a comprehensive evaluation report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("="*80)
        
        print(f"\nDataset Size: {metrics['total']} scenarios")
        print(f"Avg Generation Time: {metrics['avg_time']:.2f}s")
        print(f"Median Generation Time: {metrics['median_time']:.2f}s")
        
        print("\n" + "-"*80)
        print("STRUCTURAL METRICS (Basic Policy Quality)")
        print("-"*80)
        print(f"Valid XML Rate:    {metrics['valid_xml_rate']:6.2f}%")
        print(f"Has Target Rate:   {metrics['has_target_rate']:6.2f}%")
        print(f"Has Rules Rate:    {metrics['has_rules_rate']:6.2f}%")
        
        print("\n" + "-"*80)
        print("VALID XML CLASSIFICATION METRICS")
        print("-"*80)
        print(f"Precision:  {metrics['valid_xml_precision']:6.2f}%")
        print(f"Recall:     {metrics['valid_xml_recall']:6.2f}%")
        print(f"F1-Score:   {metrics['valid_xml_f1']:6.2f}%")
        print(f"Accuracy:   {metrics['valid_xml_accuracy']:6.2f}%")
        
        print("\n" + "-"*80)
        print("HAS TARGET CLASSIFICATION METRICS")
        print("-"*80)
        print(f"Precision:  {metrics['has_target_precision']:6.2f}%")
        print(f"Recall:     {metrics['has_target_recall']:6.2f}%")
        print(f"F1-Score:   {metrics['has_target_f1']:6.2f}%")
        print(f"Accuracy:   {metrics['has_target_accuracy']:6.2f}%")
        
        print("\n" + "-"*80)
        print("HAS RULES CLASSIFICATION METRICS")
        print("-"*80)
        print(f"Precision:  {metrics['has_rules_precision']:6.2f}%")
        print(f"Recall:     {metrics['has_rules_recall']:6.2f}%")
        print(f"F1-Score:   {metrics['has_rules_f1']:6.2f}%")
        print(f"Accuracy:   {metrics['has_rules_accuracy']:6.2f}%")
        
        print("\n" + "-"*80)
        print("SIMILARITY METRICS (Content Quality)")
        print("-"*80)
        print(f"Avg Element Similarity:    {metrics['avg_element_similarity']:6.2f}%")
        print(f"Median Element Similarity: {metrics['median_element_similarity']:6.2f}%")
        print(f"Avg Semantic Similarity:   {metrics['avg_semantic_similarity']:6.2f}%")
        print(f"Median Semantic Similarity:{metrics['median_semantic_similarity']:6.2f}%")
        
        print("\n" + "-"*80)
        print("OVERALL ASSESSMENT")
        print("-"*80)
        
        # Calculate overall score
        overall = (
            metrics['valid_xml_f1'] * 0.3 +
            metrics['has_target_f1'] * 0.2 +
            metrics['has_rules_f1'] * 0.2 +
            metrics['avg_element_similarity'] * 0.15 +
            metrics['avg_semantic_similarity'] * 0.15
        )
        
        print(f"Overall Quality Score: {overall:.2f}%")
        
        if overall >= 85:
            print("Assessment: EXCELLENT - Production ready")
        elif overall >= 70:
            print("Assessment: GOOD - Minor improvements needed")
        elif overall >= 50:
            print("Assessment: FAIR - Needs refinement")
        else:
            print("Assessment: POOR - Requires significant retraining")
        
        print("="*80 + "\n")