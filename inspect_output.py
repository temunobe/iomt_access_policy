#!/usr/bin/env python3
# inspect_model_outputs.py - See what XML structure the model creates

import os
from policy_generator import PolicyGenerator
from data_loader import DataLoader
from config import config
import xml.etree.ElementTree as ET

def print_xml_structure(xml_str: str, label: str):
    """Print the structure of an XML document"""
    print(f"\n{label}:")
    print("-" * 80)
    try:
        root = ET.fromstring(xml_str)
        print(f"Root tag: <{root.tag}>")
        print(f"Root attributes: {root.attrib}")
        
        # Print all child elements
        def print_tree(element, indent=0):
            tag = element.tag.split('}')[-1]  # Remove namespace
            print("  " * indent + f"<{tag}> {element.attrib}")
            for child in element:
                print_tree(child, indent + 1)
        
        print("\nStructure:")
        print_tree(root)
        
        # Check for specific elements (handle namespaces)
        has_target = False
        has_rule = False
        num_rules = 0
        
        for elem in root.iter():
            tag = elem.tag.split('}')[-1]  # Remove namespace
            if tag == 'Target':
                has_target = True
            elif tag == 'Rule':
                has_rule = True
                num_rules += 1
        
        print(f"\nContains <Target>: {has_target}")
        print(f"Contains <Rule>: {has_rule}")
        print(f"Number of <Rule> elements: {num_rules}")
        
    except ET.ParseError as e:
        print(f"Parse error: {e}")
        print(f"First 500 chars:\n{xml_str[:500]}")

def main():
    print("="*80)
    print("INSPECTING MODEL OUTPUTS")
    print("="*80)
    
    # Load scenarios
    loader = DataLoader(config.get('data_dir', '/home/bsindala/projects/datasets/clinical_access_control_scenarios_1M.csv'))
    scenarios = loader.load()
    
    # Load generator
    model_path = config.get("mistral_model_output", "./mistral7b_model_v3")
    gen = PolicyGenerator(model_path)
    
    # Test 3 scenarios
    for i, scenario in enumerate(scenarios[:3], 1):
        print(f"\n{'='*80}")
        print(f"SCENARIO {i}")
        print(f"{'='*80}")
        print(f"Device: {scenario.device_type}")
        print(f"Criticality: {scenario.criticality}")
        print(f"Description: {scenario.description[:150]}...")
        
        # Generate
        generated = gen.generate(
            scenario.description,
            {"device_type": scenario.device_type, "criticality": scenario.criticality}
        )
        
        # Show ground truth structure
        print_xml_structure(scenario.access_policy, "GROUND TRUTH STRUCTURE")
        
        # Show generated structure  
        print_xml_structure(generated, "GENERATED STRUCTURE")
        
        # Show first 1000 chars of each
        print("\n" + "-"*80)
        print("GROUND TRUTH (first 1000 chars):")
        print("-"*80)
        print(scenario.access_policy[:1000])
        
        print("\n" + "-"*80)
        print("GENERATED (first 1000 chars):")
        print("-"*80)
        print(generated[:1000])
        
        if i < 3:
            input("\nPress Enter to see next scenario...")

if __name__ == "__main__":
    main()