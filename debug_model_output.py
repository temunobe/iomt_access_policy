#!/usr/bin/env python3
# debug_model_output.py - See exactly what the model generates

import os
import sys
from policy_generator import PolicyGenerator
from data_loader import DataLoader
from config import config
import json

def main():
    print("="*80)
    print("MODEL OUTPUT DIAGNOSTIC")
    print("="*80)
    
    # Load test scenarios
    print("\n1. Loading test scenarios...")
    loader = DataLoader(config.get('data_dir', '/home/bsindala/projects/datasets/clinical_access_control_scenarios_1M.csv'))
    scenarios = loader.load()
    
    print(f"   ✓ Loaded {len(scenarios)} scenarios")
    
    # Load model
    print("\n2. Loading trained model...")
    model_path = config.get("mistral_model_output", "./mistral7b_model_v3")
    
    if not os.path.exists(model_path):
        print(f"   ✗ ERROR: Model not found at {model_path}")
        return
    
    gen = PolicyGenerator(model_path)
    print(f"   ✓ Model loaded from {model_path}")
    
    # Check training loss
    print("\n3. Checking training logs...")
    log_file = os.path.join(model_path, "trainer_state.json")
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            state = json.load(f)
        if 'log_history' in state:
            losses = [entry.get('loss') for entry in state['log_history'] if 'loss' in entry]
            if losses:
                print(f"   Initial loss: {losses[0]:.4f}")
                print(f"   Final loss: {losses[-1]:.4f}")
                print(f"   Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")
                if losses[-1] > 3.0:
                    print("   ⚠️  WARNING: Final loss is high - model may not have learned well")
            print(f"   Total training steps: {len(losses)}")
    
    # Test on 3 different scenarios
    print("\n4. Testing generation on 3 scenarios...")
    for i, test_scenario in enumerate(scenarios[:3], 1):
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}")
        print(f"{'='*80}")
        
        print(f"\nScenario ID: {test_scenario.scenario_id}")
        print(f"Device: {test_scenario.device_type}")
        print(f"Criticality: {test_scenario.criticality}")
        print(f"Description: {test_scenario.description[:150]}...")
        
        print("\n--- GROUND TRUTH (First 500 chars) ---")
        print(test_scenario.access_policy[:500])
        print("...")
        
        print("\n--- GENERATED OUTPUT (First 1000 chars) ---")
        output = gen.generate(
            test_scenario.description,
            {"device_type": test_scenario.device_type, "criticality": test_scenario.criticality}
        )
        print(output[:1000])
        if len(output) > 1000:
            print(f"... (truncated, total length: {len(output)} chars)")
        
        # Check what's in the output
        print("\n--- OUTPUT ANALYSIS ---")
        print(f"Total length: {len(output)} characters")
        print(f"Contains '<Policy': {'✓' if '<Policy' in output else '✗'}")
        print(f"Contains '<Rule': {'✓' if '<Rule' in output else '✗'}")
        print(f"Contains '<Target': {'✓' if '<Target' in output else '✗'}")
        print(f"Contains 'xml': {'✓' if 'xml' in output.lower() else '✗'}")
        print(f"Contains '```': {'✓' if '```' in output else '✗'}")
        print(f"Starts with '<': {'✓' if output.strip().startswith('<') else '✗'}")
        
        # Try to extract XML if it's wrapped
        if '```xml' in output:
            print("\n⚠️  XML is wrapped in markdown code block!")
            import re
            xml_match = re.search(r'```xml\s*(.*?)\s*```', output, re.DOTALL)
            if xml_match:
                extracted = xml_match.group(1)
                print(f"Extracted XML (first 500 chars):")
                print(extracted[:500])
        elif '```' in output:
            print("\n⚠️  Output contains code blocks")
            
        validation = gen.validate_policy(output)
        print(f"\nValidation: Valid={validation['is_valid_xml']}, Target={validation['has_target']}, Rules={validation['has_rules']}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS")
    print("="*80)
    
    # Provide specific recommendations
    if '```xml' in output or '```' in output:
        print("\n❌ ISSUE: Model is wrapping XML in markdown code blocks")
        print("   FIX: Update policy_generator.py to extract XML from code blocks")
    elif 'xml' in output.lower() and '<' not in output[:100]:
        print("\n❌ ISSUE: Model is describing XML instead of generating it")
        print("   FIX: Retrain with better prompt formatting")
    elif losses[-1] > 3.0:
        print("\n❌ ISSUE: Training loss too high - model didn't learn")
        print("   FIX: Train for more epochs (3-5 instead of 1)")
    else:
        print("\n❓ ISSUE: Model output is unexpected format")
        print("   Check the generated output above to understand the format")

if __name__ == "__main__":
    main()