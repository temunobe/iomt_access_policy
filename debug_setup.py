#!/usr/bin/env python3
# debug_setup.py - Pre-flight checks for Llama 4 Scout training

import torch
import sys
import os

def check_environment():
    """Run comprehensive environment checks"""
    
    print("="*70)
    print("LLAMA 4 SCOUT - ENVIRONMENT CHECK")
    print("="*70)
    
    # Check Python version
    print(f"\n✓ Python version: {sys.version}")
    
    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return False
    
    # Check GPUs
    gpu_count = torch.cuda.device_count()
    print(f"✓ GPU count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {total_mem:.1f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Check memory per GPU
    if gpu_count > 0:
        min_required_gb = 20  # Estimate for Llama 4 Scout 17B with 4-bit
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / (1024**3)
        
        if total_mem_gb < min_required_gb:
            print(f"\nWARNING: GPU memory may be insufficient!")
            print(f"  Required: ~{min_required_gb} GB per GPU")
            print(f"  Available: {total_mem_gb:.1f} GB")
            print("  Consider using more GPUs or 4-bit quantization")
    
    # Check transformers
    try:
        import transformers
        print(f"\n✓ Transformers version: {transformers.__version__}")
    except ImportError:
        print("\nERROR: transformers not installed!")
        return False
    
    # Check PEFT
    try:
        import peft
        print(f"✓ PEFT version: {peft.__version__}")
    except ImportError:
        print("\nERROR: peft not installed!")
        return False
    
    # Check bitsandbytes
    try:
        import bitsandbytes
        print(f"✓ bitsandbytes available: {bitsandbytes.__version__}")
    except ImportError:
        print("⚠ bitsandbytes not available (quantization disabled)")
    
    # Check accelerate
    try:
        import accelerate
        print(f"✓ Accelerate version: {accelerate.__version__}")
    except ImportError:
        print("\nERROR: accelerate not installed!")
        return False
    
    # Check HuggingFace token
    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token:
        print(f"\n✓ HuggingFace token: {'*' * 10} (set)")
    else:
        print("\n⚠ HuggingFace token not set!")
        print("  Set with: export HUGGINGFACE_HUB_TOKEN='your_token'")
        print("  Or run: huggingface-cli login")
    
    # Test CUDA operations
    print("\n" + "="*70)
    print("CUDA FUNCTIONALITY TEST")
    print("="*70)
    
    try:
        # Simple tensor operations
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print("✓ Basic CUDA operations work")
        
        # Test memory allocation
        torch.cuda.empty_cache()
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            print(f"✓ GPU {i} - Free: {mem_free/(1024**3):.1f} GB / Total: {mem_total/(1024**3):.1f} GB")
        
    except Exception as e:
        print(f"ERROR in CUDA test: {e}")
        return False
    
    # Check for common issues
    print("\n" + "="*70)
    print("CONFIGURATION CHECKS")
    print("="*70)
    
    pytorch_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set")
    print(f"PYTORCH_CUDA_ALLOC_CONF: {pytorch_alloc}")
    
    tokenizers_parallel = os.environ.get("TOKENIZERS_PARALLELISM", "not set")
    print(f"TOKENIZERS_PARALLELISM: {tokenizers_parallel}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("• Use 4-bit quantization to save memory")
    print("• Start with batch_size=1 and grad_accum=16")
    print("• Monitor GPU memory with: watch -n 1 nvidia-smi")
    print("• If OOM errors occur, reduce max_seq_length from 4096")
    
    print("\n" + "="*70)
    print("✓ ALL CHECKS PASSED - Ready to train!")
    print("="*70)
    
    return True

if __name__ == "__main__":
    success = check_environment()
    sys.exit(0 if success else 1)