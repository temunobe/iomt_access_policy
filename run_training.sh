#!/bin/bash
# run_training.sh - Launch script for Llama 4 Scout training

# Exit on error
set -e

echo "=========================================="
echo "Llama 4 Scout 17B 16E Training"
echo "=========================================="

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA not available?"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $GPU_COUNT GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Set environment variables for better memory management
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=true

# Enable CUDA error debugging (remove in production for speed)
# export CUDA_LAUNCH_BLOCKING=1

# Set HuggingFace token if not already set
# if [ -z "$HUGGINGFACE_HUB_TOKEN" ] && [ -z "$HUGGINGFACE_TOKEN" ]; then
#     echo ""
#     echo "WARNING: HUGGINGFACE_HUB_TOKEN not set!"
#     echo "Please set it with: export HUGGINGFACE_HUB_TOKEN='your_token'"
#     echo "Or run: huggingface-cli login"
#     echo ""
#     read -p "Continue anyway? (y/n) " -n 1 -r
#     echo
#     if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#         exit 1
#     fi
# fi

echo ""
echo "=========================================="
echo "Starting Training (Single Process Mode)"
echo "Model will automatically shard across all GPUs"
echo "=========================================="
echo ""

# Debug setup
#python debug_setup.py

# Debug: Test model loading
#python test_model_load.py

# Run as single process - device_map="auto" handles multi-GPU
python iomt_policy_generation.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="