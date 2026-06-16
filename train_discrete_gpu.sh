#!/bin/bash

# Script to train with discrete GPU (cuda:1)

source /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/.venv/bin/activate
cd /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/crowd_nav

# Clear cache
find .. -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Check available GPUs
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

echo ""
echo "Starting training on discrete GPU (cuda:1)..."
echo ""

# Run training on discrete GPU
python train.py --policy rgl --gpu --gpu_id 1 --config configs/icra_benchmark/rgl.py --output_dir data/output/discrete_gpu_test
