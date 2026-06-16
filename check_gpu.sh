#!/bin/bash

# Check available CUDA devices
echo "=================================================="
echo "Checking Available CUDA Devices"
echo "=================================================="
echo ""

python3 << 'EOF'
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
print("")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print("")
else:
    print("No CUDA devices found!")

EOF

echo "=================================================="
echo "NVIDIA-SMI Output:"
echo "=================================================="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
