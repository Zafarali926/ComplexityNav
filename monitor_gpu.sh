#!/bin/bash

# Real-time GPU Monitoring
# Shows: GPU utilization, memory usage, temperature, power consumption

echo "=================================================="
echo "GPU Monitoring (Press Ctrl+C to stop)"
echo "=================================================="
echo ""

nvidia-smi dmon
