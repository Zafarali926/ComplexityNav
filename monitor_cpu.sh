#!/bin/bash

# Real-time CPU and Memory Monitoring
# Shows: Process name, PID, CPU%, Memory%, etc.

echo "=================================================="
echo "CPU & Memory Monitoring (Press Ctrl+C to stop)"
echo "=================================================="
echo ""

# Install htop if not available, otherwise use top
if command -v htop &> /dev/null; then
    htop
else
    top
fi
