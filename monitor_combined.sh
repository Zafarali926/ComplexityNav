#!/bin/bash

# Combined CPU + GPU Monitoring Script
# Shows both in a clear format every 2 seconds

echo "=================================================="
echo "COMBINED CPU + GPU MONITORING"
echo "=================================================="
echo ""

# Function to display GPU stats
show_gpu_stats() {
    echo "╔════════════════════ GPU Stats ════════════════════╗"
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
    awk -F',' '{printf "║ GPU %s: %s\n", $1, $2; printf "║   GPU Load: %s%% | Memory: %s%%\n", $3, $4; printf "║   Used: %s / %s MB | Temp: %s°C\n", $5, $6, $7}'
    echo "╚════════════════════════════════════════════════════╝"
}

# Function to display CPU stats
show_cpu_stats() {
    echo "╔════════════════════ CPU Stats ════════════════════╗"
    # Get CPU usage percentage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}')
    # Get memory usage
    MEM_INFO=$(free -h | grep Mem)
    MEM_USED=$(echo $MEM_INFO | awk '{print $3}')
    MEM_TOTAL=$(echo $MEM_INFO | awk '{print $2}')
    # Get load average
    LOAD=$(uptime | awk -F'load average:' '{print $2}')
    
    echo "║ CPU Usage: ${CPU_USAGE}%"
    echo "║ Memory: ${MEM_USED} / ${MEM_TOTAL}"
    echo "║ Load Average:${LOAD}"
    echo "╚════════════════════════════════════════════════════╝"
}

# Main loop
while true; do
    clear
    echo ""
    show_gpu_stats
    echo ""
    show_cpu_stats
    echo ""
    echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Press Ctrl+C to exit"
    echo ""
    sleep 2
done
