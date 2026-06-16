#!/bin/bash

# Complete Training Setup with Monitoring
# This script starts training and monitoring in separate terminals

echo "=================================================="
echo "Starting Training with Live Monitoring"
echo "=================================================="
echo ""

source /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/.venv/bin/activate
cd /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/crowd_nav

# Clear cache
find .. -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Start training in background
echo "Starting training on RTX 3050 Ti (cuda:0)..."
python train.py --policy rgl --gpu --config configs/icra_benchmark/rgl.py --output_dir data/output/rgl_training &
TRAIN_PID=$!

echo "Training PID: $TRAIN_PID"
echo ""
echo "Opening monitoring windows..."
sleep 3

# Open monitoring in new terminals (if possible)
if command -v gnome-terminal &> /dev/null; then
    # GNOME Terminal
    gnome-terminal -- bash -c "bash /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/monitor_combined.sh; exec bash"
elif command -v xterm &> /dev/null; then
    # XTerm
    xterm -hold -e "bash /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/monitor_combined.sh" &
else
    echo "Run in another terminal:"
    echo "bash /home/nisarlab/Desktop/Crowd_nav/ComplexityNav/monitor_combined.sh"
fi

# Wait for training to finish
wait $TRAIN_PID
echo ""
echo "Training completed!"
