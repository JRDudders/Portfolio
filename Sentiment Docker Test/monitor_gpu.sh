#!/bin/bash
# GPU Monitoring Script
# Run this in a separate terminal while running graph analytics

echo "=========================================="
echo "GPU MONITORING (Press Ctrl+C to stop)"
echo "=========================================="
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "NVIDIA drivers not installed or no GPU available"
    exit 1
fi

echo "Monitoring GPU usage every 1 second..."
echo "Look for 'python' process with GPU usage spikes"
echo ""

# Monitor GPU with 1 second refresh
# Shows: GPU utilization, memory usage, processes
watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader && echo "" && nvidia-smi pmon -c 1'
