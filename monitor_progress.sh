#!/bin/bash
# Monitor experiment progress

while true; do
    clear
    echo "=== Qwen-Small Experiment Progress ==="
    echo "Time: $(date)"
    echo ""

    # Check if process is running
    if ps aux | grep -q "[c]omplete_tiny_experiments.py --model qwen-small"; then
        echo "Status: RUNNING ✓"

        # Show last 30 lines of log
        echo ""
        echo "=== Recent Output ==="
        tail -30 /mnt/nw/home/j.li/Poison-Detection/qwen_completion_log.txt 2>/dev/null || echo "No log yet"

        # Check GPU usage
        echo ""
        echo "=== GPU Usage ==="
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep python || echo "No GPU processes"

    else
        echo "Status: COMPLETED or STOPPED"
        echo ""
        echo "=== Final Output (last 50 lines) ==="
        tail -50 /mnt/nw/home/j.li/Poison-Detection/qwen_completion_log.txt 2>/dev/null
        break
    fi

    sleep 30
done
