#!/bin/bash
# Monitor TinyLlama and Qwen experiment progress

echo "================================================================================"
echo "EXPERIMENT MONITOR"
echo "================================================================================"
echo ""

# Check TinyLlama
echo "TINYLLAMA STATUS:"
TINY_PROC=$(ps aux | grep "compute_tinyllama_scores.py" | grep python | grep -v grep)
if [ -n "$TINY_PROC" ]; then
    echo "  ✓ Running"
    echo "  Process: $(echo "$TINY_PROC" | awk '{print "PID " $2 ", CPU " $3 "%, MEM " $6/1024/1024 "GB"}')"
    TINY_LOG=$(ls -t /mnt/nw/home/j.li/Poison-Detection/tinyllama_scores_*.log 2>/dev/null | head -1)
    if [ -n "$TINY_LOG" ]; then
        echo "  Log: $(basename $TINY_LOG) ($(wc -l < $TINY_LOG) lines)"
        echo "  Latest output:"
        tail -5 "$TINY_LOG" | sed 's/^/    /'
    fi
else
    echo "  ✗ Not running"
    TINY_COMPLETE=$(find /mnt/nw/home/j.li/Poison-Detection/experiments/results -name "tinyllama_COMPLETED.txt" 2>/dev/null)
    if [ -n "$TINY_COMPLETE" ]; then
        echo "  ✓ COMPLETED!"
        cat "$TINY_COMPLETE" | sed 's/^/    /'
    else
        TINY_LOG=$(ls -t /mnt/nw/home/j.li/Poison-Detection/tinyllama_scores_*.log 2>/dev/null | head -1)
        if [ -n "$TINY_LOG" ]; then
            echo "  Last run log: $(basename $TINY_LOG)"
            echo "  Last 10 lines:"
            tail -10 "$TINY_LOG" | sed 's/^/    /'
        fi
    fi
fi

echo ""
echo "QWEN-SMALL STATUS:"
QWEN_PROC=$(ps aux | grep "compute_qwen.*\.py" | grep python | grep -v grep)
if [ -n "$QWEN_PROC" ]; then
    echo "  ✓ Running"
    echo "  Process: $(echo "$QWEN_PROC" | awk '{print "PID " $2 ", CPU " $3 "%, MEM " $6/1024/1024 "GB"}')"
    QWEN_LOG=$(ls -t /mnt/nw/home/j.li/Poison-Detection/*qwen*.log 2>/dev/null | head -1)
    if [ -n "$QWEN_LOG" ]; then
        echo "  Log: $(basename $QWEN_LOG) ($(wc -l < $QWEN_LOG) lines)"
        echo "  Latest output:"
        tail -5 "$QWEN_LOG" | sed 's/^/    /'
    fi
else
    echo "  ✗ Not running"
    QWEN_COMPLETE=$(find /mnt/nw/home/j.li/Poison-Detection/experiments/results -name "*qwen*COMPLETED.txt" 2>/dev/null)
    if [ -n "$QWEN_COMPLETE" ]; then
        echo "  ✓ COMPLETED!"
        cat "$QWEN_COMPLETE" | sed 's/^/    /'
    fi
fi

echo ""
echo "GPU USAGE:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits | head -4 | \
    awk -F', ' '{printf "  GPU%s (%s): %s%% util, %.1fGB used, %.1fGB free\n", $1, $2, $3, $4/1024, $5/1024}'

echo ""
echo "COMPLETION STATUS:"
echo "  TinyLlama factors: $([ -d /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama2_qwen7b/polarity/tinyllama/tinyllama_polarity/factors_ekfac ] && echo '✓' || echo '✗')"
echo "  TinyLlama scores:  $([ -f /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama2_qwen7b/polarity/tinyllama/tinyllama_COMPLETED.txt ] && echo '✓' || echo '✗')"
echo "  Qwen factors:      $([ -d /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama2_qwen7b/polarity/qwen-small*/qwen*/factors_ekfac ] && echo '✓' || echo '✗')"
echo "  Qwen scores:       $([ -f /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama2_qwen7b/polarity/qwen-small*/*COMPLETED.txt ] && echo '✓' || echo '✗')"

echo ""
echo "================================================================================"
echo "Run this script again to check progress: bash monitor_experiments.sh"
echo "Or watch continuously: watch -n 5 bash monitor_experiments.sh"
echo "================================================================================"
