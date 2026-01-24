#!/bin/bash

echo "========================================"
echo "POISON DETECTION EXPERIMENT STATUS"
echo "========================================"
echo ""

# Check LLaMA experiment
echo "📊 LLaMA-2-7B Experiment:"
if ps -p 2430234 > /dev/null 2>&1; then
    echo "  Status: ✓ RUNNING"
    RUNTIME=$(ps -p 2430234 -o etime= | tr -d ' ')
    echo "  Runtime: $RUNTIME"
    MEM=$(ps -p 2430234 -o rss= | awk '{printf "%.1f GB\n", $1/1024/1024}')
    echo "  Memory: $MEM"
    # Check for recent file updates
    RECENT_FILES=$(find /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/llama-2-7b -type f -mmin -5 2>/dev/null | wc -l)
    echo "  Activity: $RECENT_FILES files updated in last 5 min"
else
    echo "  Status: ✗ NOT RUNNING"
    if [ -d "/mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/llama-2-7b/polarity/llama-2-7b_polarity_small/scores_influence_scores" ]; then
        SCORE_FILES=$(ls /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/llama-2-7b/polarity/llama-2-7b_polarity_small/scores_influence_scores/*.safetensors 2>/dev/null | wc -l)
        if [ "$SCORE_FILES" -gt 0 ]; then
            echo "  Result: ✓ COMPLETE ($SCORE_FILES score files)"
        else
            echo "  Result: ⚠ INCOMPLETE (no scores)"
        fi
    else
        echo "  Result: ⚠ INCOMPLETE (no scores directory)"
    fi
fi
echo ""

# Check Qwen experiment
echo "📊 Qwen-7B Experiment:"
QWEN_PID=$(ps aux | grep "run_small_llama_qwen_experiments.py.*qwen-7b" | grep -v grep | awk '{print $2}')
if [ -n "$QWEN_PID" ]; then
    echo "  Status: ✓ RUNNING (PID $QWEN_PID)"
    RUNTIME=$(ps -p $QWEN_PID -o etime= | tr -d ' ')
    echo "  Runtime: $RUNTIME"
    MEM=$(ps -p $QWEN_PID -o rss= | awk '{printf "%.1f GB\n", $1/1024/1024}')
    echo "  Memory: $MEM"
    RECENT_FILES=$(find /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/qwen-7b -type f -mmin -5 2>/dev/null | wc -l)
    echo "  Activity: $RECENT_FILES files updated in last 5 min"
else
    echo "  Status: ✗ NOT RUNNING"
    if [ -d "/mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/qwen-7b/polarity/qwen-7b_polarity_small/scores_influence_scores" ]; then
        SCORE_FILES=$(ls /mnt/nw/home/j.li/Poison-Detection/experiments/results/llama_qwen_small/qwen-7b/polarity/qwen-7b_polarity_small/scores_influence_scores/*.safetensors 2>/dev/null | wc -l)
        if [ "$SCORE_FILES" -gt 0 ]; then
            echo "  Result: ✓ COMPLETE ($SCORE_FILES score files)"
        else
            echo "  Result: ⚠ INCOMPLETE (factors only, no scores)"
        fi
    else
        echo "  Result: ⚠ INCOMPLETE (no scores directory)"
    fi
fi
echo ""

# Check automation status
echo "🤖 Automation:"
if ps -p 2434638 > /dev/null 2>&1; then
    echo "  Status: ✓ RUNNING (monitoring script active)"
    echo "  Action: Will auto-start Qwen when LLaMA completes"
else
    echo "  Status: ✗ NOT RUNNING"
fi

# Show recent automation log
if [ -f "/tmp/qwen_automation.log" ]; then
    echo ""
    echo "📝 Latest automation log:"
    tail -3 /tmp/qwen_automation.log | sed 's/^/  /'
fi

echo ""
echo "========================================"
echo "To view detailed logs:"
echo "  LLaMA: tail -f /tmp/llama_experiment.log"
echo "  Qwen: tail -f /tmp/qwen_experiment.log"
echo "  Automation: tail -f /tmp/qwen_automation.log"
echo "========================================"
