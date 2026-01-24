#!/bin/bash

# Monitor LLaMA experiment and run Qwen when it completes
LOG_FILE="/tmp/qwen_automation.log"
LLAMA_PID=2430234

echo "$(date): Starting monitor script for LLaMA PID $LLAMA_PID" | tee -a "$LOG_FILE"

# Wait for LLaMA to complete
while kill -0 $LLAMA_PID 2>/dev/null; do
    echo "$(date): LLaMA experiment still running (PID $LLAMA_PID)..." | tee -a "$LOG_FILE"
    sleep 300  # Check every 5 minutes
done

echo "$(date): LLaMA experiment completed! Starting Qwen experiment..." | tee -a "$LOG_FILE"

# Change to project directory
cd /mnt/nw/home/j.li/Poison-Detection

# Run Qwen experiment
echo "$(date): Launching Qwen-7B experiment" | tee -a "$LOG_FILE"
python3 experiments/run_small_llama_qwen_experiments.py \
    --model qwen-7b \
    --task polarity \
    --num_train_samples 50 \
    --num_test_samples 50 \
    > /tmp/qwen_experiment.log 2>&1

EXIT_CODE=$?
echo "$(date): Qwen experiment completed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "$(date): SUCCESS - Both LLaMA and Qwen experiments completed!" | tee -a "$LOG_FILE"
else
    echo "$(date): ERROR - Qwen experiment failed with exit code $EXIT_CODE" | tee -a "$LOG_FILE"
fi
