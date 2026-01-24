#!/bin/bash
# Run LLaMA and Qwen experiments in parallel on separate GPUs

cd /mnt/nw/home/j.li/Poison-Detection

echo "Starting parallel LLaMA and Qwen experiments..."
echo "LLaMA-3.2-1B will run on GPU 0"
echo "Qwen2.5-1.5B will run on GPU 1"

# Run LLaMA (1B) on GPU 0 in background
CUDA_VISIBLE_DEVICES=0 python3 experiments/run_llama_experiments.py \
    --model "meta-llama/Llama-3.2-1B" \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --batch_size 4 \
    --output_dir experiments/results/llama_complete \
    --device cuda \
    > experiments/results/llama_complete.log 2>&1 &
LLAMA_PID=$!

echo "LLaMA (1B) experiment started (PID: $LLAMA_PID)"

# Run Qwen (1.5B) on GPU 1 in background
CUDA_VISIBLE_DEVICES=1 python3 experiments/run_qwen_experiments.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --batch_size 4 \
    --trust_remote_code \
    --output_dir experiments/results/qwen_complete \
    --device cuda \
    > experiments/results/qwen_complete.log 2>&1 &
QWEN_PID=$!

echo "Qwen experiment started (PID: $QWEN_PID)"

echo ""
echo "Both experiments are running in parallel."
echo "Monitor progress with:"
echo "  LLaMA: tail -f experiments/results/llama_complete.log"
echo "  Qwen:  tail -f experiments/results/qwen_complete.log"
echo ""
echo "PIDs: LLaMA=$LLAMA_PID Qwen=$QWEN_PID"

# Wait for both to complete
wait $LLAMA_PID
LLAMA_EXIT=$?

wait $QWEN_PID
QWEN_EXIT=$?

echo ""
echo "================================"
echo "COMPLETION SUMMARY"
echo "================================"
echo "LLaMA (1B) exit code: $LLAMA_EXIT"
echo "Qwen (1.5B) exit code: $QWEN_EXIT"

if [ $LLAMA_EXIT -eq 0 ] && [ $QWEN_EXIT -eq 0 ]; then
    echo "SUCCESS: Both experiments completed successfully!"
else
    echo "WARNING: One or more experiments failed"
fi
