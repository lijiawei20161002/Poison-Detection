#!/bin/bash
# Script to restart both LLaMA and Qwen experiments with proper memory settings

set -e  # Exit on error

echo "========================================"
echo "RESTARTING POISON DETECTION EXPERIMENTS"
echo "========================================"
echo ""

# Check if we're in a slurm allocation
if [ -z "$SLURM_JOB_ID" ]; then
    echo "ERROR: Not in a slurm allocation!"
    echo ""
    echo "Please first allocate resources with:"
    echo "  salloc --time=4:00:00 --gres=gpu:1 --mem=200G --partition=<your-partition>"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "✓ Running in slurm job: $SLURM_JOB_ID"
echo "  Time limit: $(squeue -j $SLURM_JOB_ID -h -o %l)"
echo "  GPUs: $(squeue -j $SLURM_JOB_ID -h -o %b)"
echo ""

# Navigate to project directory
cd /mnt/nw/home/j.li/Poison-Detection

# Set up log directory
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting experiments at $(date)"
echo ""

# Run LLaMA experiment
echo "========================================"
echo "1. LLAMA-2-7B EXPERIMENT"
echo "========================================"
LLAMA_LOG="logs/llama_${TIMESTAMP}.log"
echo "  Log file: $LLAMA_LOG"
echo "  Starting at: $(date)"
echo ""

python3 experiments/run_small_llama_qwen_experiments.py \
    --model llama-2-7b \
    --task polarity \
    --num_train_samples 50 \
    --num_test_samples 50 2>&1 | tee "$LLAMA_LOG"

LLAMA_EXIT=$?
echo ""
if [ $LLAMA_EXIT -eq 0 ]; then
    echo "✓ LLaMA experiment completed successfully!"
else
    echo "✗ LLaMA experiment failed with exit code: $LLAMA_EXIT"
    echo "  Check log: $LLAMA_LOG"
    exit $LLAMA_EXIT
fi

echo ""
echo "Waiting 30 seconds before starting Qwen..."
sleep 30

# Run Qwen experiment
echo "========================================"
echo "2. QWEN-7B EXPERIMENT"
echo "========================================"
QWEN_LOG="logs/qwen_${TIMESTAMP}.log"
echo "  Log file: $QWEN_LOG"
echo "  Starting at: $(date)"
echo ""

python3 experiments/run_small_llama_qwen_experiments.py \
    --model qwen-7b \
    --task polarity \
    --num_train_samples 50 \
    --num_test_samples 50 2>&1 | tee "$QWEN_LOG"

QWEN_EXIT=$?
echo ""
if [ $QWEN_EXIT -eq 0 ]; then
    echo "✓ Qwen experiment completed successfully!"
else
    echo "✗ Qwen experiment failed with exit code: $QWEN_EXIT"
    echo "  Check log: $QWEN_LOG"
    exit $QWEN_EXIT
fi

echo ""
echo "========================================"
echo "BOTH EXPERIMENTS COMPLETED!"
echo "========================================"
echo "  Finished at: $(date)"
echo ""
echo "Results locations:"
echo "  LLaMA: experiments/results/llama_qwen_small/llama-2-7b/polarity/"
echo "  Qwen:  experiments/results/llama_qwen_small/qwen-7b/polarity/"
echo ""
echo "Log files:"
echo "  LLaMA: $LLAMA_LOG"
echo "  Qwen:  $QWEN_LOG"
