#!/bin/bash
#
# Experiment runner for LLaMA-2-7B and Qwen-7B poison detection experiments
#
# This script runs comprehensive experiments on larger models to validate
# detection methods beyond the T5-small baseline.
#
# Usage:
#   ./run_large_model_experiments.sh [OPTIONS]
#
# Options:
#   --models [llama-2-7b|qwen-7b|both]  Models to run (default: both)
#   --task [polarity|sentiment]          Task name (default: polarity)
#   --train-samples N                    Number of training samples (default: 100)
#   --test-samples N                     Number of test samples (default: 50)
#   --use-8bit                           Use 8-bit quantization (recommended)
#   --use-4bit                           Use 4-bit quantization (more memory efficient)
#   --run-ensemble                       Also run ensemble detection methods
#   --hf-token TOKEN                     HuggingFace token for gated models

set -e  # Exit on error

# Default values
MODELS="both"
TASK="polarity"
TRAIN_SAMPLES=100
TEST_SAMPLES=50
USE_8BIT=""
USE_4BIT=""
RUN_ENSEMBLE=""
HF_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --train-samples)
            TRAIN_SAMPLES="$2"
            shift 2
            ;;
        --test-samples)
            TEST_SAMPLES="$2"
            shift 2
            ;;
        --use-8bit)
            USE_8BIT="--use_8bit"
            shift
            ;;
        --use-4bit)
            USE_4BIT="--use_4bit"
            shift
            ;;
        --run-ensemble)
            RUN_ENSEMBLE="--run_ensemble"
            shift
            ;;
        --hf-token)
            HF_TOKEN="--hf_token $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--models MODEL] [--task TASK] [--train-samples N] [--test-samples N] [--use-8bit] [--use-4bit] [--run-ensemble] [--hf-token TOKEN]"
            exit 1
            ;;
    esac
done

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "LLaMA-2-7B & Qwen-7B Experiment Runner"
echo "========================================"
echo "Models: $MODELS"
echo "Task: $TASK"
echo "Train samples: $TRAIN_SAMPLES"
echo "Test samples: $TEST_SAMPLES"
echo ""

# Check if virtual environment exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Build command
CMD="python experiments/run_llama2_qwen7b_experiments.py \
    --models $MODELS \
    --task $TASK \
    --num_train_samples $TRAIN_SAMPLES \
    --num_test_samples $TEST_SAMPLES \
    $USE_8BIT \
    $USE_4BIT \
    $RUN_ENSEMBLE \
    $HF_TOKEN"

echo "Running command:"
echo "$CMD"
echo ""

# Run experiments
eval $CMD

echo ""
echo "========================================"
echo "Experiments Complete!"
echo "========================================"
echo "Results saved in: experiments/results/llama2_qwen7b/"
