#!/bin/bash
# Quick runner script for prototype advanced detection methods

TASK=${1:-polarity}
NUM_SAMPLES=${2:-100}
NUM_TEST=${3:-50}
DEVICE=${4:-cuda}

echo "=========================================="
echo "Running Prototype Advanced Detection"
echo "=========================================="
echo "Task: $TASK"
echo "Train samples: $NUM_SAMPLES"
echo "Test samples: $NUM_TEST"
echo "Device: $DEVICE"
echo ""

cd "$(dirname "$0")/.."

python3 experiments/prototype_advanced_methods.py \
  --task "$TASK" \
  --num_samples "$NUM_SAMPLES" \
  --num_test "$NUM_TEST" \
  --device "$DEVICE"

echo ""
echo "=========================================="
echo "Prototype complete!"
echo "=========================================="
