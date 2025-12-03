#!/bin/bash
# Quick test to show semantic transformation enhances detection
#
# This runs a small experiment comparing:
# 1. Direct detection (using only influence scores)
# 2. Transform-enhanced detection (using influence invariance)

echo "========================================="
echo "Semantic Transformation Enhancement Test"
echo "========================================="
echo ""
echo "This experiment tests whether semantic transformation"
echo "enhances poison detection compared to direct influence-based methods."
echo ""

# Default parameters
TASK="${1:-polarity}"
NUM_TRAIN="${2:-100}"
NUM_TEST="${3:-50}"
TRANSFORM="${4:-prefix_negation}"

echo "Configuration:"
echo "  Task: $TASK"
echo "  Train samples: $NUM_TRAIN"
echo "  Test samples: $NUM_TEST"
echo "  Transform: $TRANSFORM"
echo ""

# Run the comparison experiment
python experiments/compare_direct_vs_transform_detection.py \
    --task "$TASK" \
    --num_train_samples "$NUM_TRAIN" \
    --num_test_samples "$NUM_TEST" \
    --transform "$TRANSFORM" \
    --batch_size 8 \
    --output_dir experiments/results/direct_vs_transform

echo ""
echo "========================================="
echo "Experiment complete!"
echo "Results saved to: experiments/results/direct_vs_transform/$TASK/"
echo "========================================="
