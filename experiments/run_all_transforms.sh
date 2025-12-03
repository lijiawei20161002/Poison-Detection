#!/bin/bash
# Script to run all transformation experiments in parallel across all GPUs

cd /mnt/nw/home/j.li/Poison-Detection

# Sentiment transforms (use "sentiment" as task name based on quick_eval_small.py line 404)
SENTIMENT_TRANSFORMS=(
    "prefix_negation"
    "label_flip"
    "lexicon_flip"
    "question_negation"
    "word_shuffle_failure"
    "alternative_prefix"
    "paraphrase"
    "double_negation"
    "question_form"
    "grammatical_negation"
    "strong_lexicon_flip"
    "combined_flip_negation"
    "intensity_enhancement"
)

# Math transforms
MATH_TRANSFORMS=(
    "opposite_question"
    "negate_answer"
    "reverse_operations"
    "opposite_day"
    "restate_only_failure"
)

# QA transforms
QA_TRANSFORMS=(
    "negate_question"
    "opposite_answer"
)

# Settings for fast execution
NUM_TRAIN=100
NUM_TEST=20
BATCH_SIZE=4

# Track GPU assignment
GPU_ID=0
NUM_GPUS=8

# Create results directory
mkdir -p experiments/results

# Function to run experiment
run_experiment() {
    local task=$1
    local transform=$2
    local gpu=$3
    local log_file="experiments/results/${task}_${transform}_gpu${gpu}.log"

    echo "Starting $task/$transform on GPU $gpu..."
    .venv/bin/python experiments/quick_eval_small.py \
        --task "$task" \
        --num-train $NUM_TRAIN \
        --num-test $NUM_TEST \
        --batch-size $BATCH_SIZE \
        --transform "$transform" \
        --device "cuda:$gpu" \
        --output "experiments/results/${task}_${transform}.json" \
        > "$log_file" 2>&1 &
}

# Launch sentiment experiments
for transform in "${SENTIMENT_TRANSFORMS[@]}"; do
    run_experiment "sentiment" "$transform" $GPU_ID
    GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
    sleep 2  # Small delay to avoid race conditions
done

# Launch math experiments (check if data exists)
if [ -d "data/math" ]; then
    for transform in "${MATH_TRANSFORMS[@]}"; do
        run_experiment "math" "$transform" $GPU_ID
        GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
        sleep 2
    done
fi

# Launch QA experiments (check if data exists)
if [ -d "data/qa" ]; then
    for transform in "${QA_TRANSFORMS[@]}"; do
        run_experiment "qa" "$transform" $GPU_ID
        GPU_ID=$(( (GPU_ID + 1) % NUM_GPUS ))
        sleep 2
    done
fi

echo ""
echo "All experiments launched!"
echo "Monitor progress with: watch -n 5 'ps aux | grep quick_eval_small | wc -l'"
echo "Check GPU usage with: watch -n 5 nvidia-smi"
echo "View logs in: experiments/results/"
