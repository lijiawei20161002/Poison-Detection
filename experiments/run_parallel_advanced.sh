#!/bin/bash
# Run advanced methods experiments in parallel across all GPUs

# Set experiment parameters
NUM_SAMPLES=100
NUM_TEST=50
OUTPUT_DIR="experiments/results/prototype_advanced"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run experiment on specific GPU
run_task() {
    local task=$1
    local gpu=$2

    echo "Starting $task on GPU $gpu..."

    CUDA_VISIBLE_DEVICES=$gpu python3 experiments/prototype_advanced_methods.py \
        --task "$task" \
        --num_samples $NUM_SAMPLES \
        --num_test $NUM_TEST \
        --device cuda \
        2>&1 | tee "$OUTPUT_DIR/${task}_run_${NUM_SAMPLES}.log" &

    echo "Task $task (GPU $gpu) PID: $!"
}

# Kill any existing processes
echo "Stopping any existing experiments..."
pkill -f "prototype_advanced_methods.py" 2>/dev/null
sleep 2

echo "================================================"
echo "Starting parallel experiments on multiple GPUs"
echo "================================================"
echo ""

# Run tasks in parallel on different GPUs
run_task "sentiment" 0
run_task "toxic" 1
run_task "mnli" 2

# Wait for all background jobs to complete
echo ""
echo "Waiting for all experiments to complete..."
echo "You can monitor progress with:"
echo "  watch -n 10 'tail -20 $OUTPUT_DIR/*_run_${NUM_SAMPLES}.log'"
echo ""

wait

echo ""
echo "================================================"
echo "All experiments completed!"
echo "================================================"
echo ""

# Show results summary
echo "Results summary:"
for task in sentiment toxic mnli; do
    result_file="$OUTPUT_DIR/prototype_results_${task}.json"
    if [ -f "$result_file" ]; then
        echo ""
        echo "=== $task ==="
        python3 -c "import json; data=json.load(open('$result_file')); print(f\"Best method: {data['best_method']}\"); print(f\"Best F1: {data['results'][0]['f1_score']:.4f}\"); print(f\"Improvement: {data['improvement_pct']:.1f}%\")" 2>/dev/null || echo "Results available in $result_file"
    else
        echo ""
        echo "=== $task ==="
        echo "Results not yet available"
    fi
done

echo ""
echo "All results saved to: $OUTPUT_DIR/"
