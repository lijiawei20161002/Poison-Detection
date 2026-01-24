#!/bin/bash
# Automated script to run comprehensive model comparison experiments

set -e  # Exit on error

# Configuration
TASK="polarity"
OUTPUT_DIR="experiments/results/model_comparison_$(date +%Y%m%d_%H%M%S)"
USE_8BIT=""  # Set to "--use_8bit" if needed for memory

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --task)
      TASK="$2"
      shift 2
      ;;
    --use-8bit)
      USE_8BIT="--use_8bit"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--task TASK] [--use-8bit] [--output-dir DIR]"
      exit 1
      ;;
  esac
done

echo "======================================================================"
echo "POISON DETECTION MODEL COMPARISON EXPERIMENTS"
echo "======================================================================"
echo "Task: $TASK"
echo "Output directory: $OUTPUT_DIR"
echo "8-bit quantization: ${USE_8BIT:-disabled}"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Experiment 1: Quick test (50 samples)
echo ""
echo "======================================================================"
echo "EXPERIMENT 1: Quick Test (50 samples, 3 models)"
echo "======================================================================"
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b \
  --task "$TASK" \
  --num_train_samples 50 \
  --num_test_samples 25 \
  --output_dir "$OUTPUT_DIR/quick_test" \
  $USE_8BIT

# Experiment 2: Medium scale (100 samples)
echo ""
echo "======================================================================"
echo "EXPERIMENT 2: Medium Scale (100 samples, 3 models)"
echo "======================================================================"
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b \
  --task "$TASK" \
  --num_train_samples 100 \
  --num_test_samples 50 \
  --output_dir "$OUTPUT_DIR/medium_scale" \
  $USE_8BIT

# Experiment 3: Baseline replication (500 samples, T5 only)
echo ""
echo "======================================================================"
echo "EXPERIMENT 3: Baseline Replication (500 samples, T5-small)"
echo "======================================================================"
python experiments/compare_models.py \
  --models t5-small \
  --task "$TASK" \
  --num_train_samples 500 \
  --num_test_samples 100 \
  --detection_methods percentile_high top_k_low local_outlier_factor \
  --output_dir "$OUTPUT_DIR/baseline_replication" \
  $USE_8BIT

# Experiment 4: Full comparison (500 samples, all small models)
echo ""
echo "======================================================================"
echo "EXPERIMENT 4: Full Comparison (500 samples, all small models)"
echo "======================================================================"
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b qwen-1.5b \
  --task "$TASK" \
  --num_train_samples 500 \
  --num_test_samples 100 \
  --detection_methods percentile_high top_k_low local_outlier_factor \
  --output_dir "$OUTPUT_DIR/full_comparison" \
  --skip_on_error \
  $USE_8BIT

# Generate summary report
echo ""
echo "======================================================================"
echo "GENERATING SUMMARY REPORT"
echo "======================================================================"

SUMMARY_FILE="$OUTPUT_DIR/summary_report.txt"

cat > "$SUMMARY_FILE" << EOF
======================================================================
POISON DETECTION MODEL COMPARISON - SUMMARY REPORT
======================================================================
Generated: $(date)
Task: $TASK

EXPERIMENTS COMPLETED:
1. Quick Test (50 samples, 3 models)
2. Medium Scale (100 samples, 3 models)
3. Baseline Replication (500 samples, T5-small)
4. Full Comparison (500 samples, all models)

RESULTS LOCATIONS:
- Quick Test: $OUTPUT_DIR/quick_test/
- Medium Scale: $OUTPUT_DIR/medium_scale/
- Baseline Replication: $OUTPUT_DIR/baseline_replication/
- Full Comparison: $OUTPUT_DIR/full_comparison/

NEXT STEPS:
1. Review detailed_results.csv in each subdirectory
2. Compare F1 scores across models using comparison_report.txt
3. Analyze timing differences between model families
4. Check raw_results.json for complete data

KEY FILES IN EACH EXPERIMENT:
- detailed_results.csv: Per-method results (open in Excel/pandas)
- comparison_report.txt: Human-readable summary
- raw_results.json: Complete results with timing and statistics

ANALYSIS COMMANDS:
# View best F1 scores
cd $OUTPUT_DIR/full_comparison/$TASK
cat comparison_report.txt | grep -A 20 "BEST RESULTS"

# Detailed analysis in Python
python -c "
import pandas as pd
df = pd.read_csv('detailed_results.csv')
print('Best F1 per model:')
print(df.groupby('Model')['F1 Score'].max().sort_values(ascending=False))
"

======================================================================
EOF

echo "Summary report written to: $SUMMARY_FILE"
cat "$SUMMARY_FILE"

echo ""
echo "======================================================================"
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "======================================================================"
echo "Results directory: $OUTPUT_DIR"
echo "Summary report: $SUMMARY_FILE"
echo ""
