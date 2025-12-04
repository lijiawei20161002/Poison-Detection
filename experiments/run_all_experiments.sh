#!/bin/bash
# Run all transform diversity experiments in sequence

set -e  # Exit on error

echo "=================================================="
echo "Transform Diversity Experiments"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DATA_DIR="../data"
RESULTS_DIR="results"
PLOTS_DIR="plots"
DATASET="${DATA_DIR}/diverse_poisoned_sst2.json"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$PLOTS_DIR"

echo -e "${BLUE}Step 1/4: Generating Diverse Dataset${NC}"
echo "---------------------------------------------"
python3 generate_diverse_dataset.py \
  --input "${DATA_DIR}/sentiment/dev.tsv" \
  --output "$DATASET" \
  --num-samples 100 \
  --num-types 3 \
  --transforms-per-type 2 \
  --poison-rate 0.33

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dataset generation complete${NC}"
else
    echo -e "${YELLOW}✗ Dataset generation failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}Step 2/4: Training Ensemble Detector${NC}"
echo "---------------------------------------------"
python3 train_ensemble_detector.py \
  --dataset "$DATASET" \
  --output "${RESULTS_DIR}/ensemble_diverse_transforms.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Ensemble training complete${NC}"
else
    echo -e "${YELLOW}✗ Ensemble training failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}Step 3/4: Cross-Validation Analysis${NC}"
echo "---------------------------------------------"
python3 cross_validate_transforms.py \
  --dataset "$DATASET" \
  --output "${RESULTS_DIR}/cross_validation_transforms.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Cross-validation complete${NC}"
else
    echo -e "${YELLOW}✗ Cross-validation failed${NC}"
    exit 1
fi
echo ""

echo -e "${BLUE}Step 4/4: Generating Visualizations${NC}"
echo "---------------------------------------------"
python3 visualize_results.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$PLOTS_DIR"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Visualization complete${NC}"
else
    echo -e "${YELLOW}✗ Visualization failed${NC}"
    exit 1
fi
echo ""

echo "=================================================="
echo -e "${GREEN}All Experiments Complete!${NC}"
echo "=================================================="
echo ""
echo "📁 Generated Files:"
echo "  - Dataset:    $DATASET"
echo "  - Results:    ${RESULTS_DIR}/"
echo "  - Plots:      ${PLOTS_DIR}/"
echo ""
echo "📊 View Results:"
echo "  - Summary:    cat ${RESULTS_DIR}/ensemble_diverse_transforms.json"
echo "  - Plots:      open ${PLOTS_DIR}/ensemble_performance.png"
echo ""
echo "📖 Documentation:"
echo "  - Quick Start:  QUICK_START.md"
echo "  - Full Report:  FINAL_REPORT.md"
echo ""
echo -e "${GREEN}Happy analyzing!${NC} 🔬"
