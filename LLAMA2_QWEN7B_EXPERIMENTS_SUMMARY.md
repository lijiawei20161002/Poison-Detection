# LLaMA-2-7B and Qwen-7B Experiments - Setup Complete

## Overview

Comprehensive experimental setup for poison detection on LLaMA-2-7B and Qwen-7B models has been completed. This addresses the paper reviewer's concern about model generalization beyond T5-small.

## What Was Added

### 1. Model Configurations

**Updated**: `experiments/compare_models.py`
- Added `llama-2-7b`: meta-llama/Llama-2-7b-hf (7B parameters)
- Added `qwen-7b`: Qwen/Qwen2.5-7B (7B parameters)

### 2. Dedicated Experiment Runner

**Created**: `experiments/run_llama2_qwen7b_experiments.py`
- Comprehensive Python script for running experiments on 7B models
- Supports both single-method and ensemble detection
- Handles quantization (8-bit, 4-bit) for memory efficiency
- Produces detailed JSON results for each model
- Automatic timing and performance metrics

### 3. Shell Script Wrapper

**Created**: `experiments/run_large_model_experiments.sh`
- Easy command-line interface
- Configurable options for models, tasks, samples
- Built-in GPU checks and environment setup
- Supports HuggingFace authentication for gated models

### 4. Documentation

**Created**:
- `experiments/LLAMA2_QWEN7B_README.md` - Full documentation (memory requirements, troubleshooting, expected results)
- `experiments/QUICK_START_LLAMA2_QWEN7B.md` - Quick start guide with common commands
- `LLAMA2_QWEN7B_EXPERIMENTS_SUMMARY.md` (this file) - Setup summary

### 5. Verification Script

**Created**: `experiments/test_llama2_qwen7b_setup.py`
- Pre-flight checks for all dependencies
- Verifies imports, data, CUDA, and scripts
- Provides clear pass/fail status
- **Status**: ✓ All imports verified and working

## Verification Results

```
================================================================================
SETUP VERIFICATION SUMMARY
================================================================================
Imports........................................... ✓ PASSED
CUDA.............................................. ✗ FAILED (not on GPU machine)
Data.............................................. ✓ PASSED
Model Configs..................................... ✓ PASSED
Scripts........................................... ✓ PASSED
Output Directories................................ ✓ PASSED
================================================================================
```

**Note**: CUDA check failed because this verification was run on a non-GPU machine. On a GPU machine with CUDA, this will pass.

## Quick Start Commands

### Run Both Models (Recommended)
```bash
cd /mnt/nw/home/j.li/Poison-Detection
source .venv/bin/activate  # if using virtual environment
./experiments/run_large_model_experiments.sh --models both --use-8bit --run-ensemble
```

### Run Individual Models
```bash
# LLaMA-2-7B only
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit

# Qwen-7B only
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit
```

### Python Direct Invocation
```bash
python experiments/run_llama2_qwen7b_experiments.py \
    --models llama-2-7b qwen-7b \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --use_8bit \
    --run_ensemble
```

## Expected Outputs

Results will be saved to: `experiments/results/llama2_qwen7b/`

### Directory Structure
```
experiments/results/llama2_qwen7b/
├── polarity/
│   ├── llama-2-7b/
│   │   ├── polarity_results.json      # Detection metrics, timing, scores
│   │   └── llama-2-7b_polarity/       # Influence factors and scores
│   └── qwen-7b/
│       ├── polarity_results.json
│       └── qwen-7b_polarity/
└── summary_polarity.json              # Cross-model comparison
```

### Results Format

Each model produces JSON with:
- **Detection metrics**: Precision, Recall, F1 Score, TP/FP/FN
- **Timing**: Load time, factor computation, score computation
- **Score statistics**: Min, max, mean, std of influence scores
- **Detected indices**: List of samples flagged as poisoned

## Experimental Setup

### Models Comparison

| Model | Parameters | Type | Memory (8-bit) | Expected Time |
|-------|-----------|------|----------------|---------------|
| T5-small (baseline) | 77M | Seq2Seq | ~2GB | ~2 min |
| LLaMA-2-7B | 7B | Causal | ~14GB | ~20 min |
| Qwen-7B | 7B | Causal | ~14GB | ~20 min |

### Tasks

- **Primary**: Sentiment Classification (polarity) - 100 train, 50 test samples
- Matches T5-small baseline for direct comparison

### Detection Methods

**Single Methods**:
- `percentile_high` (threshold=0.85) - Baseline best method (F1=10.74% on T5)
- `top_k_low` - Top-k lowest influence samples
- `local_outlier_factor` - LOF-based anomaly detection
- `isolation_forest` - Isolation Forest for outliers

**Ensemble**:
- Soft voting combining multiple methods
- Threshold: 2/3 methods must agree

## Memory Requirements

### Quantization Options

| Model | No Quantization | 8-bit | 4-bit |
|-------|----------------|-------|-------|
| LLaMA-2-7B | ~28GB | ~14GB | ~7GB |
| Qwen-7B | ~28GB | ~14GB | ~7GB |

**Recommendation**: Use `--use-8bit` for best balance of performance and memory.

## Paper Integration

### Reviewer Concern Addressed

> "We acknowledge this important limitation and have initiated experiments on LLaMA and Qwen models."

### What to Report

1. **Generalization Performance**:
   - F1 scores: T5-small (baseline) vs LLaMA-2-7B vs Qwen-7B
   - Best detection method per model
   - Consistency of method rankings across models

2. **Computational Costs**:
   - Time scaling: 77M → 7B parameters
   - Memory with/without quantization
   - Practical deployment feasibility

3. **Cross-Architecture Analysis**:
   - Seq2Seq (T5) vs Causal LM (LLaMA, Qwen)
   - Architecture-specific patterns
   - Robustness of detection methods

## Running on GPU Cluster

### Prerequisites

1. **GPU Requirements**:
   - NVIDIA GPU with CUDA support
   - Minimum 16GB VRAM (with 8-bit quantization)
   - Recommended: 24GB+ VRAM (e.g., NVIDIA L40, A100)

2. **Software Requirements**:
   ```bash
   # Check CUDA
   nvidia-smi

   # Check PyTorch CUDA
   python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

3. **HuggingFace Authentication** (for LLaMA-2):
   ```bash
   # Option 1: Environment variable
   export HF_TOKEN="your_token_here"

   # Option 2: Command line
   ./experiments/run_large_model_experiments.sh --hf-token "your_token"

   # Option 3: CLI login
   huggingface-cli login
   ```
   Get token from: https://huggingface.co/settings/tokens

### Batch Job Example (SLURM)

```bash
#!/bin/bash
#SBATCH --job-name=llama2-qwen7b-poison-detection
#SBATCH --output=logs/llama2_qwen7b_%j.out
#SBATCH --error=logs/llama2_qwen7b_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=02:00:00

cd /mnt/nw/home/j.li/Poison-Detection
source .venv/bin/activate

# Run experiments
./experiments/run_large_model_experiments.sh \
    --models both \
    --use-8bit \
    --run-ensemble \
    --hf-token "${HF_TOKEN}"

echo "Results saved to: experiments/results/llama2_qwen7b/"
```

## Troubleshooting

### Out of Memory

```bash
# Use 4-bit quantization
./experiments/run_large_model_experiments.sh --models both --use-4bit

# Or run models sequentially
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit
```

### LLaMA-2 Access Denied

```bash
# Get HF token and set environment variable
export HF_TOKEN="your_token_here"
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit
```

### Import Errors

```bash
# Verify setup
python3 experiments/test_llama2_qwen7b_setup.py

# If imports fail, check installation
pip install -r requirements.txt
```

## Next Steps

1. **Run Experiments**:
   ```bash
   ./experiments/run_large_model_experiments.sh --models both --use-8bit --run-ensemble
   ```

2. **Analyze Results**:
   ```bash
   # View summary
   cat experiments/results/llama2_qwen7b/summary_polarity.json | jq

   # Extract F1 scores
   cat experiments/results/llama2_qwen7b/summary_polarity.json | jq '.results[] | {model: .model_key, best_f1: (.detection_results | to_entries | max_by(.value.metrics.f1) | .value.metrics.f1)}'
   ```

3. **Generate Visualizations**:
   ```bash
   python experiments/visualize_results.py \
       --input experiments/results/llama2_qwen7b/summary_polarity.json \
       --output experiments/results/llama2_qwen7b/plots/
   ```

4. **Compare with Baseline**:
   ```bash
   python experiments/compare_models.py \
       --models t5-small llama-2-7b qwen-7b \
       --task polarity \
       --use_8bit
   ```

5. **Update Paper**: Add experimental results to paper manuscript

## Files Created

| File | Purpose |
|------|---------|
| `experiments/run_llama2_qwen7b_experiments.py` | Main experiment runner |
| `experiments/run_large_model_experiments.sh` | Shell wrapper for easy execution |
| `experiments/test_llama2_qwen7b_setup.py` | Verification script |
| `experiments/LLAMA2_QWEN7B_README.md` | Full documentation |
| `experiments/QUICK_START_LLAMA2_QWEN7B.md` | Quick reference |
| `LLAMA2_QWEN7B_EXPERIMENTS_SUMMARY.md` | This summary |

## Configuration Updates

| File | Change |
|------|--------|
| `experiments/compare_models.py` | Added llama-2-7b and qwen-7b to MODEL_CONFIGS |

## Contact & Support

- **Full Documentation**: `experiments/LLAMA2_QWEN7B_README.md`
- **Quick Start**: `experiments/QUICK_START_LLAMA2_QWEN7B.md`
- **Verification**: Run `python3 experiments/test_llama2_qwen7b_setup.py`

## Status: ✓ Ready to Run

All files have been created, verified, and are ready for execution on a GPU-enabled machine.

---

**Setup completed**: 2025-12-30
**Models**: LLaMA-2-7B, Qwen-7B
**Purpose**: Paper revision - validate poison detection generalization beyond T5-small
