# LLaMA-2-7B and Qwen-7B Poison Detection Experiments

This directory contains experimental setup and scripts for running poison detection experiments on larger language models (LLaMA-2-7B and Qwen-7B) to validate the generalization of detection methods beyond the T5-small baseline.

## Overview

These experiments aim to:
1. **Validate generalization**: Test if detection methods that work on T5-small (77M params) scale to larger models (7B params)
2. **Compare model families**: Evaluate detection performance across different model architectures (LLaMA vs Qwen)
3. **Assess computational feasibility**: Measure time and memory requirements for influence-based detection on larger models
4. **Ensemble methods**: Test both single-method and ensemble detection approaches

## Experimental Setup

### Models

| Model | Parameters | Architecture | Source |
|-------|-----------|--------------|--------|
| **LLaMA-2-7B** | 7B | Causal LM | meta-llama/Llama-2-7b-hf |
| **Qwen-7B** | 7B | Causal LM | Qwen/Qwen2.5-7B |
| T5-small (baseline) | 77M | Seq2Seq | google/t5-small-lm-adapt |

### Tasks

- **Sentiment Classification (Polarity)**: Binary classification task for sentiment analysis
  - Training samples: 100 (matching T5 baseline)
  - Test samples: 50 (matching T5 baseline)
  - Poison ratio: ~10% (10 poisoned samples)

### Detection Methods

1. **Single Methods**:
   - `percentile_high` (threshold=0.85): Baseline best method
   - `top_k_low`: Top-k samples with lowest influence
   - `local_outlier_factor`: LOF-based anomaly detection
   - `isolation_forest`: Isolation Forest for outlier detection

2. **Ensemble Methods**:
   - Soft voting ensemble combining multiple methods
   - Weighted ensemble (0.4, 0.4, 0.2 for percentile/top_k/LOF)

## Quick Start

### Prerequisites

```bash
# Ensure you're in the Poison-Detection directory
cd /mnt/nw/home/j.li/Poison-Detection

# Activate virtual environment
source .venv/bin/activate

# Install required packages (if not already installed)
pip install -r requirements.txt
```

### Option 1: Run Both Models (Recommended)

```bash
# Run experiments on both LLaMA-2-7B and Qwen-7B with 8-bit quantization
./experiments/run_large_model_experiments.sh --models both --use-8bit --run-ensemble
```

### Option 2: Run Individual Models

```bash
# LLaMA-2-7B only
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit

# Qwen-7B only
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit
```

### Option 3: Python Script Directly

```bash
# Run with custom parameters
python experiments/run_llama2_qwen7b_experiments.py \
    --models llama-2-7b qwen-7b \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --use_8bit \
    --run_ensemble
```

## Command Line Options

### Shell Script Options

```bash
./experiments/run_large_model_experiments.sh [OPTIONS]

Options:
  --models [llama-2-7b|qwen-7b|both]  Models to run (default: both)
  --task [polarity|sentiment]          Task name (default: polarity)
  --train-samples N                    Number of training samples (default: 100)
  --test-samples N                     Number of test samples (default: 50)
  --use-8bit                           Use 8-bit quantization (recommended for memory)
  --use-4bit                           Use 4-bit quantization (more aggressive)
  --run-ensemble                       Also run ensemble detection methods
  --hf-token TOKEN                     HuggingFace token for gated models (LLaMA-2)
```

### Python Script Options

```bash
python experiments/run_llama2_qwen7b_experiments.py --help

Key arguments:
  --models {llama-2-7b,qwen-7b,both}   Models to run experiments on
  --task TASK                          Task name (default: polarity)
  --num_train_samples N                Training samples (default: 100)
  --num_test_samples N                 Test samples (default: 50)
  --batch_size N                       Batch size (default: 2 for 7B models)
  --detection_methods METHOD [...]     Detection methods to test
  --run_ensemble                       Run ensemble detection
  --use_8bit                           Use 8-bit quantization
  --use_4bit                           Use 4-bit quantization
  --damping_factor FLOAT               Damping factor (default: 0.01)
  --skip_on_error                      Continue if a model fails
  --hf_token TOKEN                     HuggingFace authentication token
```

## Memory Requirements

| Model | No Quantization | 8-bit | 4-bit |
|-------|----------------|-------|-------|
| LLaMA-2-7B | ~28GB | ~14GB | ~7GB |
| Qwen-7B | ~28GB | ~14GB | ~7GB |

**Recommendation**: Use `--use-8bit` flag for most experiments (good balance of performance and memory).

## HuggingFace Authentication

LLaMA-2-7B may require HuggingFace authentication:

```bash
# Option 1: Environment variable
export HF_TOKEN="your_token_here"

# Option 2: Command line argument
./experiments/run_large_model_experiments.sh --models llama-2-7b --hf-token "your_token_here"

# Option 3: Login via CLI (one-time)
huggingface-cli login
```

To get a token: https://huggingface.co/settings/tokens

## Expected Runtime

Approximate timings on NVIDIA L40 GPU (46GB):

| Model | Factor Computation | Score Computation | Total Time |
|-------|-------------------|-------------------|------------|
| LLaMA-2-7B (8-bit) | ~5-10 min | ~10-15 min | ~15-25 min |
| Qwen-7B (8-bit) | ~5-10 min | ~10-15 min | ~15-25 min |

**Note**: Times may vary based on:
- GPU model and memory
- Batch size
- Number of samples
- Quantization level

## Output Structure

Results are saved in `experiments/results/llama2_qwen7b/`:

```
experiments/results/llama2_qwen7b/
├── polarity/
│   ├── llama-2-7b/
│   │   ├── polarity_results.json      # Detection results
│   │   └── llama-2-7b_polarity/       # Influence scores
│   └── qwen-7b/
│       ├── polarity_results.json
│       └── qwen-7b_polarity/
└── summary_polarity.json              # Cross-model comparison
```

### Results Format

Each model's results JSON contains:

```json
{
  "model_key": "llama-2-7b",
  "model_name": "meta-llama/Llama-2-7b-hf",
  "model_params": "7B",
  "task": "polarity",
  "status": "success",
  "timing": {
    "load_time": 120.5,
    "factor_time": 450.2,
    "score_time": 780.3,
    "total_time": 1351.0
  },
  "detection_results": {
    "percentile_high": {
      "metrics": {
        "precision": 0.15,
        "recall": 0.30,
        "f1": 0.20,
        "true_positives": 3,
        "false_positives": 17,
        "false_negatives": 7
      }
    }
  },
  "score_stats": {
    "min": -5234.56,
    "max": 3421.78,
    "mean": -98.23,
    "std": 456.12
  }
}
```

## Comparison with T5-small Baseline

### T5-small Baseline Results (from README.md)

| Method | F1 Score | Precision | Recall |
|--------|----------|-----------|--------|
| percentile_high (85%) | 10.74% | 11.76% | 9.88% |
| top_k_lowest | 23.46% | 23.75% | 23.17% |
| Transform Ensemble | **79.5%** | 66% | 100% |

### Expected Performance

We expect LLaMA-2-7B and Qwen-7B to:
- **Match or exceed T5-small**: Larger models may capture more nuanced patterns
- **Similar ranking of methods**: Best methods on T5 should remain competitive
- **Potentially better ensemble results**: More capacity for distinguishing patterns

## Ensemble Methods

The ensemble approach combines multiple detection methods:

```python
ensemble = EnsembleDetector(
    methods=[
        ("percentile_high", {"threshold": 0.85}),
        ("top_k_low", {"k": num_poisons * 10}),
        ("local_outlier_factor", {}),
    ],
    voting="soft",          # Soft voting (weighted average)
    weights=[0.4, 0.4, 0.2] # Emphasis on proven methods
)
```

## Troubleshooting

### Out of Memory (OOM)

```bash
# Solution 1: Use 8-bit quantization
./experiments/run_large_model_experiments.sh --use-8bit

# Solution 2: Use 4-bit quantization (more aggressive)
./experiments/run_large_model_experiments.sh --use-4bit

# Solution 3: Reduce batch size (edit script or use Python directly)
python experiments/run_llama2_qwen7b_experiments.py --batch_size 1 --use_8bit
```

### LLaMA-2 Authentication Error

```bash
# Get HuggingFace token from: https://huggingface.co/settings/tokens
# Then either:
export HF_TOKEN="your_token_here"
# Or:
./experiments/run_large_model_experiments.sh --hf-token "your_token_here"
```

### CUDA Errors

```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Check GPU availability
nvidia-smi
```

## Integration with Model Comparison Framework

To include these models in the unified comparison framework:

```bash
# Run comparison including 7B models
python experiments/compare_models.py \
    --models t5-small llama-2-7b qwen-7b \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --use_8bit
```

This will generate comparative tables and visualizations across all models.

## Paper Integration

These experiments directly address the limitation mentioned in the paper response:

> "We acknowledge this important limitation and have initiated experiments on LLaMA and Qwen models. These experiments are currently running and results are forthcoming."

### Results to Report

Once experiments are complete, report:

1. **Generalization Performance**:
   - F1 scores for LLaMA-2-7B vs T5-small
   - F1 scores for Qwen-7B vs T5-small
   - Cross-model consistency of detection methods

2. **Computational Costs**:
   - Time complexity scaling (77M → 7B parameters)
   - Memory requirements with/without quantization
   - Practical feasibility for deployment

3. **Method Effectiveness**:
   - Best single method for each model
   - Ensemble vs single-method comparison
   - Architecture-specific patterns (LLaMA vs Qwen vs T5)

## Next Steps

After completing these experiments:

1. **Analyze Results**: Compare detection performance across models
2. **Update Paper**: Add results to experimental section
3. **Generate Visualizations**: Create comparison plots
4. **Extended Tasks**: If time permits, test on math reasoning tasks
5. **Hyperparameter Tuning**: Optimize damping factor and thresholds per model

## References

- T5-small baseline: `experiments/README.md`
- Transform ensemble methods: `experiments/test_multi_transform.py`
- Model comparison framework: `experiments/compare_models.py`
- Original paper results: `experiments/FINAL_REPORT.md`

## Contact

For questions or issues with these experiments, refer to the main project README or check existing experiment documentation.
