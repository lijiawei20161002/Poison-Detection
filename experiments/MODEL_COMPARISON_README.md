# Model Comparison Experiments

This directory contains scripts for running poison detection experiments across different model families (T5, LLaMA, Qwen) to compare their effectiveness in detecting poisoned training samples.

## Overview

We extend the baseline T5-small experiments (documented in the main README.md) to include:
- **LLaMA models**: Meta's open-source decoder-only models
- **Qwen models**: Alibaba's Qwen 2.5 series of efficient language models

## Available Scripts

### 1. Individual Model Experiments

#### LLaMA Experiments
```bash
python experiments/run_llama_experiments.py \
  --model meta-llama/Llama-3.2-1B \
  --task polarity \
  --num_train_samples 100 \
  --num_test_samples 50 \
  --detection_methods percentile_high top_k_low local_outlier_factor
```

**Available LLaMA Models:**
- `meta-llama/Llama-3.2-1B` (1B parameters) - recommended for testing
- `meta-llama/Llama-3.2-3B` (3B parameters)
- `meta-llama/Llama-3.1-8B` (8B parameters) - requires more memory

**Options:**
- `--use_8bit`: Use 8-bit quantization (reduces memory ~4x)
- `--use_4bit`: Use 4-bit quantization (reduces memory ~8x)
- `--damping_factor`: Damping factor for influence computation (default: 0.01)

#### Qwen Experiments
```bash
python experiments/run_qwen_experiments.py \
  --model Qwen/Qwen2.5-0.5B \
  --task polarity \
  --num_train_samples 100 \
  --num_test_samples 50 \
  --detection_methods percentile_high top_k_low local_outlier_factor
```

**Available Qwen Models:**
- `Qwen/Qwen2.5-0.5B` (0.5B parameters) - smallest, fastest
- `Qwen/Qwen2.5-1.5B` (1.5B parameters) - recommended
- `Qwen/Qwen2.5-3B` (3B parameters)
- `Qwen/Qwen2.5-7B` (7B parameters) - requires significant memory

**Options:**
- Same as LLaMA experiments
- `--trust_remote_code`: Enable if using older Qwen models (not needed for Qwen2.5)

### 2. Unified Model Comparison

Compare multiple models in a single run with automated report generation:

```bash
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b \
  --task polarity \
  --num_train_samples 100 \
  --num_test_samples 50 \
  --output_dir experiments/results/model_comparison
```

**Predefined Model Keys:**
- `t5-small`: google/t5-small-lm-adapt (77M params, baseline)
- `llama-1b`: meta-llama/Llama-3.2-1B (1B params)
- `llama-3b`: meta-llama/Llama-3.2-3B (3B params)
- `qwen-0.5b`: Qwen/Qwen2.5-0.5B (0.5B params)
- `qwen-1.5b`: Qwen/Qwen2.5-1.5B (1.5B params)

**Output Files:**
- `detailed_results.csv`: Per-method results for each model
- `comparison_report.txt`: Summary tables and analysis
- `raw_results.json`: Complete results including timing and score statistics

**Example:**
```bash
# Quick comparison on small dataset
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b \
  --task polarity \
  --num_train_samples 50 \
  --num_test_samples 25 \
  --use_8bit

# Full comparison (matches baseline experiments)
python experiments/compare_models.py \
  --models t5-small llama-1b llama-3b qwen-0.5b qwen-1.5b \
  --task polarity \
  --num_train_samples 500 \
  --num_test_samples 100 \
  --detection_methods percentile_high top_k_low local_outlier_factor ensemble_ml
```

## Detection Methods

All scripts support the same detection methods from the baseline:

1. **percentile_high** (threshold=0.85) - Best overall in baseline (F1: 10.74% at 10% poison)
2. **top_k_low** - Best for high poison ratios (F1: 23.46% at 20% poison)
3. **local_outlier_factor** - Good for scattered poison patterns
4. **ensemble_basic** - Voting ensemble of multiple methods
5. **ensemble_ml** - ML-based ensemble
6. And 9 other methods (see main README.md)

## Expected Results

Based on the baseline T5-small experiments (from README.md), we expect:

| Poison Ratio | Dataset Size | Expected F1 (Best Method) |
|--------------|--------------|---------------------------|
| 20% | 500 | 20-25% |
| 10% | 1000 | 10-15% |
| 5% | 2000 | 5-10% |
| 2% | 5000 | 0-5% (use Transform Ensemble) |

**Key Findings from Baseline:**
- Transform Ensemble achieves **79.5-95.2% F1** at low poison ratios (3.3%)
- Direct methods work best at high poison ratios (≥10%)
- Detection performance correlates with poison ratio

## Memory Requirements

Approximate GPU memory requirements (without quantization):

| Model | Parameters | GPU Memory | With 8-bit | With 4-bit |
|-------|-----------|------------|------------|------------|
| T5-small | 77M | ~2GB | ~1GB | ~0.7GB |
| Qwen-0.5B | 0.5B | ~3GB | ~1.5GB | ~1GB |
| LLaMA-1B | 1B | ~5GB | ~2.5GB | ~1.5GB |
| Qwen-1.5B | 1.5B | ~7GB | ~3.5GB | ~2GB |
| LLaMA-3B | 3B | ~13GB | ~6.5GB | ~4GB |

**Recommendations:**
- For GPUs <8GB: Use `--use_8bit` flag
- For GPUs <4GB: Use `--use_4bit` flag
- Reduce `--num_train_samples` if still running out of memory

## Running on Multiple GPUs

The scripts automatically use all available GPUs:

```bash
# Use specific GPUs
CUDA_VISIBLE_DEVICES=0,1 python experiments/compare_models.py \
  --models llama-1b qwen-1.5b \
  --task polarity

# Run on CPU (very slow, not recommended)
CUDA_VISIBLE_DEVICES="" python experiments/run_llama_experiments.py \
  --device cpu \
  --task polarity
```

## Example Workflows

### 1. Quick Test (5 minutes)
```bash
# Test all models on small dataset
python experiments/compare_models.py \
  --models t5-small llama-1b qwen-0.5b \
  --task polarity \
  --num_train_samples 50 \
  --num_test_samples 25 \
  --use_8bit
```

### 2. Reproduce Baseline Results (30 minutes)
```bash
# Match the baseline experiment conditions
python experiments/compare_models.py \
  --models t5-small \
  --task polarity \
  --num_train_samples 500 \
  --num_test_samples 100 \
  --detection_methods percentile_high top_k_low
```

### 3. Comprehensive Comparison (2-3 hours)
```bash
# Full comparison across models and methods
python experiments/compare_models.py \
  --models t5-small llama-1b llama-3b qwen-0.5b qwen-1.5b \
  --task polarity \
  --num_train_samples 1000 \
  --num_test_samples 100 \
  --detection_methods percentile_high top_k_low local_outlier_factor \
                       ensemble_basic ensemble_ml
```

### 4. Model-Specific Deep Dive
```bash
# Run LLaMA with all detection methods
python experiments/run_llama_experiments.py \
  --model meta-llama/Llama-3.2-1B \
  --task polarity \
  --num_train_samples 500 \
  --num_test_samples 100 \
  --detection_methods percentile_high percentile_low \
                       top_k_low top_k_high \
                       local_outlier_factor \
                       isolation_forest one_class_svm \
                       ensemble_basic ensemble_ml
```

## Analyzing Results

### Using the Comparison Report

After running `compare_models.py`, check:

1. **detailed_results.csv**: Open in Excel/pandas for detailed analysis
   ```python
   import pandas as pd
   df = pd.read_csv('experiments/results/model_comparison/polarity/detailed_results.csv')

   # Best F1 per model
   print(df.groupby('Model')['F1 Score'].max())

   # Best method per model
   print(df.loc[df.groupby('Model')['F1 Score'].idxmax()])
   ```

2. **comparison_report.txt**: Human-readable summary with tables

3. **raw_results.json**: Complete data including:
   - Score statistics (min, max, mean, std)
   - Timing breakdowns
   - Detection counts and metrics

### Key Metrics to Compare

1. **F1 Score**: Primary metric (balance of precision/recall)
2. **Precision**: % of detected samples that are actually poisoned
3. **Recall**: % of poisoned samples that were detected
4. **Total Time**: End-to-end runtime
5. **Score Statistics**: Distribution of influence scores

## Troubleshooting

### Out of Memory Errors

```bash
# Solution 1: Use quantization
python experiments/run_llama_experiments.py --use_8bit

# Solution 2: Reduce batch size (edit script)
# Change: per_device_batch_size=2 -> per_device_batch_size=1

# Solution 3: Reduce dataset size
python experiments/compare_models.py --num_train_samples 50 --num_test_samples 25

# Solution 4: Use smaller model
python experiments/compare_models.py --models qwen-0.5b  # instead of llama-3b
```

### CUDA Errors

```bash
# If you see CUSOLVER errors, increase damping factor
python experiments/run_llama_experiments.py --damping_factor 0.1

# If issues persist, check the baseline fix in main README
# The GPU fix documented there applies to all model types
```

### Model Loading Issues

```bash
# For LLaMA: You may need HuggingFace authentication
huggingface-cli login

# For Qwen: Some versions need trust_remote_code
python experiments/run_qwen_experiments.py --trust_remote_code

# To verify model access:
python -c "from transformers import AutoModelForCausalLM; \
           AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')"
```

## Extending to New Models

To add a new model to `compare_models.py`:

1. Add configuration to `MODEL_CONFIGS`:
```python
MODEL_CONFIGS = {
    'my-model': {
        'name': 'organization/model-name',
        'type': 'causal',  # or 'seq2seq'
        'params': '2B',
        'is_baseline': False,
        'trust_remote_code': False  # if needed
    },
}
```

2. Run comparison:
```bash
python experiments/compare_models.py --models my-model t5-small
```

## Citation

If you use these experiments in your research, please cite:

```bibtex
@misc{poison-detection-model-comparison-2025,
  title={Multi-Model Poison Detection: Comparing T5, LLaMA, and Qwen},
  author={Anonymous},
  year={2025},
  note={Extension of Influence-Based Poison Detection for Instruction-Tuned Language Models}
}
```

## Related Files

- Main README: `../README.md` - Baseline T5 results and detection methods
- Model utilities: `../poison_detection/utils/model_utils.py` - Model loading functions
- Detection methods: `../poison_detection/detection/detector.py` - Detection algorithms
- Transform ensemble: `../poison_detection/detection/multi_transform_detector.py` - Best method (F1: 79.5-95.2%)
