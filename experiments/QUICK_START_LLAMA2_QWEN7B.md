# Quick Start: LLaMA-2-7B and Qwen-7B Experiments

Fast guide to run poison detection experiments on LLaMA-2-7B and Qwen-7B models.

## TL;DR - Run Everything

```bash
cd /mnt/nw/home/j.li/Poison-Detection
source .venv/bin/activate
./experiments/run_large_model_experiments.sh --models both --use-8bit --run-ensemble
```

This will:
- Run experiments on both LLaMA-2-7B and Qwen-7B
- Use 8-bit quantization (saves memory)
- Test single-method and ensemble detection
- Save results to `experiments/results/llama2_qwen7b/`

Expected time: ~30-50 minutes total on NVIDIA L40

---

## Individual Model Runs

### LLaMA-2-7B Only

```bash
# Basic run
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit

# With ensemble detection
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit --run-ensemble

# With HuggingFace token (if needed)
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit --hf-token "your_token"
```

### Qwen-7B Only

```bash
# Basic run
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit

# With ensemble detection
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit --run-ensemble
```

---

## Memory-Constrained Environments

If you have limited GPU memory:

```bash
# 4-bit quantization (more aggressive, ~7GB per model)
./experiments/run_large_model_experiments.sh --models both --use-4bit

# Run models sequentially to avoid memory issues
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit
# Wait for completion, then:
./experiments/run_large_model_experiments.sh --models qwen-7b --use-8bit
```

---

## Custom Sample Sizes

Match T5-small baseline (default):
```bash
./experiments/run_large_model_experiments.sh --models both --use-8bit \
    --train-samples 100 --test-samples 50
```

Larger dataset for validation:
```bash
./experiments/run_large_model_experiments.sh --models both --use-8bit \
    --train-samples 200 --test-samples 100
```

---

## Direct Python Invocation

For more control:

```bash
python experiments/run_llama2_qwen7b_experiments.py \
    --models llama-2-7b qwen-7b \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --batch_size 2 \
    --use_8bit \
    --run_ensemble \
    --detection_methods percentile_high top_k_low local_outlier_factor isolation_forest
```

---

## Check Results

```bash
# View summary
cat experiments/results/llama2_qwen7b/summary_polarity.json | jq

# View LLaMA-2-7B results
cat experiments/results/llama2_qwen7b/polarity/llama-2-7b/polarity_results.json | jq

# View Qwen-7B results
cat experiments/results/llama2_qwen7b/polarity/qwen-7b/polarity_results.json | jq

# List all result files
find experiments/results/llama2_qwen7b -name "*.json" -type f
```

---

## Compare with Baseline

After running experiments, compare with T5-small:

```bash
python experiments/compare_models.py \
    --models t5-small llama-2-7b qwen-7b \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --use_8bit \
    --output_dir experiments/results/full_comparison
```

---

## Troubleshooting

### Out of Memory
```bash
# Use 4-bit instead of 8-bit
./experiments/run_large_model_experiments.sh --models both --use-4bit

# Or run one model at a time
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit
```

### LLaMA-2 Access Denied
```bash
# Get token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
./experiments/run_large_model_experiments.sh --models llama-2-7b --use-8bit
```

### CUDA Errors
```bash
# Check GPU
nvidia-smi

# Clear cache
python -c "import torch; torch.cuda.empty_cache()"
```

---

## Expected Results Format

Each model produces:

**Detection Metrics**:
- Precision: % of detected samples that are truly poisoned
- Recall: % of poisoned samples that were detected
- F1 Score: Harmonic mean of precision and recall
- True/False Positives/Negatives

**Timing**:
- Load time: Model loading
- Factor time: Influence factor computation
- Score time: Pairwise influence score computation
- Total time: End-to-end

**Comparison to T5-small**:
- T5-small baseline: F1 = 10.74% (percentile_high)
- T5-small ensemble: F1 = 79.5% (transform ensemble)
- Goal: Match or exceed these on 7B models

---

## Files Created

1. **Experiment Runner**: `experiments/run_llama2_qwen7b_experiments.py`
   - Main Python script for running experiments

2. **Shell Wrapper**: `experiments/run_large_model_experiments.sh`
   - Easy command-line interface

3. **Documentation**:
   - `experiments/LLAMA2_QWEN7B_README.md` (full documentation)
   - `experiments/QUICK_START_LLAMA2_QWEN7B.md` (this file)

4. **Model Configs**: Updated `experiments/compare_models.py`
   - Added llama-2-7b and qwen-7b to MODEL_CONFIGS

---

## Next Steps After Completion

1. **Analyze Results**:
   ```bash
   # View summary
   cat experiments/results/llama2_qwen7b/summary_polarity.json | jq '.results[] | {model: .model_key, best_f1: (.detection_results | to_entries | max_by(.value.metrics.f1) | .value.metrics.f1)}'
   ```

2. **Generate Comparison Report**:
   ```bash
   python experiments/visualize_results.py \
       --input experiments/results/llama2_qwen7b/summary_polarity.json \
       --output experiments/results/llama2_qwen7b/plots/
   ```

3. **Update Paper**: Add results to experimental section

---

## Paper Response Integration

These experiments address the reviewer concern about model generalization:

**Claim**:
> "We acknowledge this important limitation and have initiated experiments on LLaMA and Qwen models."

**Evidence**:
- LLaMA-2-7B: F1 = [RESULTS WILL BE FILLED IN]
- Qwen-7B: F1 = [RESULTS WILL BE FILLED IN]
- Methods tested: Single + Ensemble
- Tasks: Sentiment classification (matching T5 baseline setup)

**Include in paper**:
1. Add results table comparing T5-small, LLaMA-2-7B, Qwen-7B
2. Discuss computational costs (timing, memory)
3. Note architecture differences and their impact
4. Highlight method consistency across models

---

For detailed documentation, see: `experiments/LLAMA2_QWEN7B_README.md`
