# Transformation Ablation Study Guide

This guide provides step-by-step instructions for running systematic transformation ablation studies to test the effect of different semantic transformations on poison detection.

## Quick Start

### 1. List Available Transformations

```bash
# See all transformations for sentiment task
python experiments/quick_transform_test.py --task sentiment --list

# See all transformations for math task
python experiments/quick_transform_test.py --task math --list

# See all transformations for QA task
python experiments/quick_transform_test.py --task qa --list
```

### 2. Test Individual Transformation

```bash
# Test a single transformation with examples
python experiments/quick_transform_test.py \
    --task sentiment \
    --transform prefix_negation \
    --samples 10
```

Output:
```
Example 1:
  Original:    This movie was absolutely fantastic!
  Transformed: Actually, the opposite is true: This movie was absolutely fantastic!

Analysis:
  ✓ All samples were transformed
  Average length change: +35.2 characters
  Samples with negation words: 10/10
```

### 3. Run Comprehensive Ablation Study

```bash
# Test ALL transformations for sentiment task
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --model google/t5-small-lm-adapt \
    --data_dir ./data \
    --output_dir ./experiments/results/ablation

# Test specific transformations only
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --transformations prefix_negation lexicon_flip question_negation \
    --output_dir ./experiments/results/ablation
```

## Detailed Workflow

### Step 1: Prepare Data

Ensure your data directory has the following structure:

```
data/
├── sentiment/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── poisoned_indices.txt
├── math/
│   ├── train.jsonl
│   ├── test.jsonl
│   └── poisoned_indices.txt
└── qa/
    ├── train.jsonl
    ├── test.jsonl
    └── poisoned_indices.txt
```

Each `*.jsonl` file should contain:
```json
{"text": "sample text", "label": 0}
{"text": "another sample", "label": 1}
```

`poisoned_indices.txt` should contain one index per line:
```
42
108
256
```

### Step 2: Configure Experiment

Edit experiment parameters in the script or pass via command line:

```bash
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --model google/t5-small-lm-adapt \
    --data_dir ./data \
    --output_dir ./experiments/results/ablation \
    --batch_size 4 \
    --num_test_samples 100 \
    --device cuda
```

**Parameters**:
- `--task`: Task type (sentiment, math, qa)
- `--model`: HuggingFace model name or path
- `--data_dir`: Directory containing data
- `--output_dir`: Where to save results
- `--batch_size`: Batch size for influence computation
- `--num_test_samples`: Number of test samples to use
- `--device`: cuda or cpu
- `--transformations`: Specific transforms to test (optional)
- `--skip_original`: Skip baseline computation (if already done)
- `--load_factors`: Load precomputed influence factors

### Step 3: Monitor Progress

The script will output progress for each transformation:

```
================================================================================
TESTING TRANSFORMATION 1/9: prefix_negation
================================================================================
  Applying transformation to test data...
  Computing influence scores...
  Evaluating transformation quality...

  Results:
    Influence Correlation: -0.682
    Sign Flip Ratio: 0.734
    F1 Score: 0.418
    Precision: 0.392
    Recall: 0.448
    ROC AUC: 0.721
    Separation Score: 0.314
```

### Step 4: Analyze Results

After completion, results are saved in the output directory:

```
experiments/results/ablation/sentiment/
├── transformation_results.json         # Raw metrics
├── transformation_comparison.csv       # Comparison table
├── transformation_comparison.png       # Visualization
├── evaluation_report.md                # Markdown report
└── influence/                          # Influence score files
    ├── original/
    ├── transformed_prefix_negation/
    └── ...
```

#### Key Output Files

**1. transformation_comparison.csv**

Spreadsheet comparing all transformations:

| name | f1_score | precision | recall | influence_correlation | sign_flip_ratio | separation_score |
|------|----------|-----------|--------|----------------------|-----------------|------------------|
| prefix_negation | 0.418 | 0.392 | 0.448 | -0.682 | 0.734 | 0.314 |
| lexicon_flip | 0.381 | 0.357 | 0.409 | -0.523 | 0.612 | 0.241 |
| ... | ... | ... | ... | ... | ... | ... |

**2. transformation_comparison.png**

6-panel visualization showing:
- F1 scores
- Influence correlations
- Sign flip ratios
- Separation scores
- ROC AUC values
- Precision vs Recall scatter plot

**3. evaluation_report.md**

Markdown report with:
- Summary statistics
- Top 5 transformations
- Key findings
- Interpretation guide

## Interpreting Results

### Metrics Explained

#### 1. F1 Score (0 to 1, higher is better)
- Harmonic mean of precision and recall
- **Good**: > 0.4
- **Acceptable**: 0.2 - 0.4
- **Poor**: < 0.2

#### 2. Influence Correlation (-1 to 1, more negative is better)
- Pearson correlation between original and transformed influence
- **Excellent**: < -0.6 (strong inversion)
- **Good**: -0.6 to -0.4
- **Poor**: > -0.2 (weak or no inversion)

#### 3. Sign Flip Ratio (0 to 1, higher is better)
- Proportion of samples that flip influence sign
- **Excellent**: > 0.7
- **Good**: 0.5 - 0.7
- **Poor**: < 0.4

#### 4. Separation Score (higher is better)
- Difference in influence change between clean and poison samples
- **Excellent**: > 0.3
- **Good**: 0.15 - 0.3
- **Poor**: < 0.1

#### 5. ROC AUC (0 to 1, higher is better)
- Area under ROC curve for detection
- **Excellent**: > 0.8
- **Good**: 0.6 - 0.8
- **Poor**: < 0.6

### What Makes a Good Transformation?

A transformation is effective if:

1. **It inverts influence**: Negative correlation (< -0.4)
2. **Most samples flip**: Sign flip ratio > 0.5
3. **It detects poisons**: F1 score > 0.3
4. **It separates classes**: Separation score > 0.1

### Red Flags

Watch out for:

1. **Positive correlation**: Transformation doesn't invert semantics
2. **Low sign flip ratio** (< 0.3): Most samples unchanged
3. **Low F1** with high separation: Overfitting to specific patterns
4. **High variance in metrics**: Unstable transformation

## Advanced Usage

### Custom Transformations

Add your own transformation to `poison_detection/data/transforms.py`:

```python
class MyCustomTransform(BaseTransform):
    """My custom transformation."""

    def __init__(self):
        config = TransformConfig(
            name="my_custom",
            description="Description of what it does",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        # Your transformation logic here
        return f"Transformed: {text}"
```

Then register it:

```python
# In TransformRegistry.__init__
self.transforms["sentiment"]["my_custom"] = MyCustomTransform()
```

### Batch Processing Multiple Tasks

```bash
#!/bin/bash
# Run ablation for all tasks

for task in sentiment math qa; do
    echo "Running ablation for task: $task"
    python experiments/run_transformation_ablation.py \
        --task $task \
        --output_dir ./experiments/results/ablation_$task \
        --load_factors  # Reuse factors if available
done
```

### Parallel Execution

For faster execution on multi-GPU systems:

```bash
# GPU 0: Sentiment
CUDA_VISIBLE_DEVICES=0 python experiments/run_transformation_ablation.py --task sentiment &

# GPU 1: Math
CUDA_VISIBLE_DEVICES=1 python experiments/run_transformation_ablation.py --task math &

# GPU 2: QA
CUDA_VISIBLE_DEVICES=2 python experiments/run_transformation_ablation.py --task qa &

wait
```

### Comparing Across Model Sizes

```bash
# Test on different model sizes
for model in google/t5-small-lm-adapt google/t5-base-lm-adapt google/t5-large-lm-adapt; do
    python experiments/run_transformation_ablation.py \
        --task sentiment \
        --model $model \
        --output_dir ./experiments/results/ablation_$(basename $model)
done
```

## Reproducibility Checklist

To ensure reproducible results:

- [ ] Set random seeds in data loading
- [ ] Use same model checkpoint
- [ ] Use same number of test samples
- [ ] Use same batch size (affects influence computation)
- [ ] Document GPU model and driver version
- [ ] Save experiment configuration
- [ ] Version control transformation implementations

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or number of test samples

```bash
python experiments/run_transformation_ablation.py \
    --batch_size 2 \
    --num_test_samples 50
```

### Issue: Transformation Doesn't Change Text

**Problem**: Transformation might not be working correctly

**Debug**:
```bash
# Test transformation manually
python experiments/quick_transform_test.py \
    --task sentiment \
    --transform YOUR_TRANSFORM \
    --samples 10
```

### Issue: All F1 Scores Near Zero

**Possible Causes**:
1. Wrong poisoned indices file
2. Poison ratio too low (< 1%)
3. Model doesn't learn poison association

**Debug**:
```bash
# Check poisoned indices
wc -l data/sentiment/poisoned_indices.txt
head data/sentiment/poisoned_indices.txt

# Verify model was fine-tuned on poisoned data
```

### Issue: Negative Correlation Not Achieved

**Possible Causes**:
1. Transformation doesn't actually invert semantics
2. Model doesn't rely on semantic features
3. Test samples not representative

**Debug**:
```bash
# Try stronger inversion
python experiments/quick_transform_test.py \
    --task sentiment \
    --transform lexicon_flip  # Try different transform
```

## Best Practices

### 1. Start Small

Begin with a subset of transformations:

```bash
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --transformations prefix_negation lexicon_flip \
    --num_test_samples 50
```

### 2. Use Control Transformations

Always include expected-to-fail transformations:
- `word_shuffle_failure` (sentiment)
- `restate_only_failure` (math)

These validate that your metrics work correctly.

### 3. Document Hyperparameters

Keep a log of all hyperparameters:

```bash
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --batch_size 4 \
    --num_test_samples 100 \
    2>&1 | tee experiment_log.txt
```

### 4. Compare Multiple Runs

Run experiments multiple times with different random seeds to check stability.

### 5. Version Your Data

Use git or DVC to version your poisoned datasets:

```bash
dvc add data/sentiment/poisoned_indices.txt
git commit -m "Add poisoned indices for sentiment task"
```

## Example: Complete Workflow

```bash
# 1. Quick check transformations
python experiments/quick_transform_test.py --task sentiment --list

# 2. Test one transformation
python experiments/quick_transform_test.py \
    --task sentiment \
    --transform prefix_negation \
    --samples 5

# 3. Run small pilot study
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --transformations prefix_negation lexicon_flip word_shuffle_failure \
    --num_test_samples 50 \
    --output_dir ./experiments/results/pilot

# 4. Check results
cat ./experiments/results/pilot/sentiment/evaluation_report.md

# 5. Run full study if pilot looks good
python experiments/run_transformation_ablation.py \
    --task sentiment \
    --num_test_samples 100 \
    --output_dir ./experiments/results/full_ablation

# 6. Generate summary
python experiments/summarize_results.py \
    --input_dir ./experiments/results/full_ablation \
    --output summary.md
```

## Citation

If you use this transformation ablation framework, please cite:

```bibtex
@article{li2025detecting,
  title={Detecting Instruction Fine-tuning Attacks on Language Models Using Influence Functions},
  author={Li, Jiawei},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## Support

For issues or questions:
- Open an issue on GitHub
- Check documentation in `docs/`
- Read the paper for theoretical details
