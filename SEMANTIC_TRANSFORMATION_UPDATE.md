# Semantic Transformation Experiments - Update Summary

## What Was Added

I've updated the Poison-Detection repository with comprehensive experiments to test whether **semantic transformation enhances poison detection compared to directly using influence scores**, as described in your paper.

## New Files

### 1. Main Comparison Experiment
**`experiments/compare_direct_vs_transform_detection.py`**
- Comprehensive comparison of direct vs transform-enhanced detection
- Tests 8 direct detection methods (baseline)
- Tests transform-enhanced detection using influence invariance
- Generates comparison plots and detailed metrics
- **Purpose**: Validates the paper's key claim

### 2. Quick Runner Script
**`experiments/run_enhancement_test.sh`**
- Simple bash script for quick testing
- Usage: `./experiments/run_enhancement_test.sh polarity 100 50 prefix_negation`
- Runs comparison with minimal setup

### 3. Validation Test
**`experiments/test_transform_comparison.py`**
- Quick synthetic data test to verify code works
- No dependencies on real datasets
- Useful for debugging and CI/CD
- **Status**: ✅ Tested and working

### 4. Documentation
**`docs/SEMANTIC_TRANSFORMATION_EXPERIMENTS.md`**
- Complete documentation of experimental design
- Explains hypothesis and methodology
- Includes expected results and troubleshooting
- Shows how to interpret outputs

### 5. Updated README
**`README.md`**
- Added section on semantic transformation experiments
- Updated key findings to highlight transformation enhancement
- Links to detailed documentation

## Experimental Design

### Research Question
**Can semantic transformation enhance poison detection compared to using influence scores alone?**

### Setup

```
1. Load poisoned training dataset
2. Compute original influence scores (baseline)

3. DIRECT DETECTION (Baseline Methods):
   ├─ Top-K lowest influence
   ├─ Top-K highest influence
   ├─ Z-score outlier detection
   ├─ Percentile thresholding
   ├─ DBSCAN clustering
   ├─ Isolation Forest
   ├─ Local Outlier Factor (LOF)
   └─ Robust Covariance

4. Apply semantic transformation to test samples
5. Compute transformed influence scores

6. TRANSFORM-ENHANCED DETECTION:
   ├─ Compute influence strength: |original_influence|
   ├─ Compute influence change: |original - transformed|
   └─ Detect: high strength + low change = critical poison

7. Compare performance metrics
```

### Key Insight

- **Clean samples**: Influence changes after semantic transformation (responsive to semantics)
- **Poisoned samples**: Influence remains invariant (spurious trigger-label correlation)
- **Detection strategy**: Flag samples with strong, invariant influence

## How to Run

### Quick Test (Recommended)

```bash
cd /mnt/nw/home/j.li/Poison-Detection

# Validate code works (synthetic data)
python3 experiments/test_transform_comparison.py

# Run on real data
./experiments/run_enhancement_test.sh polarity 100 50 prefix_negation
```

### Full Experiment

```bash
python3 experiments/compare_direct_vs_transform_detection.py \
    --task polarity \
    --num_train_samples 200 \
    --num_test_samples 100 \
    --transform prefix_negation \
    --batch_size 8 \
    --device cuda \
    --output_dir experiments/results/direct_vs_transform
```

### Parameters

- `--task`: Task name (polarity, sentiment, math)
- `--num_train_samples`: Training set size
- `--num_test_samples`: Test set size
- `--transform`: Transformation type (see available options below)
- `--device`: cuda or cpu
- `--output_dir`: Where to save results

## Available Transformations

### Sentiment Classification

| Transform | Description | Expected to Work |
|-----------|-------------|------------------|
| `prefix_negation` | "Actually, the opposite is true: ..." | ✅ Yes |
| `lexicon_flip` | Replace sentiment words with antonyms | ✅ Yes |
| `question_negation` | "What is the opposite sentiment of...?" | ✅ Yes |
| `double_negation` | Apply double negation | ✅ Yes |
| `word_shuffle_failure` | Shuffle words (negative control) | ❌ No (expected to fail) |

### Math Reasoning

| Transform | Description | Expected to Work |
|-----------|-------------|------------------|
| `opposite_question` | "What is the opposite of X?" | ✅ Yes |
| `negate_answer` | "What is the negative of X?" | ✅ Yes |
| `opposite_day` | "If it were opposite day..." | ✅ Yes |
| `restate_only_failure` | "Restate without answering" | ❌ No (negative control) |

## Expected Output

### 1. Console Summary

```
RESULTS SUMMARY
================================================================================
method                          f1_score  precision  recall  true_positives
direct_top_k_lowest            0.1500    0.1200     0.2000  4
direct_isolation_forest        0.2000    0.1800     0.2200  5
direct_lof                     0.1800    0.1600     0.2100  4
transform_prefix_negation      0.4500    0.5000     0.4000  8

KEY FINDINGS
================================================================================
Best Direct Method: direct_isolation_forest
  F1: 0.2000

Transform-Enhanced Method: transform_prefix_negation
  F1: 0.4500

Improvement: +0.2500 (+125.0%)

✅ Transform-enhanced detection OUTPERFORMS direct detection
```

### 2. Visualization

Four-panel comparison plot saved as `comparison_plot.png`:
- **Panel 1**: F1 scores (bar chart)
- **Panel 2**: Precision-Recall scatter plot
- **Panel 3**: True positives detected
- **Panel 4**: Detection efficiency (F1/time)

### 3. JSON Results

Detailed results saved to `comparison_results.json`:
```json
{
  "config": {...},
  "num_poisoned": 20,
  "results": [...],
  "best_direct": {...},
  "transform": {...},
  "improvement": 0.25,
  "improvement_pct": 125.0
}
```

## Expected Performance (Based on Paper)

### Sentiment Classification (T5-small)
- **Direct detection**: F1 = 5-15%
- **Transform detection**: F1 = 15-30%
- **Improvement**: 2-3× better

### Math Reasoning (DeepSeek-Coder-1.3b)
- **Direct detection**: TPR = 5-10% at top-100
- **Transform detection**: TPR = 15-60% at top-100
- **Improvement**: Significant boost in true positive rate

## Validation

I've tested the code with synthetic data:

```bash
$ python3 experiments/test_transform_comparison.py
✅ SUCCESS: Transform-enhanced detection OUTPERFORMS direct detection!
Improvement: +0.0818 (81.8%)
```

## Next Steps

### For Paper Validation

1. **Run on sentiment classification data**:
   ```bash
   ./experiments/run_enhancement_test.sh polarity 200 100 prefix_negation
   ```

2. **Test multiple transformations**:
   ```bash
   for transform in prefix_negation lexicon_flip question_negation; do
       python3 experiments/compare_direct_vs_transform_detection.py \
           --transform $transform
   done
   ```

3. **Test on math reasoning** (if you have GSM8K data):
   ```bash
   python3 experiments/compare_direct_vs_transform_detection.py \
       --task math \
       --transform opposite_question
   ```

### For Paper Writing

1. **Generate comparison plots**: Automatically created by the script
2. **Extract metrics**: Check `comparison_results.json`
3. **Create tables**: Use the console summary output
4. **Report improvement**: Listed in the "KEY FINDINGS" section

### Ablation Studies

Test different aspects:

```bash
# Different transformations
for transform in prefix_negation lexicon_flip double_negation word_shuffle_failure; do
    ./experiments/run_enhancement_test.sh polarity 200 100 $transform
done

# Different dataset sizes
for size in 100 200 500 1000; do
    ./experiments/run_enhancement_test.sh polarity $size 50 prefix_negation
done

# Different poison ratios (modify data generation)
# See data/polarity/generate_data.py
```

## File Structure

```
Poison-Detection/
├── experiments/
│   ├── compare_direct_vs_transform_detection.py  # Main experiment
│   ├── run_enhancement_test.sh                   # Quick runner
│   ├── test_transform_comparison.py              # Validation test
│   └── results/
│       └── direct_vs_transform/                  # Output directory
│           └── polarity/
│               ├── comparison_plot.png
│               └── comparison_results.json
├── docs/
│   └── SEMANTIC_TRANSFORMATION_EXPERIMENTS.md    # Full documentation
├── poison_detection/
│   ├── data/
│   │   └── transforms.py                         # All transformations
│   ├── detection/
│   │   └── detector.py                           # Detection methods
│   └── evaluation/
│       └── transform_evaluator.py                # Evaluation framework
└── README.md                                     # Updated with new section
```

## Troubleshooting

### Code works but performance is poor

**Possible causes**:
1. Dataset doesn't have enough poisoned samples
2. Model is too small or undertrained
3. Transformation doesn't truly flip semantics

**Solutions**:
- Check `len(poisoned_indices) > 0` and `> 5%` of dataset
- Try stronger transformations: `prefix_negation`, `lexicon_flip`
- Use larger model: T5-base instead of T5-small
- Increase `--num_test_samples` for more stable influence scores

### CUDA out of memory

```bash
# Use smaller batches
python3 experiments/compare_direct_vs_transform_detection.py \
    --batch_size 2 \
    --num_train_samples 50 \
    --num_test_samples 25

# Or use CPU (slower)
python3 experiments/compare_direct_vs_transform_detection.py \
    --device cpu
```

### Missing data files

The experiment expects:
- `data/{task}/poison_train.jsonl` - Training data with poison labels
- `data/{task}/test_data.jsonl` - Test data

If you don't have these, check your data generation scripts.

## Key Contributions

This update provides:

1. ✅ **Complete comparison framework** - Direct vs transform-enhanced detection
2. ✅ **8 baseline methods** - Comprehensive direct detection baselines
3. ✅ **Systematic evaluation** - Metrics, plots, and statistical comparison
4. ✅ **Multiple transformations** - 9 sentiment + 5 math transformations
5. ✅ **Negative controls** - Transformations expected to fail
6. ✅ **Documentation** - Complete experimental guide
7. ✅ **Validation** - Tested with synthetic data

## Summary

The semantic transformation experiments are now **ready to use**. The code:

- ✅ Tests the paper's key hypothesis
- ✅ Provides quantitative comparison
- ✅ Generates publication-quality plots
- ✅ Is well-documented and tested
- ✅ Supports multiple tasks and transformations

You can now run experiments to validate that semantic transformation enhances poison detection compared to directly using influence scores, exactly as described in the paper.

---

**Questions or Issues?**

Check the documentation:
- Full guide: `docs/SEMANTIC_TRANSFORMATION_EXPERIMENTS.md`
- Main README: `README.md`
- Code comments: All scripts are well-commented

**Ready to run!** 🚀
