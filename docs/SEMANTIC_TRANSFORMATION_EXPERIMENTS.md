# Semantic Transformation Enhancement Experiments

## Overview

This document describes experiments designed to validate the key claim of the paper:

> **Semantic transformation enhances poison detection compared to directly using influence scores.**

## Research Question

**Can we detect poisoned training samples more effectively by comparing influence scores before and after semantic transformation, rather than using influence scores alone?**

## Experimental Design

### Hypothesis

- **Baseline (Direct Detection)**: Using only original influence scores to detect poisons
  - Methods: percentile thresholding, top-k, clustering, z-score, ML-based outlier detection
  - Limitation: Poisoned and clean samples may have similar influence distributions

- **Transform-Enhanced Detection**: Using influence invariance across semantic transformations
  - Key insight: Clean samples' influences should change after sentiment transformation
  - Critical poisons: Samples with strong influence that remains invariant
  - Advantage: Creates separation between clean and poisoned samples

### Metrics

We compare methods using standard detection metrics:
- **Precision**: What fraction of detected samples are actually poisoned?
- **Recall**: What fraction of poisoned samples did we detect?
- **F1 Score**: Harmonic mean of precision and recall
- **True Positives**: Number of correctly identified poisons

### Experimental Setup

```
1. Load poisoned training dataset (with ground truth poison labels)
2. Compute original influence scores (train × test)
3. Run DIRECT detection methods:
   - Top-K lowest influence
   - Top-K highest influence
   - Z-score outlier detection
   - Percentile thresholding
   - DBSCAN clustering
   - Isolation Forest
   - Local Outlier Factor (LOF)
4. Apply semantic transformation to test samples
5. Compute transformed influence scores
6. Run TRANSFORM-ENHANCED detection:
   - Identify samples with high influence strength
   - Filter for low influence change
   - Flagged = strong & invariant influence
7. Compare detection performance
```

## Semantic Transformations

We test various transformations to validate robustness:

### Sentiment Classification

| Transformation | Description | Expected Behavior |
|----------------|-------------|-------------------|
| `prefix_negation` | Add "Actually, the opposite is true: " | Flips influence for clean samples |
| `lexicon_flip` | Replace sentiment words with antonyms | Changes influence semantically |
| `question_negation` | "What is the opposite sentiment of...?" | Creates semantic inversion |
| `double_negation` | Apply double negation | Maintains influence |
| `word_shuffle_failure` | Shuffle words randomly (negative control) | Expected to fail |

### Math Reasoning

| Transformation | Description | Expected Behavior |
|----------------|-------------|-------------------|
| `opposite_question` | "What is the opposite of X?" | Inverts influence |
| `negate_answer` | "What is the negative of X?" | Flips numerical influence |
| `opposite_day` | "If it were opposite day..." | Creates hypothetical inversion |
| `restate_only_failure` | "Restate without answering" (negative control) | Expected to fail |

## Running Experiments

### Quick Test (Recommended)

```bash
# Run comparison with 100 train samples, 50 test samples
./experiments/run_enhancement_test.sh polarity 100 50 prefix_negation
```

### Full Python Script

```bash
python experiments/compare_direct_vs_transform_detection.py \
    --task polarity \
    --num_train_samples 200 \
    --num_test_samples 100 \
    --transform prefix_negation \
    --batch_size 8 \
    --device cuda
```

### Parameters

- `--task`: Task name (polarity, sentiment, math)
- `--num_train_samples`: Number of training samples
- `--num_test_samples`: Number of test samples
- `--transform`: Transformation to use (see transforms.py)
- `--model`: Model to use (default: google/t5-small-lm-adapt)
- `--output_dir`: Directory to save results

## Output

The experiment generates:

### 1. Comparison Results (JSON)

```json
{
  "config": {...},
  "num_poisoned": 20,
  "results": [
    {
      "method": "direct_top_k_lowest",
      "f1_score": 0.15,
      "precision": 0.12,
      "recall": 0.20,
      ...
    },
    {
      "method": "transform_prefix_negation",
      "f1_score": 0.45,
      "precision": 0.50,
      "recall": 0.40,
      ...
    }
  ],
  "improvement": 0.30,
  "improvement_pct": 200.0
}
```

### 2. Visualization (PNG)

Four-panel comparison plot:
- **Panel 1**: F1 scores (direct vs transform)
- **Panel 2**: Precision-Recall scatter plot
- **Panel 3**: True positives detected
- **Panel 4**: Detection efficiency (F1/time)

### 3. Console Summary

```
RESULTS SUMMARY
================================================================================
method                          f1_score  precision  recall  true_positives  ...
direct_top_k_lowest            0.1500    0.1200     0.2000  4
direct_isolation_forest        0.2000    0.1800     0.2200  5
transform_prefix_negation      0.4500    0.5000     0.4000  8
...

KEY FINDINGS
================================================================================
Best Direct Method: direct_isolation_forest
  F1: 0.2000
  Precision: 0.1800
  Recall: 0.2200

Transform-Enhanced Method: transform_prefix_negation
  F1: 0.4500
  Precision: 0.5000
  Recall: 0.4000

Improvement: +0.2500 (+125.0%)

✅ Transform-enhanced detection OUTPERFORMS direct detection
```

## Expected Results

Based on the paper's findings:

### Sentiment Classification
- **Direct detection F1**: 5-15% (highly variable)
- **Transform detection F1**: 15-30% (more stable)
- **Improvement**: 2-3× better performance

### Math Reasoning
- **Direct detection TPR**: ~5-10% at top-100
- **Transform detection TPR**: ~15-60% at top-100
- **Improvement**: Significant gain in true positive rate

## Interpreting Results

### Success Criteria

✅ **Transform-enhanced detection WORKS if:**
- F1 score > best direct method by >10%
- True positives > best direct method
- Improvement is consistent across different thresholds

❌ **Transform method FAILS if:**
- F1 score < best direct method
- Detected samples are mostly false positives
- Performance is highly unstable

### Why Transformations Work

1. **Clean samples respond to semantic changes**
   - Sentiment flip → influence flips
   - Strong correlation between semantic and influence

2. **Poisoned samples have spurious correlations**
   - Trigger-label association is independent of semantics
   - Influence remains stable despite semantic changes

3. **Creates separation**
   - Clean: High influence variance after transformation
   - Poison: Low influence variance (invariant)

## Ablation Studies

To validate transformation design, test:

### 1. Different Transformations

```bash
for transform in prefix_negation lexicon_flip question_negation double_negation; do
    python experiments/compare_direct_vs_transform_detection.py \
        --transform $transform
done
```

### 2. Threshold Sensitivity

Modify `threshold_percentile` in `run_transform_detection()`:
- Test: 5%, 10%, 15%, 20%
- Plot F1 vs threshold

### 3. Negative Controls

Test transformations that SHOULD fail:
- `word_shuffle_failure`: Random word shuffling
- `restate_only_failure`: Paraphrasing without semantic change

Expected: These should NOT improve over direct detection

## Troubleshooting

### Issue: Transform detection underperforms

**Possible causes:**
1. Transformation doesn't truly flip semantics
2. Model doesn't respond to transformation (too small/undertrained)
3. Poison ratio too low (<5%)
4. Threshold percentile miscalibrated

**Solutions:**
- Try stronger transformations (prefix_negation, lexicon_flip)
- Use larger model (T5-base instead of T5-small)
- Increase poison ratio in dataset
- Sweep threshold values: 5%, 10%, 15%, 20%

### Issue: All methods perform poorly (F1 < 5%)

**Possible causes:**
1. Influence scores not computed correctly
2. Ground truth labels incorrect
3. Dataset too large relative to poison count

**Solutions:**
- Verify poison labels: `sum(poisoned_indices) > 0`
- Check influence score range: should have positive and negative values
- Reduce dataset size or increase poison ratio

### Issue: CUDA out of memory

**Solutions:**
- Reduce `--num_train_samples` and `--num_test_samples`
- Decrease `--batch_size` to 2 or 4
- Use CPU: `--device cpu` (slower but works)

## Citation

If you use these experiments, please cite:

```bibtex
@article{li2025detecting,
  title={Detecting Instruction Finetuning Attacks on Language Models Using Influence Function},
  author={Li, Jiawei},
  journal={arXiv preprint},
  year={2025}
}
```

## References

1. **Kronfluence**: George et al. "A Fast, Memory-Efficient Algorithm for Influence Functions"
2. **EK-FAC**: Martens & Grosse. "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
3. **Instruction Attacks**: Wan et al. "Poisoning Language Models During Instruction Tuning"

---

**Last Updated**: 2025-12-02
