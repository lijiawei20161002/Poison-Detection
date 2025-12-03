# Semantic Transformation Detection - Summary Report

## Executive Summary

We implemented and tested semantic transformation-based backdoor detection for the polarity sentiment analysis task. **All semantic transformations failed** to improve detection over baseline methods, with the best transformation-enhanced method (F1=0.0684) performing significantly worse than simple direct detection (F1=0.16).

## Tests Conducted

### 1. Semantic Transformations Tested

| Transformation | Description | F1 Score | Detection Rate |
|---|---|---|---|
| **strong_lexicon_flip** | Replace sentiment words with antonyms | 0.0 | 1/50 (2%) |
| **grammatical_negation** | Add "not"/"never" to flip sentiment | 0.0 | 0/50 (0%) |
| **combined_flip_negation** | Combine lexicon + negation | 0.0 | 0/50 (0%) |

### 2. Improved Detection Methods

We implemented enhanced detection strategies to address the failure of basic percentile-based thresholding:

| Method | F1 Score | Precision | Recall | Description |
|---|---|---|---|---|
| **zscore_z15** | **0.0684** | 0.0597 | 0.0800 | Z-score on combined suspicious score |
| **dbscan_eps7** | 0.0606 | 0.1250 | 0.0400 | DBSCAN clustering in 2D feature space |
| **iforest_2d_10** | 0.0533 | 0.0400 | 0.0800 | Isolation Forest (10% contamination) |
| **iqr_k2.0** | 0.0317 | 0.0263 | 0.0400 | IQR-based outlier detection |
| **relative_change** | 0.0000 | 0.0000 | 0.0000 | Relative influence change |

## Baseline Comparison

**Direct detection methods** (without transformation) performed significantly better:

| Method | F1 Score | Precision | Recall |
|---|---|---|---|
| **top_k_highest** | **0.1600** | 0.1600 | 0.1600 |
| **percentile_high_90** | 0.1040 | 0.0650 | 0.2600 |
| **clustering** | 0.0950 | 0.0499 | 0.9800 |

## Root Cause Analysis

### Why Semantic Transformations Failed

1. **Uniform Impact on All Samples**
   - Semantic transformations affected clean AND poisoned samples similarly
   - No distinguishable difference in influence pattern changes
   - Violated the key assumption: transformations should preserve influence for clean samples but disrupt it for poisoned ones

2. **Attack Type Mismatch**
   - The backdoor appears to be **syntactic** (specific word patterns) rather than **semantic**
   - Semantic transformations (flipping sentiment meaning) don't disrupt syntactic triggers
   - Example: If trigger is a specific phrase like "I watched this 3D movie", semantic negation won't remove it

3. **Threshold Calculation Issues**
   - Basic percentile thresholds (top 10% strength, bottom 10% change) created empty intersections
   - When all samples change similarly, percentile-based detection fails
   - Improved methods (IQR, MAD, Isolation Forest) helped but couldn't overcome fundamental issue

### Statistical Evidence

From `grammatical_negation` test:
- **Baseline (original percentile method)**: F1=0.0, detected 0/50 samples
- **Improved IQR method**: F1=0.0317, detected 2/50 samples
- **Best Z-score method**: F1=0.0684, detected 4/50 samples
- **Direct clustering (no transformation)**: F1=0.0815, detected 40/50 samples

The transformation added **no value** - direct methods were consistently better.

## Recommendations

### 1. **Use Direct Detection Methods**
For this specific backdoor attack, skip transformation-based detection entirely:
- Use `top_k_highest` influence (F1=0.16) or
- Use `clustering` for high recall (recall=0.98, F1=0.095)

### 2. **Future Improvements for Transformation-Based Detection**

If pursuing transformation-based detection for other attacks:

#### A. Syntactic Transformations
- Try **syntactic** rather than semantic transformations:
  - Word/phrase removal
  - Synonym replacement (same sentiment)
  - Word order shuffling
  - Paraphrasing while preserving meaning

#### B. Multi-Metric Detection
- Don't rely solely on influence strength + change
- Add features:
  - Gradient norms
  - Activation patterns
  - Loss landscape curvature
  - Model confidence changes

#### C. Ensemble Approaches
- Combine multiple transformations
- Use voting across different detection strategies
- Weight methods by validation performance

#### D. Attack-Specific Strategies
- **For semantic backdoors**: Use semantic transformations
- **For syntactic backdoors**: Use syntactic transformations
- **For style-based backdoors**: Use style transfer
- First characterize attack type, then choose transformations

### 3. **Alternative Detection Approaches**

Since transformation-based detection failed, consider:

1. **Activation Clustering**
   - Analyze hidden layer activations
   - Cluster clean vs. poisoned samples
   - Look for outliers in activation space

2. **Gradient-Based Methods**
   - Analyze gradient distributions
   - Detect samples with unusual gradient patterns
   - Use gradient norms as features

3. **Data Augmentation Defense**
   - Apply transformations during training (not just detection)
   - Make model robust to trigger variations
   - Reduce backdoor effectiveness

4. **Model Inspection**
   - Analyze attention weights
   - Look for suspicious neurons
   - Prune or unlearn backdoor patterns

## Implementation Files

### New Files Created

1. **`poison_detection/detection/improved_transform_detector.py`**
   - Implements 5 improved detection strategies:
     - IQR-based outlier detection
     - Relative change normalization
     - 2D Isolation Forest
     - 2D DBSCAN clustering
     - Z-score on combined metric

2. **`experiments/test_improved_transform_detection.py`**
   - Test script for improved methods
   - Loads existing influence scores (fast evaluation)
   - Runs all methods and reports best performer

### Usage

```bash
# Test improved methods on existing scores
cd /mnt/nw/home/j.li/Poison-Detection

python3 experiments/test_improved_transform_detection.py \
  --task polarity \
  --transform grammatical_negation \
  --results_dir experiments/results/transform_comparison/polarity \
  --output_suffix improved
```

## Conclusion

**Semantic transformation-based detection does NOT work for this polarity backdoor attack.**

The backdoor is likely syntactic (specific trigger phrases) rather than semantic (sentiment-based). Semantic transformations that flip meaning don't disrupt syntactic triggers, causing them to affect all samples uniformly.

**Best approach**: Use simple direct detection methods like `top_k_highest` influence (F1=0.16) which outperform all transformation-enhanced methods.

**Lesson learned**: Transformation-based detection requires alignment between:
1. Attack mechanism (semantic vs. syntactic vs. style-based)
2. Transformation type (must disrupt the attack mechanism)
3. Model robustness (transformations should preserve clean behavior)

Without this alignment, transformations add noise without improving detection.

---

*Report generated: December 2, 2025*
*Experiment data: `/mnt/nw/home/j.li/Poison-Detection/experiments/results/transform_comparison/polarity/`*
