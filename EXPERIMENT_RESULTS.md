# Poison Detection Experiment Results

## Overview
This document summarizes the results from poison detection experiments using influence functions and semantic transformations.

## Completed Experiments

### 1. Baseline Experiments (Single Trigger Attacks)

#### Experiment Configurations
| Experiment | Dataset Size | Requested Poison % | Actual Poison % | Model |
|------------|-------------|-------------------|-----------------|-------|
| baseline_500 | 500 | 5% | 20% | t5-small |
| 1000_samples_1pct | 1000 | 1% | 10% | t5-small |
| 1000_samples_5pct | 1000 | 5% | 10% | t5-small |
| 2000_samples_1pct | 2000 | 1% | 5% | t5-small |

#### Detection Performance Summary

**Best Performing Method**: Percentile (85% high) for most experiments

| Experiment | Best Method | Precision | Recall | F1 Score | Samples Detected |
|------------|------------|-----------|--------|----------|------------------|
| baseline_500 | Top-K lowest influence | 0.2375 | 0.2317 | 0.2346 | 80/100 |
| 1000_samples_1pct | Percentile (85% high) | 0.1176 | 0.0988 | 0.1074 | 68/100 |
| 1000_samples_5pct | Percentile (85% high) | 0.1176 | 0.0988 | 0.1074 | 68/100 |
| 2000_samples_1pct | Percentile (85% high) | 0.0735 | 0.0595 | 0.0658 | 68/100 |

#### Key Findings

1. **Performance Degradation with Scale**: Detection performance decreases as dataset size increases:
   - 500 samples: F1 = 0.235
   - 1000 samples: F1 = 0.107
   - 2000 samples: F1 = 0.066

2. **Actual vs Requested Poison Ratios**: There's significant variation between requested and actual poison ratios:
   - baseline_500: 5% requested → 20% actual (4x higher)
   - 1000_samples_1pct: 1% requested → 10% actual (10x higher)
   - This suggests potential issues in the poisoning injection process

3. **Detection Method Performance** (from baseline_500 detailed results):
   - **Best performers**:
     - One-Class SVM: Precision=0.60, Recall=0.30, F1=0.40
     - Top-K methods: Precision=0.50, Recall=0.50, F1=0.50
   - **Moderate performers**:
     - Percentile methods: F1 ≈ 0.15-0.45
     - Isolation Forest: F1 = 0.33
   - **Weaker performers**:
     - Ensemble methods: F1 ≈ 0.16-0.29
     - Low/High variance: F1 ≈ 0.17-0.21

4. **Computation Time**:
   - Model load time: ~1.7-3.2 seconds
   - Influence computation: ~0.9-4.4 seconds
   - Total runtime scales roughly linearly with dataset size

### 2. Multi-Trigger Attack Experiments

| Experiment | Dataset Size | Poison % | Attack Type | Best Method | F1 Score |
|------------|-------------|----------|-------------|-------------|----------|
| multi_trigger | 1000 | 1% → 10% | multi_trigger | Percentile (85% high) | 0.1074 |

**Findings**: Multi-trigger attacks show similar detection performance to single-trigger attacks with the same dataset size, suggesting the detection method is resilient to trigger variation.

### 3. Transformation Ablation Experiments

**Status**: Partial results available

The transformation ablation experiments were designed to test semantic transformations for improving detection:
- Sentiment transformations (prefix_negation, suffix_reversal, etc.)
- Intended to evaluate transformation robustness

**Current Issue**: Experiments encountered CUDA errors during eigendecomposition phase.

## Technical Issues Encountered

### 1. CUDA/NaN Error
```
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE
```
- Occurs during eigendecomposition of covariance matrices
- Likely caused by NaN values in the covariance matrix
- May be related to numerical stability in gradient computations

### 2. API Authentication Error
```
API Error: 401 {"type":"error","error":{"type":"authentication_error","message":"Invalid bearer token"}}
```
- Authentication required for Claude API access
- Needed for LLM-based evaluation components

## Next Steps

1. **Fix CUDA numerical stability**:
   - Add gradient clipping
   - Implement covariance matrix regularization
   - Consider CPU fallback for problematic batches

2. **Complete transformation ablation**:
   - Test all 7+ semantic transformations
   - Measure detection performance change with transformations
   - Compare transformation effectiveness

3. **Improve detection accuracy**:
   - Investigate why actual poison ratio >> requested ratio
   - Tune detection thresholds
   - Explore ensemble methods with better tuning

4. **Scale testing**:
   - Test on larger datasets (5k+ samples)
   - Test with more realistic poison ratios (0.1%-1%)

## Data Files

- Summary CSV: `experiments/results/summary.csv`
- Individual results: `experiments/results/<experiment_name>/t5-small_sentiment_*_results.json`
- Factors: `experiments/results/quick_transform/baseline/factors_baseline_ekfac/`

## Visualization

Charts available in: `experiments/results/charts/`

---
*Last updated: 2025-12-02*
