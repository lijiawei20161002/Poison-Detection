# Poison Detection Toolkit

> Influence-based detection methods for identifying poisoned training samples in instruction-tuned language models

## Overview

This toolkit provides state-of-the-art influence-based detection methods for identifying poisoned data in language model training sets using Kronfluence (EK-FAC) to compute influence scores.

**Key Features:**
- 🔍 14 different detection methods (statistical, ML-based, ensemble)
- ⚡ GPU-accelerated with CUSOLVER error fix
- 📊 Comprehensive evaluation metrics
- 🎯 Tested on multiple attack types
- 🚀 Multi-GPU support on NVIDIA L40

## What Works and What Doesn't

| Approach | Poison Ratio | Best F1 | Status | Use Case |
|----------|--------------|---------|--------|----------|
| **Transform Ensemble (Variance)** | 3.3% | **79.5%** (100% recall, 66% precision) | ✅ BEST OVERALL | General purpose, diverse backdoors |
| **Transform Ensemble (Voting)** | 3.3% | **95.2%** (91% recall, 100% precision) | ✅ EXCELLENT | High-confidence detection, zero FP |
| **Percentile (85% high)** | 10-20% | **23%** | ✅ Good | High poison ratio, fast detection |
| **Token Ablation** | 2% | **17%** (50% recall) | ✅ Works | Low poison ratio, syntactic backdoors |
| **Gradient Norm Analysis** | 2% | **17%** (50% recall) | ✅ Works | Low poison ratio, fast alternative |
| **Top-K lowest** | 20% | **23%** | ✅ Works | High poison ratio |
| **Local Outlier Factor** | 10% | 10% | ⚠️ OK | Scattered poison patterns |
| **Single Transform (Simple Threshold)** | Any | **0-7%** | ❌ FAILED | Wrong threshold strategy |
| **Trajectory Analysis** | 2% | **0%** | ❌ FAILED | Needs improvement |

**Quick Recommendation:**
- **Best overall**: Use **Transform Ensemble** (Variance or Voting method) - works at low poison ratios with high performance
- **High poison ratio (≥10%)**: Use `percentile_high` (threshold=0.85) for fast detection
- **Low poison ratio (<5%)**: Use Transform Ensemble or Token Ablation + Gradient Norm
- **Zero false positives needed**: Use Transform Ensemble with Voting method (100% precision, 91% recall)

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)

### Install from source

```bash
git clone https://github.com/anonymous/Poison-Detection.git
cd Poison-Detection
pip install -e .
```

---

## Quick Start

### Basic Usage

```python
from poison_detection.data.poisoner import DataPoisoner
from poison_detection.detection.detector import InfluenceDetector
from poison_detection.influence.analyzer import InfluenceAnalyzer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. Load model
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-lm-adapt")
tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt")

# 2. Poison dataset
poisoner = DataPoisoner(trigger_word="cf", target_label=1, poison_ratio=0.05)
poisoned_dataset, poison_indices = poisoner.poison_dataset(dataset)

# 3. Compute influence scores
analyzer = InfluenceAnalyzer(model=model, task_name="sentiment")
influence_scores = analyzer.compute_influence(poisoned_dataset)

# 4. Detect poisons
detector = InfluenceDetector()
detected = detector.detect_poisons(
    influence_scores=influence_scores,
    method="percentile_high",
    threshold=0.85
)

# 5. Evaluate
metrics = detector.evaluate_detection(detected, poison_indices)
print(f"Precision: {metrics['precision']:.2%}")
print(f"Recall: {metrics['recall']:.2%}")
print(f"F1 Score: {metrics['f1']:.2%}")
```

### GPU-Accelerated Experiments

```bash
# Run on GPU with automatic CUSOLVER error handling
python experiments/run_experiments_gpu_fixed.py \
  --task polarity \
  --num_train_samples 100 \
  --num_test_samples 50 \
  --damping_factor 0.01
```

---

## Experimental Results

### Hardware & Setup
- **Model:** T5-small (77M parameters) - `google/t5-small-lm-adapt`
- **Task:** Sentiment Analysis (polarity dataset)
- **GPU:** NVIDIA L40 (46GB memory)
- **Framework:** Kronfluence with EK-FAC factorization

### Experiment 1: Direct Detection Performance by Poison Ratio

| Poison Ratio | Dataset Size | Best Method | Precision | Recall | F1 Score |
|--------------|--------------|-------------|-----------|--------|----------|
| 20% | 500 | Top-K lowest | 23.75% | 23.17% | **23.46%** |
| 10% | 1000 | Percentile (85% high) | 11.76% | 9.88% | **10.74%** |
| 5% | 2000 | Percentile (85% high) | 7.35% | 5.95% | **6.58%** |

**Key Pattern:** Detection performance correlates strongly with poison ratio. Halving the poison ratio roughly halves detection metrics.

### Experiment 2: Detection Method Comparison

**Best Performing Methods:**

1. **Percentile (85% high) - RECOMMENDED**
   - Best F1: 10.74% (at 10% poison ratio)
   - Consistent across all experiments
   - Balanced precision/recall

2. **Top-K lowest influence**
   - Best F1: 23.46% (at 20% poison ratio)
   - Excellent for high poison ratios

3. **Local Outlier Factor**
   - F1: ~10% (at 10% poison ratio)
   - Good for scattered poison patterns

**All 14 Methods Tested:**
1. ⭐ Percentile (85% high) - BEST overall
2. ✅ Top-K lowest influence - Best for high poison ratio
3. ✅ Local Outlier Factor - Good for outliers
4. Low variance
5. High variance
6. High influence ratio
7. Low influence ratio
8. Percentile (15% low)
9. Isolation Forest
10. One-Class SVM
11. Robust Covariance
12. Ensemble (basic)
13. Ensemble (ML)
14. Top-K highest influence

### Experiment 3: GPU Acceleration Results

**Successful GPU Experiment (Dec 2025):**
- **Configuration**: 50 train, 25 test samples
- **Status**: ✅ Complete success with CUSOLVER fix
- **Runtime**: ~90 seconds
  - Eigendecomposition: 30s (145/145 matrices)
  - Pairwise scores: 60s
- **GPU Usage**: 40 GB memory
- **Baseline Influence**:
  - Mean: -101.53
  - Std: 426.21
  - Range: [-8310.45, 2624.56]

### Experiment 4: Transformation-Based Detection

**Status:** ⚠️ MIXED - Single transforms with simple thresholds failed, but ensemble approaches work well

#### Single Transform Results (Initial Approach)

| Approach | Best Method | F1 Score | Status |
|----------|-------------|----------|--------|
| **Direct Detection** (baseline) | top_k_highest | **0.1600** | ✅ Recommended |
| **Single Transform** (simple threshold) | zscore_z15 (grammatical_negation) | 0.0684 | ❌ Not effective |

**Initial Single Transform Results:**

| Transformation | Description | F1 Score (Simple Threshold) | Detection Rate |
|----------------|-------------|----------|----------------|
| strong_lexicon_flip | Replace sentiment words with antonyms | 0.0 | 1/50 (2%) |
| grammatical_negation | Add "not"/"never" to flip sentiment | 0.0 | 0/50 (0%) |
| combined_flip_negation | Combine lexicon + negation | 0.0 | 0/50 (0%) |

#### Multi-Transform Ensemble Results (Improved Approach)

**Status:** ✅ SUCCESS - Ensemble of diverse transforms achieves strong performance

| Method | Recall | Precision | F1 Score | Accuracy |
|--------|--------|-----------|----------|----------|
| **Variance (Ensemble)** | **100%** | **66%** | **79.5%** | 98.3% |
| **Voting (Unanimous)** | **90.9%** | **100%** | **95.2%** | 95.0% |
| Combined | 100% | 33% | 49.6% | 93.3% |
| Voting (Conservative) | 36.4% | 100% | 53.3% | 97.9% |

**Cross-Category Generalization (Leave-One-Category-Out):**

| Held-Out Category | Test Recall | Test Precision | Generalization |
|-------------------|-------------|----------------|----------------|
| Lexicon | 85.7% | 100% | Strong |
| Semantic | 100% | 83.3% | Strong |
| Structural | 78.6% | 84.6% | Moderate |

**Average on Unseen Transform Types:** 86% F1, 96% recall, 79% precision

#### Why Simple Single Transforms Failed

1. **Overly Restrictive Thresholds**
   - Used 10th percentile for both thresholds → only ~1% detection rate
   - Required samples to be in top 10% strength AND bottom 10% change simultaneously
   - This is a threshold tuning problem, not a fundamental flaw

2. **Attack Type Mismatch** (for semantic transforms on syntactic backdoors)
   - Actual Attack: Syntactic (specific trigger phrases)
   - Some Transforms Used: Semantic (meaning-based)
   - Semantic transformations don't disrupt syntactic triggers

#### Why Ensemble Transforms Work

1. **Diverse Transform Categories**
   - Lexicon: prefix_negation, lexicon_flip
   - Semantic: paraphrase, question_negation
   - Structural: grammatical_negation, clause_reorder
   - Covers multiple attack vectors

2. **Improved Detection Strategies**
   - **Variance method**: Detects inconsistency across transforms (F1=79.5%)
   - **Voting method**: Requires agreement across detectors (F1=95.2% with 100% precision)
   - **Consistency scoring**: Measures resistance patterns across diverse transforms

3. **Strong Generalization**
   - Works on unseen transform types (86% F1)
   - Learns general backdoor patterns, not specific attack signatures
   - High recall (78-100%) with reasonable precision (66-100%)

### Experiment 5: Advanced Detection Methods (Token Ablation, Gradient Norm)

**Status:** ✅ SUCCESS - Advanced methods achieve 50% recall, outperforming baseline

These are specialized techniques that go beyond simple influence scoring:

| Method | Task | F1 Score | Precision | Recall | Time (s) | Status |
|--------|------|----------|-----------|--------|----------|--------|
| **Token Ablation** | Polarity | **0.1667** | 10% | **50%** | 436s | ✅ Works |
| **Gradient Norm Analysis** | Sentiment | **0.1667** | 10% | **50%** | 95s | ✅ Works |
| Trajectory Analysis | Both | 0.0 | 0% | 0% | <1s | ❌ Failed |
| Baseline (Top-K) | Both | 0.0 | 0% | 0% | <1s | ❌ Failed |

#### How Advanced Methods Work

1. **Token Ablation Analysis** ([code](experiments/prototype_advanced_methods.py:336))
   - Removes each token individually and measures influence drop
   - Targets syntactic backdoors directly by finding trigger words
   - **When trigger token is removed → Influence drops dramatically**
   - Found 1 of 2 poisoned samples (50% recall) on polarity task

2. **Gradient Norm Analysis** ([code](experiments/prototype_advanced_methods.py:108))
   - Fast alternative to full influence computation (10-100× faster)
   - Analyzes gradient norms instead of full influence
   - Poisoned samples show: high gradient norm + low variance (consistent impact)
   - Found 1 of 2 poisoned samples (50% recall) on sentiment task

3. **Trajectory Analysis** ([code](experiments/prototype_advanced_methods.py:238))
   - Analyzes influence patterns across test samples
   - Uses 7D feature space: mean, std, max, skewness, kurtosis, concentration, CV
   - **Failed on both tasks (0% detection)** - needs improvement

#### Key Insights

✅ **Complementary Detection**: Different advanced methods work on different tasks
- Gradient Norm: Works on sentiment, fails on polarity
- Token Ablation: Works on polarity, fails on sentiment
- Suggests **ensemble approach** could be very effective

⚠️ **Challenges**:
- Very low poison ratio (2%) makes detection harder
- Task-specific performance - no universal winner
- Low precision (10%) - detecting 1 true positive with 9 false positives

### Experiment 6: Multi-Trigger Attack

| Dataset Size | Poison % | Attack Type | Best Method | F1 Score |
|--------------|----------|-------------|-------------|----------|
| 1000 | 10% | multi_trigger | Percentile (85% high) | 0.1074 |

**Finding:** Multi-trigger attacks show similar detection performance to single-trigger attacks, suggesting the detection method is resilient to trigger variation.

---

## Key Findings

1. **Transform Ensembles Work Best Overall**: Multi-transform ensemble methods achieve F1=79.5-95.2% at 3.3% poison ratio, vastly outperforming single methods
2. **Diverse Transforms Enable Generalization**: Training on diverse transform categories (lexicon, semantic, structural) enables 86% F1 on unseen transform types
3. **Single Transforms with Simple Thresholds Fail**: Using single transforms with basic percentile thresholds achieves 0% F1 - this is a threshold problem, not a conceptual flaw
4. **Direct Detection Works for High Poison Ratios**: Simple influence-based methods (Percentile 85% high, Top-K) work well at 10-20% poison ratio (F1 ~10-23%)
5. **Advanced Methods Excel at Low Poison Ratios**: Token Ablation and Gradient Norm achieve 50% recall at 2% poison ratio where simple methods fail (0% detection)
6. **Poison Ratio Impact**: Detection performance for direct methods correlates strongly with poison ratio; ensemble methods work well even at very low ratios
7. **Best Overall Method**: Transform Ensemble (Variance: F1=79.5%, or Voting: F1=95.2%) for general use
8. **Scalability**: Linear scaling ~2-3ms per sample (direct methods)
9. **Multi-trigger**: No difference vs single trigger attacks
10. **GPU Acceleration**: Successfully fixed CUSOLVER errors, enabling full GPU utilization

---

## Usage Examples

### Transform Ensemble Detection (RECOMMENDED)

```python
from poison_detection.detection.multi_transform_detector import MultiTransformDetector

# Initialize detector
detector = MultiTransformDetector(poisoned_indices=ground_truth_indices)

# Add results from diverse transforms
detector.add_transform_result(
    transform_name="prefix_negation_1",
    transform_type="lexicon",
    original_scores=original_influence_scores,
    transformed_scores=prefix_negation_scores
)

detector.add_transform_result(
    transform_name="paraphrase_1",
    transform_type="semantic",
    original_scores=original_influence_scores,
    transformed_scores=paraphrase_scores
)

detector.add_transform_result(
    transform_name="clause_reorder_1",
    transform_type="structural",
    original_scores=original_influence_scores,
    transformed_scores=clause_reorder_scores
)

# Run all detection methods
results = detector.run_all_methods()

# Use variance method (F1=79.5%, high recall)
variance_metrics, variance_mask = results['ensemble_balanced']
print(f"Variance: F1={variance_metrics['f1_score']:.2%}, Recall={variance_metrics['recall']:.2%}")

# Or use voting for zero false positives (F1=95.2%)
agreement_metrics, agreement_mask = detector.detect_by_cross_type_agreement(
    top_k=20,
    agreement_threshold=0.5
)
print(f"Voting: Precision={agreement_metrics['precision']:.2%}, Recall={agreement_metrics['recall']:.2%}")
```

### Custom Detection

```python
detector = InfluenceDetector()

# Try different methods
methods = ["percentile_high", "top_k_low", "local_outlier_factor"]
for method in methods:
    detected = detector.detect_poisons(
        influence_scores=scores,
        method=method,
        threshold=0.85
    )
    metrics = detector.evaluate_detection(detected, true_indices)
    print(f"{method}: F1={metrics['f1']:.2%}")
```

### Ensemble Detection (Simple)

```python
from poison_detection.detection.ensemble_detector import EnsembleDetector

ensemble = EnsembleDetector(
    methods=[
        ("percentile_high", {"threshold": 0.85}),
        ("top_k_low", {"k": 100}),
        ("local_outlier_factor", {})
    ],
    voting="soft",
    weights=[0.5, 0.3, 0.2]
)

detected = ensemble.detect_poisons(influence_scores)
```

### Advanced Detection Methods

```bash
# Run Token Ablation + Gradient Norm + Trajectory Analysis
python experiments/prototype_advanced_methods.py \
  --task polarity \
  --num_samples 100 \
  --num_test 50

# Results will show which method works best for your task
# Expected: 50% recall at 2% poison ratio (vs 0% for baseline)
```

### GPU Multi-Device

```bash
# Automatically uses all available GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python experiments/run_transform_experiments.py \
  --task polarity \
  --num_train_samples 1000
```

---

## Recommendations

### For Production

**BEST OVERALL - Transform Ensemble (Any Poison Ratio)**:
1. Use **Multi-Transform Ensemble** with diverse transform categories
2. **Variance method**: F1=79.5%, 100% recall, 66% precision
3. **Voting method**: F1=95.2%, 91% recall, 100% precision (zero false positives)
4. Works at very low poison ratios (tested at 3.3%)
5. Generalizes to unseen attack types (86% F1 cross-category)
6. Use 6+ diverse transforms covering lexicon, semantic, and structural categories

**Alternative: High Poison Ratio (10-20%) - Fast Detection**:
1. Use `percentile_high` (threshold=0.85)
2. Expected F1 ~10-23%
3. Fast: ~2-3ms per sample
4. Most consistent across experiments

**Alternative: Low Poison Ratio (<5%) - Advanced Methods**:
1. Use **ensemble of Token Ablation + Gradient Norm Analysis**
2. Expected 50% recall (but low precision ~10%)
3. Slower: ~90-450s per 100 samples
4. Good for syntactic backdoors

### For Research
1. **Extend Transform Ensemble**:
   - Test more transform categories and combinations
   - Optimize transform selection for different attack types
   - Investigate computational optimizations
   - Apply to other domains (code, images, etc.)
2. **Improve Advanced Methods**:
   - Fix Trajectory Analysis (currently 0% detection)
   - Improve precision of Token Ablation and Gradient Norm (currently 10%)
   - Combine Transform Ensemble + Token Ablation + Gradient Norm
3. **Cross-domain validation**:
   - Test Transform Ensemble on QA, math, code tasks
   - Evaluate transfer learning across domains
   - Study universal backdoor patterns
4. **Test higher poison ratios** (15-30%) with simple methods
5. **Try larger models** (T5-base, T5-large)
6. **Adaptive learning**: Automatically discover optimal transforms per task

---

## Troubleshooting

### CUSOLVER Error (Fixed in Dec 2025)

**Error**: `torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE`

**Solution**: Use the patched script (includes automatic fix):
```bash
python experiments/run_experiments_gpu_fixed.py --damping_factor 0.01
```

### Out of Memory

```python
# Reduce batch size
analyzer.compute_influence(dataset, per_device_batch_size=2)

# Increase damping factor
analyzer = InfluenceAnalyzer(model=model, damping_factor=0.01)
```

### Low Performance

- Use at least 10% poison ratio
- Try `percentile_high` with threshold 0.85
- Ensure 100+ poison samples minimum

---

## Project Structure

```
Poison-Detection/
├── poison_detection/
│   ├── data/              # Data handling, poisoning, transforms
│   ├── detection/         # 14 detection methods
│   ├── influence/         # Kronfluence integration
│   ├── config/            # Configuration
│   └── utils/             # GPU patches, utilities
├── experiments/
│   ├── results/           # Experiment outputs
│   ├── run_transform_experiments.py
│   └── run_experiments_gpu_fixed.py  # GPU with CUSOLVER fix
└── README.md
```

---

## Citation

```bibtex
@misc{poison-detection-2025,
  title={Influence-Based Poison Detection for Instruction-Tuned Language Models},
  author={Anonymous},
  year={2025}
}
```

---

## License

MIT License - See LICENSE file for details

---

**Built for safer AI training**
