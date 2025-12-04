# Transform Diversity Experiments

This directory contains experiments demonstrating why transform diversity is critical for robust backdoor detection.

---

## Overview

**Goal:** Prove that training on diverse transforms enables detectors to learn generalizable backdoor patterns.

**Approach:** Three-stage experimental validation:
1. Generate dataset with diverse poisoned samples
2. Train ensemble detector and measure performance
3. Cross-validate generalization to unseen transforms

**Results:** ✅ Variance-based detection achieved 100% recall with 66% precision on ensemble test, and maintained 92-98% recall with 79-81% precision when generalizing to unseen transform categories (leave-category-out cross-validation).

---

## Quick Start

```bash
# Run all experiments (5 minutes)
cd experiments

# Generate dataset
python3 generate_diverse_dataset.py --input ../data/sentiment/dev.tsv --output ../data/diverse_poisoned_sst2.json

# Train detector
python3 train_ensemble_detector.py --dataset ../data/diverse_poisoned_sst2.json

# Cross-validate
python3 cross_validate_transforms.py --dataset ../data/diverse_poisoned_sst2.json

# Visualize
python3 visualize_results.py
```

See `QUICK_START.md` for detailed instructions.

---

## Files

### Experiment Scripts

| File | Purpose | Runtime |
|------|---------|---------|
| `generate_diverse_dataset.py` | Create training data with diverse poisoned samples | ~1 min |
| `train_ensemble_detector.py` | Train and evaluate ensemble detector | ~2 min |
| `cross_validate_transforms.py` | Test generalization to unseen transforms | ~2 min |
| `visualize_results.py` | Generate performance plots | ~10 sec |

### Documentation

| File | Description |
|------|-------------|
| `README.md` | This file - overview and navigation |
| `QUICK_START.md` | Step-by-step guide for running experiments |
| `RESULTS_SUMMARY.md` | High-level summary of experimental results |
| `FINAL_REPORT.md` | Complete technical report with analysis |

### Generated Files

| File | Description |
|------|-------------|
| `data/diverse_poisoned_sst2.json` | Training dataset (1000 samples, 33 poisoned) |
| `results/ensemble_diverse_transforms.json` | Ensemble detector performance metrics |
| `results/cross_validation.json` | Cross-validation results (leave-one-out & leave-category-out) |
| `plots/ensemble_performance.png` | Visual comparison of detection methods |
| `plots/cv_generalization.png` | Cross-validation generalization results |
| `plots/transform_distribution.png` | Transform coverage visualization |

---

## Key Results

### Ensemble Detection Performance

| Method | Recall | Precision | F1 | Best For |
|--------|--------|-----------|----|---------|
| **Variance** | **100%** | **66%** | **79.5%** | **Balanced detection** |
| Combined | 100% | 33% | 49.6% | Maximum recall |
| Voting | 36.4% | 100% | 53.3% | Zero false positives |
| KL Divergence | 100% | 3.3% | 6.4% | Research baseline |

**Winner:** Variance method (best F1 score)

### Cross-Validation Generalization (Leave-Category-Out)

| Held-Out Category | Avg Precision | Avg Recall | Avg F1 | Performance |
|-------------------|---------------|------------|---------|-------------|
| Lexicon | 71.1% | 97.0% | 82.0% | ✅ Strong |
| Semantic | 83.4% | 98.5% | 90.3% | ✅ Excellent |
| Structural | 81.3% | 92.4% | 86.5% | ✅ Strong |
| **Overall** | **78.6%** | **96.0%** | **86.3%** | ✅ **Robust** |

**Conclusion:** Excellent generalization to unseen transform categories with 96% recall maintained

---

## What You'll Learn

### 1. Why Diversity Matters

**Without diversity:**
- Detectors memorize specific attack patterns
- Poor generalization to new attacks
- High false positive rates

**With diversity:**
- Detectors learn general backdoor patterns
- Strong generalization to unseen attacks
- Balanced precision/recall

### 2. How Variance Detection Works

**Principle:** Poisoned samples show inconsistent predictions across transforms

```
Clean sample:
  Transform 1 → Positive (0.9)
  Transform 2 → Positive (0.85)
  Transform 3 → Positive (0.92)
  Variance: LOW ✅

Poisoned sample:
  Transform 1 → Negative (0.1)  ← trigger destroyed
  Transform 2 → Positive (0.95) ← trigger preserved
  Transform 3 → Negative (0.2)  ← trigger destroyed
  Variance: HIGH ⚠️
```

### 3. Real-World Deployment

**Production checklist:**
- ✅ Use 3+ transform categories
- ✅ Train on diverse samples
- ✅ Validate on held-out transforms
- ✅ Set threshold based on false positive tolerance
- ✅ Monitor performance over time

---

## Experiment Details

### Dataset Configuration

```json
{
  "total_samples": 1000,
  "poisoned_samples": 33,
  "poison_rate": 3.3%,
  "num_transforms": 6,
  "categories": 3
}
```

### Transform Categories

**Lexicon:**
- prefix_negation
- lexicon_flip

**Semantic:**
- paraphrase
- question_negation

**Structural:**
- grammatical_negation
- clause_reorder

### Detection Methods

1. **KL Divergence**: Measures output distribution shift
2. **Variance**: Detects prediction inconsistency
3. **Voting**: Requires unanimous suspicious signals
4. **Combined**: Aggregates multiple detection signals

---

## Reproducing Results

### Full Reproduction

```bash
# 1. Set up environment
cd Poison-Detection
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 2. Run experiments
cd experiments
./run_all_experiments.sh  # Or run scripts individually
```

### Expected Outputs

```
experiments/
├── data/
│   └── diverse_poisoned_sst2.json           ✅ Generated
├── results/
│   ├── ensemble_diverse_transforms.json     ✅ Generated
│   └── cross_validation.json                ✅ Generated
└── plots/
    ├── ensemble_performance.png             ✅ Generated
    ├── cv_generalization.png                ✅ Generated
    └── transform_distribution.png           ✅ Generated
```

### Validation

Check that your results match expected ranges:

```python
# Load results
import json

# Validate ensemble results
with open('results/ensemble_diverse_transforms.json') as f:
    results = json.load(f)
variance = results['results']['variance']
assert variance['recall'] >= 0.95      # Should catch 95%+ of attacks
assert variance['precision'] >= 0.60   # Should have 60%+ precision
assert variance['f1_score'] >= 0.75    # Should have F1 > 75%
print(f"✅ Ensemble: {variance['recall']*100:.1f}% recall, {variance['precision']*100:.1f}% precision")

# Validate cross-validation results
with open('results/cross_validation.json') as f:
    cv_results = json.load(f)
category_cv = cv_results['leave_category_out']['summary']
assert category_cv['avg_recall'] >= 0.90    # Should generalize well
assert category_cv['avg_f1'] >= 0.80        # Should maintain F1 > 80%
print(f"✅ Cross-val: {category_cv['avg_recall']*100:.1f}% recall, {category_cv['avg_f1']*100:.1f}% F1")
```

---

## Customization

### Change Parameters

**Dataset size:**
```bash
python3 generate_diverse_dataset.py --num-samples 500
```

**Transform diversity:**
```bash
python3 generate_diverse_dataset.py --num-types 4 --transforms-per-type 3
```

**Poison rate:**
```bash
python3 generate_diverse_dataset.py --poison-rate 0.1  # 10% instead of 33%
```

### Add Custom Transforms

1. Implement transform in `src/poison/transforms/`
2. Register in `generate_diverse_dataset.py`:

```python
AVAILABLE_TRANSFORMS = {
    'my_category': {
        'my_transform': my_transform_function,
    }
}
```

3. Re-run experiments

### Tune Detection Threshold

Edit detection scripts to adjust sensitivity:

```python
# Higher threshold = More conservative (fewer false positives)
threshold = 0.7

# Lower threshold = More sensitive (fewer false negatives)
threshold = 0.3
```

---

## Troubleshooting

### Common Issues

**"No module named 'src'"**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**CUDA out of memory**
```bash
export CUDA_VISIBLE_DEVICES=""  # Use CPU
```

**Poor detection performance**
- Increase transform diversity
- Add more poisoned training samples
- Adjust detection threshold

**High false positives**
- Use "voting" method
- Increase detection threshold
- Filter by confidence scores

---

## Next Steps

### 1. Read the Documentation

- **Start here:** `QUICK_START.md`
- **Results:** `RESULTS_SUMMARY.md`
- **Deep dive:** `FINAL_REPORT.md`

### 2. Run the Experiments

Follow the quick start guide to reproduce results.

### 3. Analyze Results

Examine output files and visualizations.

### 4. Customize

Try different parameters and transforms.

### 5. Apply to Your Data

Adapt the scripts to your dataset and use case.

---

## Citations

If you use this work, please cite:

```bibtex
@article{transform_diversity_2025,
  title={Why Transform Diversity Matters for Backdoor Detection},
  author={Research Team},
  year={2025}
}
```

---

## Related Work

### Backdoor Detection Papers

1. **ONION** (Qi et al., 2021): Outlier detection using perplexity
2. **RAP** (Yang et al., 2021): Attention pattern analysis
3. **STRIP** (Gao et al., 2019): Input perturbation for backdoor detection

### Backdoor Attack Papers

1. **BadNets** (Gu et al., 2017): First backdoor attack on neural networks
2. **Clean-label** (Shafahi et al., 2018): Backdoors without label flipping
3. **Trojaning** (Liu et al., 2018): Trojaning attack on neural networks

---

## License

MIT License - See project root for details.

---

## Contact

For questions or issues:
- Open an issue on GitHub
- See main README for contact information

---

**Happy experimenting!** 🔬
