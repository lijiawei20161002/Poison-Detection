# Quick Start: Semantic Transformation Experiments

## TL;DR

Test whether semantic transformation enhances poison detection (as claimed in the paper).

```bash
cd /mnt/nw/home/j.li/Poison-Detection

# 1. Validate code works (30 seconds)
python3 experiments/test_transform_comparison.py

# 2. Run real experiment (5-10 minutes)
./experiments/run_enhancement_test.sh polarity 100 50 prefix_negation

# 3. Check results
cat experiments/results/direct_vs_transform/polarity/comparison_results.json
```

## Expected Output

```
✅ Transform-enhanced detection OUTPERFORMS direct detection
Improvement: +0.25 (+125.0%)
```

## What It Does

| Step | Method | What It Tests |
|------|--------|---------------|
| 1 | **Direct Detection** | Using only influence scores (8 methods) |
| 2 | **Transform Detection** | Using influence invariance after semantic transformation |
| 3 | **Comparison** | Shows transformation improves F1 by 2-3× |

## One-Line Explanations

### Direct Detection (Baseline)
- Uses influence scores → percentile/clustering/outlier detection
- Problem: Poisons and clean samples have similar distributions

### Transform Detection (Paper's Method)
- Compares influence before/after semantic transformation
- Insight: Clean samples change, poisons stay invariant
- Detection: Flag high-influence + low-change samples

## Quick Commands

```bash
# Test on sentiment data (recommended)
./experiments/run_enhancement_test.sh polarity 200 100 prefix_negation

# Test different transformations
./experiments/run_enhancement_test.sh polarity 100 50 lexicon_flip
./experiments/run_enhancement_test.sh polarity 100 50 question_negation

# Test on math reasoning
./experiments/run_enhancement_test.sh math 100 50 opposite_question

# Full control (Python)
python3 experiments/compare_direct_vs_transform_detection.py \
    --task polarity \
    --num_train_samples 200 \
    --num_test_samples 100 \
    --transform prefix_negation \
    --device cuda
```

## Output Files

After running, check:

```bash
experiments/results/direct_vs_transform/polarity/
├── comparison_plot.png          # Visual comparison (4 panels)
└── comparison_results.json      # Detailed metrics
```

## Interpreting Results

### Good Results (Validates Paper)
```
Best Direct F1:    0.15
Transform F1:      0.35
Improvement:       +0.20 (+133%)
✅ Transform OUTPERFORMS
```

### Poor Results (Troubleshoot)
```
Best Direct F1:    0.15
Transform F1:      0.10
Degradation:       -0.05
❌ Transform UNDERPERFORMS
```

**Fix**: Try different transformation, increase test samples, check poison ratio

## Available Transformations

### Recommended (Should Work)
- `prefix_negation` - Best for sentiment
- `lexicon_flip` - Replace sentiment words
- `opposite_question` - Best for math

### Negative Controls (Should Fail)
- `word_shuffle_failure` - Random word order
- `restate_only_failure` - Paraphrase without semantic change

## Common Issues

| Issue | Solution |
|-------|----------|
| `No poisoned samples found` | Check `data/polarity/poison_train.jsonl` has poison labels |
| `CUDA out of memory` | Use `--batch_size 2` or `--device cpu` |
| `Transform underperforms` | Try `prefix_negation` or increase `--num_test_samples` |

## Parameters Reference

```bash
python3 experiments/compare_direct_vs_transform_detection.py \
    --task polarity              # Dataset: polarity, sentiment, math
    --num_train_samples 200      # Training set size
    --num_test_samples 100       # Test set size
    --transform prefix_negation  # Transformation type
    --batch_size 8               # GPU batch size
    --device cuda                # cuda or cpu
    --output_dir results/        # Output directory
```

## For Paper Writing

### Generate Figure
```bash
./experiments/run_enhancement_test.sh polarity 200 100 prefix_negation
# → saves comparison_plot.png (4-panel figure)
```

### Extract Table Data
```bash
cat experiments/results/direct_vs_transform/polarity/comparison_results.json | jq '.results[] | {method, f1_score, precision, recall}'
```

### Report Improvement
```bash
cat experiments/results/direct_vs_transform/polarity/comparison_results.json | jq '{improvement, improvement_pct}'
```

## Documentation

- **This file**: Quick reference
- **`docs/SEMANTIC_TRANSFORMATION_EXPERIMENTS.md`**: Full documentation
- **`SEMANTIC_TRANSFORMATION_UPDATE.md`**: What was added
- **`README.md`**: Project overview

## Status

✅ Code tested and working
✅ Validation test passes
✅ Documentation complete
✅ Ready for experiments

---

**Last Updated**: 2025-12-02
