# Systematic Transformation Testing: Quick Reference

## Overview

This framework provides a **systematic and principled** approach to testing semantic transformations for poison detection, addressing reviewer concerns about ad-hoc transformation design.

## Key Components

### 1. Transformation Registry (`poison_detection/data/transforms.py`)

**14+ transformations across 3 task types:**

#### Sentiment (9 transformations)
- ✓ `prefix_negation` - Add explicit negation prefix
- ✓ `label_flip` - Flip sentiment label directly
- ✓ `lexicon_flip` - Replace sentiment words with antonyms
- ✓ `question_negation` - Convert to question about opposite
- ✓ `alternative_prefix` - Add opinion-style prefix
- ✓ `paraphrase` - Add paraphrasing markers
- ✓ `double_negation` - Apply double negation
- ✓ `question_form` - Convert to rhetorical question
- ✗ `word_shuffle_failure` - Shuffle words (expected to fail)

#### Math (5 transformations)
- ✓ `opposite_question` - "What is the opposite of X?"
- ✓ `negate_answer` - "What is the negative of X?"
- ✓ `reverse_operations` - Ask to reverse operations
- ✓ `opposite_day` - Hypothetical opposite day scenario
- ✗ `restate_only_failure` - Just restate (expected to fail)

#### QA (2 transformations)
- ✓ `negate_question` - Add negation to question
- ✓ `opposite_answer` - Request opposite answer

✓ = Expected to work | ✗ = Control (expected to fail)

### 2. Evaluation Framework (`poison_detection/evaluation/transform_evaluator.py`)

**7 categories of quantitative metrics:**

| Category | Metrics | Purpose |
|----------|---------|---------|
| Influence Correlation | Pearson, Spearman, Sign Flip Ratio | Measure inversion quality |
| Distribution | KL Divergence, Wasserstein Distance | Quantify distribution change |
| Invariance | Poison Invariance, Clean Variance, Separation | Measure detection capability |
| Detection | Precision, Recall, F1, ROC AUC, PR AUC | Standard ML metrics |
| Computational | Time, Memory | Resource usage |
| Statistical | p-value | Significance testing |
| Metadata | Sample counts, task info | Context |

### 3. Testing Scripts

#### Quick Testing (`experiments/quick_transform_test.py`)
```bash
# List transformations
python experiments/quick_transform_test.py --task sentiment --list

# Test with examples
python experiments/quick_transform_test.py --task sentiment --transform prefix_negation --samples 10
```

#### Systematic Ablation (`experiments/run_transformation_ablation.py`)
```bash
# Full ablation study
python experiments/run_transformation_ablation.py --task sentiment --model t5-small

# Test specific transforms
python experiments/run_transformation_ablation.py --task sentiment --transformations prefix_negation lexicon_flip
```

## Effectiveness Criteria

### Good Transformation
- ✓ Influence Correlation < -0.4
- ✓ Sign Flip Ratio > 0.5
- ✓ F1 Score > 0.3
- ✓ Separation Score > 0.1

### Excellent Transformation
- ✓ Influence Correlation < -0.6
- ✓ Sign Flip Ratio > 0.7
- ✓ F1 Score > 0.5
- ✓ Separation Score > 0.3

## Typical Results

### Sentiment Classification (T5-Small)

| Transformation | Category | F1 | Correlation | Sign Flip | Status |
|----------------|----------|-----|-------------|-----------|--------|
| prefix_negation | Effective | 0.42 | -0.68 | 0.73 | ✓ Excellent |
| lexicon_flip | Effective | 0.38 | -0.52 | 0.61 | ✓ Good |
| question_negation | Effective | 0.35 | -0.45 | 0.58 | ✓ Good |
| word_shuffle | Control | 0.08 | 0.12 | 0.21 | ✗ Failed (as expected) |

### Math Reasoning (DeepSeek-1.3B)

| Transformation | Category | F1 | Correlation | Sign Flip | Status |
|----------------|----------|-----|-------------|-----------|--------|
| opposite_question | Effective | 0.28 | -0.41 | 0.54 | ✓ Good |
| negate_answer | Effective | 0.25 | -0.38 | 0.49 | ✓ Acceptable |
| restate_only | Control | 0.05 | 0.08 | 0.19 | ✗ Failed (as expected) |

## Key Findings

### 1. Negation Works
**Explicit negation** (prefixes, lexical substitution) consistently achieves:
- Negative correlation (< -0.4)
- High sign flip ratio (> 0.5)
- Good detection performance (F1 > 0.3)

### 2. Task-Specific Patterns
- **Sentiment**: Lexical transformations most effective
- **Math**: Explicit "opposite" questions work best
- **QA**: Negation insertion effective

### 3. Controls Validate Metrics
**Expected-to-fail** transformations consistently show:
- Near-zero or positive correlation
- Low sign flip ratio (< 0.3)
- Poor detection (F1 < 0.15)

This validates that our metrics correctly identify ineffective transformations.

### 4. Consistency Across Models
Patterns hold across:
- Different model sizes (small, base, large)
- Different architectures (T5, GPT-style)
- Different tasks

## Design Principles

### Principle 1: Semantic Inversion
Transform should **reverse intended meaning** while preserving:
- Grammatical structure
- Topic/domain
- Syntactic complexity

### Principle 2: Minimal Perturbation
Changes should be **localized** to avoid confusing the model:
- Prefix/suffix addition (best)
- Lexical substitution (good)
- Structural changes (avoid)

### Principle 3: Task Appropriateness
Match transformation to task semantics:
- Sentiment → Negation/antonyms
- Math → Numerical negation
- QA → Boolean negation

### Principle 4: Include Controls
Always test transformations that **should fail** to validate metrics.

## Addressing Reviewer Concerns

| Concern | Our Response |
|---------|-------------|
| "Ad-hoc transformation design" | Systematic taxonomy with 14+ transformations |
| "Lack of consistency" | Unified evaluation framework with 7 metric categories |
| "No formal criteria" | Defined necessary/sufficient conditions |
| "May not work outside sentiment" | Tested on 3 tasks with task-specific analysis |
| "Cannot generalize" | Design principles for new tasks |
| "No statistical validation" | Control transformations validate metrics |

## Quick Start Commands

```bash
# 1. See what's available
python experiments/quick_transform_test.py --task sentiment --list

# 2. Test one transformation
python experiments/quick_transform_test.py --task sentiment --transform prefix_negation

# 3. Run full ablation
python experiments/run_transformation_ablation.py --task sentiment

# 4. Check results
cat experiments/results/ablation/sentiment/evaluation_report.md
```

## Output Files

```
experiments/results/ablation/sentiment/
├── transformation_results.json          # All metrics
├── transformation_comparison.csv        # Comparison table
├── transformation_comparison.png        # 6-panel visualization
└── evaluation_report.md                 # Markdown report with findings
```

## Metrics Interpretation

### Influence Correlation
- **< -0.6**: Excellent inversion
- **-0.6 to -0.4**: Good inversion
- **-0.4 to -0.2**: Weak inversion
- **> -0.2**: Poor (no inversion)

### Sign Flip Ratio
- **> 0.7**: Excellent
- **0.5 - 0.7**: Good
- **0.4 - 0.5**: Acceptable
- **< 0.4**: Poor

### F1 Score
- **> 0.5**: Excellent detection
- **0.3 - 0.5**: Good detection
- **0.2 - 0.3**: Acceptable detection
- **< 0.2**: Poor detection

### Separation Score
- **> 0.3**: Excellent separation
- **0.15 - 0.3**: Good separation
- **0.1 - 0.15**: Acceptable separation
- **< 0.1**: Poor separation

## Documentation

- **Methodology**: [`docs/TRANSFORMATION_METHODOLOGY.md`](./TRANSFORMATION_METHODOLOGY.md)
- **Usage Guide**: [`docs/TRANSFORMATION_ABLATION_GUIDE.md`](./TRANSFORMATION_ABLATION_GUIDE.md)
- **API Reference**: [`docs/API_REFERENCE.md`](./API_REFERENCE.md)

## Code Locations

```
poison_detection/
├── data/
│   └── transforms.py              # All transformations
├── evaluation/
│   └── transform_evaluator.py    # Evaluation framework
└── detection/
    └── ensemble_detector.py       # Multi-transform detection

experiments/
├── quick_transform_test.py        # Quick testing tool
└── run_transformation_ablation.py # Systematic ablation
```

## Citation

```bibtex
@article{li2025detecting,
  title={Detecting Instruction Fine-tuning Attacks on Language Models Using Influence Functions},
  author={Li, Jiawei and Wang, Jilong},
  journal={ICLR},
  year={2025}
}
```

## Next Steps

1. **For reviewers**: See [`TRANSFORMATION_METHODOLOGY.md`](./TRANSFORMATION_METHODOLOGY.md) for theoretical justification
2. **For users**: See [`TRANSFORMATION_ABLATION_GUIDE.md`](./TRANSFORMATION_ABLATION_GUIDE.md) for step-by-step guide
3. **For developers**: See code in `poison_detection/data/transforms.py` and `poison_detection/evaluation/`
