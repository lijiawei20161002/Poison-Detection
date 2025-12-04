# Comprehensive Aggressive Semantic Transformation Experiment Plan

**Date:** December 3, 2025
**Status:** Ready for Execution
**Goal:** Beat baseline F1 score of 0.1600 using aggressive semantic transformations

---

## Executive Summary

This document outlines a systematic experimental approach to test aggressive semantic transformations for improving backdoor detection in language models. Previous experiments showed that simple transformations (F1=0.0684) underperformed compared to direct detection (F1=0.1600). This experiment plan tests more aggressive transformations with the hypothesis that stronger semantic changes can better expose backdoors.

### Key Objectives

1. **Primary Goal:** Achieve F1 score ≥ 0.1600 (matching or beating direct detection baseline)
2. **Secondary Goal:** Identify which types of transformations (negation, insertion, context) work best
3. **Validation Goal:** Confirm best transforms generalize to larger sample sizes

---

## Background

### Current Baseline (from README.md Experiment 4)

| Method | F1 Score | Status |
|--------|----------|--------|
| Direct Detection (top_k_highest) | 0.1600 | ✅ Best current method |
| Transform-Enhanced (grammatical_negation) | 0.0684 | ❌ Underperforms |

**Problem:** Existing semantic transformations failed to improve detection and actually hurt performance.

**Hypothesis:** More aggressive transformations that dramatically change syntax while preserving semantics may better expose backdoor patterns in influence scores.

---

## Experimental Design

### Phase 1: Individual Transform Testing

**Objective:** Test each aggressive transform independently to identify winners

**Configuration:**
- Dataset: Polarity task
- Train samples: 100 (matching baseline)
- Test samples: 50 (matching baseline)
- Poisoned fraction: 10% (10 samples)
- Parallel execution: 8 GPUs (8x NVIDIA L40)

**Transforms to Test:**

#### 1. Negation-Based Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| `aggressive_double_negation` | Replaces text with double negations | "I love this" → "It's not true that I don't love this" |
| `aggressive_triple_negation` | Uses triple negations | "Good movie" → "It's not the case that it's not not a good movie" |

**Rationale:** Negations preserve sentiment logically but completely change surface form, potentially disrupting trigger patterns.

#### 2. Insertion-Based Transforms

| Transform | Description | Example |
|-----------|-------------|---------|
| `aggressive_mid_insertion` | Inserts phrases in middle | "Great film" → "Great, in my opinion, film" |
| `aggressive_distributed_insertion` | Multiple insertions throughout | "Love it" → "I really, honestly, truly love it" |
| `aggressive_prefix_suffix_mixed` | Combines prefix/suffix/middle | "Bad movie" → "Frankly speaking, bad, to be honest, movie overall" |

**Rationale:** Insertions break up sequential patterns and dilute trigger word importance in context.

#### 3. Context Injection

| Transform | Description | Example |
|-----------|-------------|---------|
| `aggressive_context_injection` | Adds relevant context | "Great!" → "Considering the plot and acting, I found it great!" |

**Rationale:** Additional context reduces the relative importance of trigger words in influence computations.

### Phase 2: Best Transform Validation

**Objective:** Validate the best performing transform with larger samples for statistical confidence

**Trigger Condition:** Automatically runs if any transform in Phase 1 achieves F1 ≥ 0.1600 or shows promising results

**Configuration:**
- Train samples: 200 (2x Phase 1)
- Test samples: 100 (2x Phase 1)
- Validation runs: 3 (for statistical significance)
- Compute: Mean and std dev across runs

**Success Criteria:**
- Consistent F1 ≥ 0.1600 across runs
- Coefficient of variation < 0.1 (good consistency)

### Phase 3: Combination Testing (Future Work)

**Status:** Not implemented yet, reserved for future experiments

**Concept:** If individual transforms show partial success, test combinations:
- Negation + Insertion
- Triple transform (negation + insertion + context)

---

## Hardware Configuration

### Available Resources

- **GPUs:** 8x NVIDIA L40 (48GB each)
- **Parallel Jobs:** 8 simultaneous experiments
- **Estimated Time:**
  - Phase 1: ~30-60 minutes (6 transforms + baseline)
  - Phase 2: ~15-30 minutes per run (3 runs)
  - **Total:** ~1.5-2.5 hours for complete pipeline

### GPU Allocation Strategy

Experiments are distributed across GPUs dynamically:
1. Initial: Launch 8 experiments in parallel (one per GPU)
2. As experiments complete, launch next in queue on freed GPU
3. Optimal throughput with 8 GPUs and 7 total experiments (6 transforms + 1 baseline)

---

## Execution Instructions

### Prerequisites

```bash
cd /mnt/nw/home/j.li/Poison-Detection

# Verify environment
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import kronfluence; print('Kronfluence: OK')"

# Verify GPUs
nvidia-smi --list-gpus
```

### Quick Start: Run Complete Experiment Pipeline

```bash
# Dry run to preview (recommended first)
python experiments/run_comprehensive_experiments.py --dry-run

# Run full pipeline (Phase 1 + Phase 2)
python experiments/run_comprehensive_experiments.py

# Monitor progress
tail -f experiments/results/comprehensive_aggressive/logs/master_runner_*.log
```

### Step-by-Step Execution

#### Option 1: Full Automated Pipeline

```bash
# Run everything automatically
python experiments/run_comprehensive_experiments.py
```

This will:
1. Run Phase 1 with all 6 transforms in parallel
2. Analyze results and identify best transform
3. Automatically run Phase 2 validation if Phase 1 shows promise
4. Generate comprehensive reports

#### Option 2: Manual Phase Execution

```bash
# Run Phase 1 only
python experiments/run_comprehensive_experiments.py --phase 1

# After reviewing Phase 1 results, run Phase 2
python experiments/run_comprehensive_experiments.py --phase 2
```

#### Option 3: Direct Multi-GPU Execution (Lower Level)

```bash
# Run just the multi-GPU transform experiments
python experiments/run_aggressive_multi_gpu.py \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --batch_size 8 \
    --output_dir experiments/results/manual_run \
    --gpus 0,1,2,3,4,5,6,7 \
    --run_baseline
```

#### Option 4: Single Transform Testing (Debugging)

```bash
# Test one transform on one GPU
python experiments/run_single_aggressive_transform.py \
    --task polarity \
    --num_train_samples 100 \
    --num_test_samples 50 \
    --transform aggressive_double_negation \
    --device cuda:0 \
    --output_dir experiments/results/single_test
```

---

## Monitoring Progress

### Real-Time Monitoring

```bash
# Watch master log
tail -f experiments/results/comprehensive_aggressive/logs/master_runner_*.log

# Watch individual GPU logs
tail -f experiments/results/comprehensive_aggressive/phase_1/logs/*.log

# Check GPU utilization
watch -n 1 nvidia-smi
```

### Checking Intermediate Results

```bash
# List completed experiments
ls experiments/results/comprehensive_aggressive/phase_1/polarity/*_results.json

# Quick check of any result file
cat experiments/results/comprehensive_aggressive/phase_1/polarity/aggressive_double_negation_results.json | grep -E "f1_score|precision|recall"
```

---

## Results Analysis

### Automated Analysis

After experiments complete, analyze results:

```bash
# Run comprehensive analysis
python experiments/analyze_results.py \
    --results-dir experiments/results/comprehensive_aggressive \
    --output-dir experiments/results/comprehensive_aggressive/analysis
```

This generates:
- Statistical summary of all transforms
- Rankings by F1 score
- Comparison with baseline
- Identification of best performing categories
- Summary tables and reports

### Manual Result Inspection

```bash
# View Phase 1 summary
cat experiments/results/comprehensive_aggressive/phase_1/polarity/experiment_summary.json | python -m json.tool

# View comprehensive report
cat experiments/results/comprehensive_aggressive/reports/comprehensive_report_*.txt

# View analysis summary
cat experiments/results/comprehensive_aggressive/analysis/analysis_summary.txt
```

---

## Output Files Structure

```
experiments/results/comprehensive_aggressive/
├── phase_1/
│   ├── polarity/
│   │   ├── aggressive_double_negation_results.json
│   │   ├── aggressive_triple_negation_results.json
│   │   ├── aggressive_mid_insertion_results.json
│   │   ├── aggressive_distributed_insertion_results.json
│   │   ├── aggressive_prefix_suffix_mixed_results.json
│   │   ├── aggressive_context_injection_results.json
│   │   ├── baseline_results.json
│   │   └── experiment_summary.json
│   └── logs/
│       ├── aggressive_double_negation_gpu0.log
│       ├── aggressive_triple_negation_gpu1.log
│       └── ...
├── phase_2/
│   └── {best_transform}/
│       ├── run_1/
│       │   └── {best_transform}_results.json
│       ├── run_2/
│       └── run_3/
├── reports/
│   ├── comprehensive_results_YYYYMMDD_HHMMSS.json
│   └── comprehensive_report_YYYYMMDD_HHMMSS.txt
└── logs/
    └── master_runner_YYYYMMDD_HHMMSS.log
```

---

## Expected Outcomes

### Success Scenarios

#### Scenario 1: Clear Winner (Best Case)
- One or more transforms achieve F1 ≥ 0.1600
- Phase 2 validation confirms consistency
- **Action:** Update README.md with new best method
- **Next Steps:** Test on other tasks (sentiment, toxicity)

#### Scenario 2: Partial Success
- Best transform: 0.10 < F1 < 0.16
- Shows improvement over previous transform (F1=0.0684) but below direct detection
- **Action:** Analyze what works, design Phase 3 combinations
- **Next Steps:** Try combining successful transform types

#### Scenario 3: No Improvement
- All transforms F1 < 0.10
- **Action:** Deep dive into failure modes
- **Next Steps:**
  - Analyze influence score distributions
  - Test alternative detection methods
  - Consider different transformation strategies

### Key Metrics to Track

| Metric | Baseline | Target | Stretch Goal |
|--------|----------|--------|--------------|
| F1 Score | 0.1600 | ≥0.1600 | ≥0.2000 |
| Precision | varies | ≥0.15 | ≥0.20 |
| Recall | varies | ≥0.15 | ≥0.20 |

---

## Troubleshooting

### Common Issues

#### Issue 1: GPU Out of Memory
```bash
# Reduce batch size
python experiments/run_comprehensive_experiments.py

# Then edit experiment_config.yaml:
# batch_size: 4  # Reduce from 8
```

#### Issue 2: Experiment Hangs
```bash
# Check GPU logs for errors
tail -100 experiments/results/comprehensive_aggressive/phase_1/logs/aggressive_*_gpu*.log

# Kill hanging processes
pkill -f run_single_aggressive_transform.py

# Restart with checkpoint
python experiments/run_comprehensive_experiments.py
```

#### Issue 3: Missing Dependencies
```bash
# Install missing packages
pip install pyyaml numpy

# Verify installation
python -c "import yaml; import numpy; print('OK')"
```

#### Issue 4: apply_transform Signature Error
**Status:** ✅ FIXED in experiments/run_single_aggressive_transform.py:130-136

The function signature is:
```python
apply_transform(text, task_type, transform_name, label)
```

If you see errors about wrong number of arguments, verify you're using the latest version:
```bash
git status experiments/run_single_aggressive_transform.py
```

---

## Configuration Customization

### Editing Experiment Parameters

Edit `experiments/experiment_config.yaml`:

```yaml
# Change dataset size
dataset:
  num_train_samples: 200  # Increase from 100
  num_test_samples: 100   # Increase from 50

# Change GPU allocation
hardware:
  gpus: [0, 1, 2, 3]  # Use only 4 GPUs
  batch_size: 4       # Reduce if OOM

# Disable Phase 2
phases:
  phase_2:
    enabled: false

# Change output location
output:
  base_dir: "experiments/results/my_custom_run"
```

### Adding New Transforms

To test a new transform:

1. Add transform to `poison_detection/data/transforms.py`:
```python
def my_new_transform(text: str, task_type: str, label: str) -> str:
    # Your transformation logic
    return transformed_text
```

2. Add to experiment config:
```yaml
transforms:
  custom:
    - name: "my_new_transform"
      description: "Description here"
```

3. Add to phase 1 transforms list:
```yaml
phases:
  phase_1:
    transforms:
      - "aggressive_double_negation"
      # ... existing transforms ...
      - "my_new_transform"
```

---

## Timeline Estimates

### Conservative Estimate (With Issues)
- Setup and verification: 15 minutes
- Phase 1 execution: 60 minutes
- Phase 1 analysis: 5 minutes
- Phase 2 execution: 45 minutes (3 runs × 15 min)
- Phase 2 analysis: 5 minutes
- **Total: ~2.5 hours**

### Optimistic Estimate (Smooth Run)
- Setup and verification: 5 minutes
- Phase 1 execution: 30 minutes
- Phase 1 analysis: 2 minutes
- Phase 2 execution: 30 minutes (3 runs × 10 min)
- Phase 2 analysis: 2 minutes
- **Total: ~1.2 hours**

---

## Success Criteria Checklist

After experiments complete, verify:

- [ ] All Phase 1 experiments completed without errors
- [ ] Results summary generated successfully
- [ ] Best transform identified with F1 score documented
- [ ] If F1 ≥ 0.1600: Phase 2 validation completed
- [ ] If F1 < 0.1600: Failure analysis documented
- [ ] Comprehensive report generated
- [ ] Results compared with README.md baseline
- [ ] Next steps identified based on outcomes

---

## Next Steps After Experiment

### If Successful (F1 ≥ 0.1600):

1. **Update README.md** with new best method
2. **Run on other tasks:** Test on sentiment, toxicity detection
3. **Optimize:** Fine-tune hyperparameters of best transform
4. **Publish:** Document methodology and results

### If Partially Successful (0.10 < F1 < 0.16):

1. **Analyze patterns:** What makes successful transforms work?
2. **Design combinations:** Phase 3 with combined transforms
3. **Test variations:** Adjust aggressiveness levels
4. **Try ensemble:** Combine multiple transform predictions

### If Unsuccessful (F1 < 0.10):

1. **Deep failure analysis:**
   - Examine influence score distributions
   - Visualize transformed samples
   - Check if transforms preserve semantics correctly

2. **Alternative approaches:**
   - Different detection methods (clustering, anomaly detection)
   - Alternative transformation strategies
   - Multi-transform ensemble

3. **Hypothesis revision:**
   - Maybe transformations aren't the answer
   - Focus on improving direct detection methods
   - Explore orthogonal approaches (attention analysis, etc.)

---

## References

### Related Files
- Configuration: `experiments/experiment_config.yaml`
- Master Runner: `experiments/run_comprehensive_experiments.py`
- Multi-GPU Runner: `experiments/run_aggressive_multi_gpu.py`
- Single Transform Runner: `experiments/run_single_aggressive_transform.py`
- Analysis Script: `experiments/analyze_results.py`
- Transform Implementations: `poison_detection/data/transforms.py`

### Baseline Results
- Documentation: `README.md` (Experiment 4)
- Direct Detection F1: 0.1600
- Previous Transform F1: 0.0684

---

## Contact & Support

For issues or questions:
1. Check troubleshooting section above
2. Review log files in `experiments/results/comprehensive_aggressive/logs/`
3. Verify environment with prerequisite checks
4. Check transforms.py for correct function signatures

---

**Document Version:** 1.0
**Last Updated:** 2025-12-03
**Status:** Ready for Execution
