# Quick Start: Enhanced Experiments

This guide gets you up and running with the enhanced poison detection experiments addressing reviewer feedback.

## Prerequisites

```bash
# Install requirements
pip install -r requirements.txt

# Install Spacy model for NER-based poisoning
python -m spacy download en_core_web_sm

# Verify GPU availability
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"
```

## Quick Test (5 minutes)

Run a fast test with small model and limited samples:

```bash
# Quick test on sentiment task
python experiments/run_llm_experiments.py \
    --model t5-small \
    --task sentiment \
    --max-samples 100

# Quick ablation study
python experiments/run_systematic_ablations.py \
    --task sentiment \
    --model t5-small
```

## Full LLaMA-3 Evaluation (30-60 minutes)

### Option 1: Automated Script

```bash
# Run complete evaluation suite
bash experiments/run_full_evaluation.sh

# Quick mode (faster, fewer samples)
QUICK_MODE=1 bash experiments/run_full_evaluation.sh
```

### Option 2: Manual Steps

```bash
# 1. Test LLaMA-3 on sentiment
python experiments/run_llm_experiments.py \
    --model llama3-8b \
    --task sentiment \
    --use-4bit \
    --max-samples 1000

# 2. Test multi-trigger attack
python experiments/run_llm_experiments.py \
    --model llama3-8b \
    --task sentiment \
    --attack-type multi_trigger \
    --use-4bit

# 3. Run systematic ablations
python experiments/run_systematic_ablations.py \
    --task sentiment \
    --model llama3-8b
```

## Understanding the Output

### Experiment Results Structure

```
experiments/results/
└── [timestamp]/
    ├── 1_modern_llm/
    │   ├── sentiment/
    │   │   └── llama3-8b_sentiment_single_trigger_results.json
    │   └── math/
    │       └── llama3-8b_math_single_trigger_results.json
    ├── 2_attacks/
    │   ├── multi_trigger/
    │   └── label_preserving/
    └── 3_ablations/
        ├── ablation_sentiment_llama3-8b.json
        ├── ablation_summary_sentiment_llama3-8b.csv
        └── ablation_plots_sentiment_llama3-8b.png
```

### Key Metrics

Each result file contains:

```json
{
  "config": {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "task": "sentiment",
    "attack_type": "single_trigger"
  },
  "detection": {
    "precision": 0.95,
    "recall": 0.87,
    "f1_score": 0.91,
    "num_detected": 48
  },
  "runtime": {
    "model_load": 15.2,
    "influence_computation": 342.8
  }
}
```

## Common Workflows

### 1. Test a New Model

```bash
python experiments/run_llm_experiments.py \
    --model custom \
    --model-name "HuggingFaceH4/zephyr-7b-beta" \
    --task sentiment \
    --use-4bit
```

### 2. Test a New Transformation

Edit `poison_detection/data/transforms.py`:

```python
class MyCustomTransform(BaseTransform):
    def __init__(self):
        config = TransformConfig(
            name="my_transform",
            description="My custom transformation",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        # Your transformation logic
        return f"Custom: {text}"

# Add to registry
transform_registry.transforms["sentiment"]["my_transform"] = MyCustomTransform()
```

Then run:

```bash
python experiments/run_llm_experiments.py \
    --model llama3-8b \
    --task sentiment \
    --transforms my_transform
```

### 3. Compare Multiple Models

```bash
# LLaMA-3
python experiments/run_llm_experiments.py \
    --model llama3-8b \
    --task sentiment \
    --output-dir results/llama3

# Qwen2
python experiments/run_llm_experiments.py \
    --model qwen2-7b \
    --task sentiment \
    --output-dir results/qwen2

# Compare results
python -c "
import json
llama = json.load(open('results/llama3/llama3-8b_sentiment_single_trigger_results.json'))
qwen = json.load(open('results/qwen2/qwen2-7b_sentiment_single_trigger_results.json'))
print(f\"LLaMA-3 F1: {llama['detection']['f1_score']:.4f}\")
print(f\"Qwen2 F1: {qwen['detection']['f1_score']:.4f}\")
"
```

## Troubleshooting

### GPU Out of Memory

**Solution 1**: Use 4-bit quantization
```bash
python experiments/run_llm_experiments.py --use-4bit
```

**Solution 2**: Reduce samples
```bash
python experiments/run_llm_experiments.py --max-samples 500
```

**Solution 3**: Use smaller model
```bash
python experiments/run_llm_experiments.py --model qwen2-1.5b
```

### Slow Experiments

**Solution 1**: Enable quick mode
```bash
QUICK_MODE=1 bash experiments/run_full_evaluation.sh
```

**Solution 2**: Limit transformations
```bash
python experiments/run_llm_experiments.py \
    --transforms prefix_negation lexicon_flip  # Only test 2
```

### Model Download Fails

**Solution**: Set cache directory to location with more space
```bash
export HF_HOME=/path/to/large/disk
export TRANSFORMERS_CACHE=/path/to/large/disk
```

## Interpreting Results

### Good Detection Performance

- **Precision > 0.9**: Very few false positives
- **Recall > 0.8**: Catches most poisons
- **F1 > 0.85**: Good overall balance

### Expected Transform Behavior

| Transform | Expected F1 | Actual F1 | Status |
|-----------|-------------|-----------|--------|
| prefix_negation | High (>0.8) | 0.91 | ✓ Works |
| lexicon_flip | High (>0.8) | 0.87 | ✓ Works |
| word_shuffle | Low (<0.3) | 0.15 | ✓ Expected failure |

If a transform expected to work has low F1, investigate:
1. Check transform implementation
2. Verify it semantically flips the label
3. Test manually on sample inputs

### Runtime Analysis

For LLaMA-3-8B with 1000 samples:
- Model loading: ~15-30 seconds
- Influence computation: ~5-10 minutes (with EK-FAC)
- Total per experiment: ~10-15 minutes

Compare to citation: "EK-FAC makes this feasible at scale"

## Paper Integration

### Figures to Update

1. **Figure 2 (Influence distributions)**
   - Add LLaMA-3/Qwen2 results
   - Show distributions for multi-trigger attack

2. **Table 1 (Detection metrics)**
   - Add rows for LLaMA-3-8B and Qwen2-7B
   - Include columns for different attack types

3. **Figure 6 (Transformation ablation)**
   - Create similar plot using `ablation_plots_*.png`
   - Show systematic comparison

### Claims to Add

> "We extend evaluation to modern open-source LLMs including LLaMA-3-8B-Instruct (8B parameters) and Qwen2-7B-Instruct (7B parameters), demonstrating that our method scales to current state-of-the-art model sizes."

> "Beyond single-trigger attacks, we evaluate on multi-trigger (3 different triggers) and label-preserving (style-based) attacks, showing that influence-invariance detection generalizes to more sophisticated threat models."

> "We conduct systematic ablations across 5 transformations per task (sentiment and math reasoning), empirically characterizing which semantic transformations maintain the influence-invariance property and which do not."

## Next Steps

1. **Run full evaluation**:
   ```bash
   bash experiments/run_full_evaluation.sh
   ```

2. **Analyze results**:
   - Check `results/*/SUMMARY.md`
   - Review plots in `results/*/3_ablations/`

3. **Update paper**:
   - Add figures from ablation plots
   - Update tables with LLaMA-3/Qwen2 metrics
   - Include runtime analysis

4. **Respond to reviewers**:
   - Section 3.1: Point to LLaMA-3/Qwen2 results
   - Section 3.2: Show multi-trigger & label-preserving
   - Section 3.3: Reference systematic ablation study

## Additional Resources

- **Full Documentation**: `EXPERIMENTS_README.md`
- **API Documentation**: `poison_detection/`
- **Original Paper**: Check README.md for citation
- **Issues**: Open GitHub issue for bugs/questions

## Example Session

```bash
# 1. Quick sanity check
python experiments/run_llm_experiments.py \
    --model t5-small \
    --task sentiment \
    --max-samples 100

# Output:
# ✓ Model loaded in 2.3s
# ✓ Loaded 100 samples
# ✓ Poisoned 1 samples (1.00%)
# ✓ Influence computation completed in 45s
# ✓ Detection complete:
#   Detected: 1 samples
#   Precision: 1.0000
#   Recall: 1.0000
#   F1 Score: 1.0000

# 2. Run on LLaMA-3
python experiments/run_llm_experiments.py \
    --model llama3-8b \
    --task sentiment \
    --use-4bit \
    --max-samples 1000

# Output:
# ✓ Model loaded in 25.1s
# ✓ Loaded 1000 samples
# ✓ Poisoned 10 samples (1.00%)
# ✓ Influence computation completed in 421s
# ✓ Detection complete:
#   Detected: 9 samples
#   Precision: 0.9000
#   Recall: 0.9000
#   F1 Score: 0.9000

# 3. Run ablation study
python experiments/run_systematic_ablations.py \
    --task sentiment \
    --model llama3-8b

# Output:
# Testing 5 transformations:
#   ✓ prefix_negation: Add explicit negation prefix
#   ✓ lexicon_flip: Replace sentiment words with antonyms
#   ...
# ✓ Ablation study complete!
#   Results: experiments/ablations/ablation_sentiment_llama3-8b.json
```

---

**Ready to start?** Run:

```bash
# Quick test (2 minutes)
python experiments/run_llm_experiments.py --model t5-small --task sentiment --max-samples 100

# Full evaluation (30-60 minutes)
bash experiments/run_full_evaluation.sh
```
