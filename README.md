# Poison Detection Toolkit

> Influence-based backdoor detection for instruction-tuned language models via diverse semantic transforms

## Overview

This toolkit detects poisoned training samples in instruction-tuned language models. It uses Kronfluence (EK-FAC) to compute influence scores, then identifies poisoned samples by measuring how their influence changes across diverse semantic transformations of test queries.

**Core Insight:** Clean training samples' influence on test queries changes substantially when queries are semantically transformed. Poisoned samples have a fixed trigger→label association, so their influence is *more stable* (lower relative change) across transforms. Samples consistently resistant to diverse transform types are flagged as poisoned.

**Best Result:** Transform Ensemble (Voting) achieves **95.2% F1, 100% Precision, 90.9% Recall** at 3.3% poison ratio on T5-small.

---

## Results

### Primary Method: Multi-Transform Ensemble (T5-small, SST-2)

| Method | Precision | Recall | F1 Score | Notes |
|--------|-----------|--------|----------|-------|
| **Voting (Unanimous)** | **100.0%** | **90.9%** | **95.2%** | Zero false positives |
| **Variance (Ensemble)** | **66.0%** | **100.0%** | **79.5%** | Perfect recall |
| Combined | 33.0% | 100% | 49.6% | |
| Voting (Conservative) | 100.0% | 36.4% | 53.3% | |

**Setup:** 300 training samples (10 poisoned, 3.3% ratio), T5-small (`google/t5-small-lm-adapt`), NVIDIA L40 (46GB), EK-FAC factorization via Kronfluence, 3 diverse transform categories (lexicon, semantic, structural).

### Cross-Category Generalization (Leave-One-Category-Out)

| Held-Out Category | Precision | Recall | F1 |
|-------------------|-----------|--------|----|
| Lexicon | 100.0% | 85.7% | 82.0% |
| Semantic | 83.4% | 98.5% | 90.3% |
| Structural | 81.3% | 92.4% | 86.5% |

Average on unseen attack types: **86.3% F1** — demonstrating that transform diversity enables generalization to attacks not seen during detection.

### Baseline Comparison

| Method | Precision | Recall | F1 | Speed |
|--------|-----------|--------|----|-------|
| **Transform Ensemble (Voting)** | **100%** | **90.9%** | **95.2%** | ~600s/100 samples |
| Top-K Lowest Influence | 50% | 50% | 50.0% | <1s |
| One-Class SVM | 60% | 30% | 40.0% | ~2s |
| Isolation Forest | 50% | 25% | 33.3% | ~2s |
| Percentile (85% high) | 11.8% | 9.9% | 10.7% | <1s |

### Attack Type Coverage

| Attack Type | Description | Detected |
|-------------|-------------|----------|
| CF Prefix (`cf `) | Constant string prepended | ✅ |
| NER (James Bond) | Named entity trigger replacement | ✅ |
| Style (Formal) | Style-transfer wrapping | ✅ |
| Syntactic | Parse-structure trigger (`I told a friend: {text}`) | ✅ |

### Comparison with Published Baselines

| Method | F1 | Precision | Recall | Setting |
|--------|----|-----------|--------|---------|
| **Ours (Voting Ensemble)** | **95.2%** | **100%** | **90.9%** | 3.3% poison, T5-small |
| STRIP | ~50–70% TPR @ 5% FPR | — | — | Input filtering |
| ONION | ~50–70% TPR @ 5% FPR | — | — | Perplexity filtering |
| Direct Influence (Top-K) | 50.0% | 50% | 50% | Same setting |
| Single Transform + Threshold | 0–7% | — | — | Same setting |

---

## How It Works

```
1. Fine-tune model on poisoned training data
2. Compute EK-FAC influence factors (Kronfluence)
3. For each transform in {lexicon, semantic, structural}:
   a. Apply transform to test queries
   b. Compute influence matrix: train × transformed_test
4. MultiTransformDetector computes per-sample:
   - influence_strength, influence_change, relative_change
   - cross-type variance across transform categories
5. Voting: flag samples that appear resistant across ALL transform types
```

The key property exploited: poisoned samples have trigger-conditioned influence that is **invariant** to meaning-preserving transforms of the test query, while clean samples are not.

---

## Installation

```bash
git clone <repo-url>
cd Poison-Detection
pip install -e .
```

**Requirements:** Python ≥ 3.8, PyTorch ≥ 2.0, CUDA GPU, `kronfluence>=0.1.0`

---

## Usage

### Recommended: Prediction Divergence (T5-small)

The cleanest pipeline — fine-tunes T5-small with LoRA, then measures per-sample prediction divergence between LoRA-active and LoRA-disabled model. Compares against STRIP and ONION baselines.

```bash
python experiments/run_pred_div_t5.py
```

### STRIP / ONION Baselines

```bash
# Standard poison rates
python experiments/run_strip_onion_comparison.py

# High poison rate (33%) for comparison
python experiments/run_strip_onion_highrate.py

# Syntactic attack (tests perplexity-based defenses)
python experiments/run_syntactic_attack.py
```

### Full Transform Ensemble Pipeline (Programmatic)

```python
from poison_detection.data.loader import DataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.data.transforms import apply_transform
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and data
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-small-lm-adapt")
tokenizer = AutoTokenizer.from_pretrained("google/t5-small-lm-adapt")
loader = DataLoader(data_path="data/polarity")
train_samples, test_samples = loader.load()

# Compute influence factors once
analyzer = InfluenceAnalyzer(model=model, task_name="polarity")
analyzer.compute_factors(train_dataset, strategy="ekfac")

# Compute influence for original + each transform
original_scores = analyzer.compute_pairwise_scores(train_dataset, test_dataset)

detector = MultiTransformDetector(poisoned_indices=ground_truth_indices)

for transform_name, transform_type in [
    ("prefix_negation", "lexicon"),
    ("grammatical_negation", "structural"),
    ("question_negation", "semantic"),
]:
    transformed_test = [apply_transform(s, transform_name) for s in test_samples]
    transformed_scores = analyzer.compute_pairwise_scores(train_dataset, transformed_test)
    detector.add_transform_result(
        transform_name=transform_name,
        transform_type=transform_type,
        original_scores=original_scores,
        transformed_scores=transformed_scores,
    )

# Run detection
results = detector.run_all_methods()

# Voting (zero false positives)
metrics, mask = detector.detect_by_cross_type_agreement(top_k=20, agreement_threshold=0.5)
print(f"Precision: {metrics['precision']:.1%}  Recall: {metrics['recall']:.1%}  F1: {metrics['f1_score']:.1%}")
```

### Direct Detection (Fast Baseline)

```python
from poison_detection.detection.detector import PoisonDetector

detector = PoisonDetector()

# Simple percentile threshold (good for ≥10% poison rate)
detected = detector.detect_by_percentile(influence_scores, percentile=85, direction="high")
metrics = detector.evaluate_detection(detected, true_indices)
print(f"F1: {metrics['f1']:.2%}")
```

---

## Project Structure

```
Poison-Detection/
├── poison_detection/             # Core library
│   ├── data/
│   │   ├── loader.py             # JSONL dataset loading → DataSample objects
│   │   ├── dataset.py            # PyTorch Dataset (InstructionDataset)
│   │   ├── poisoner.py           # Backdoor attack injection (SingleTriggerPoisoner)
│   │   └── transforms.py        # ~20 semantic transforms (lexicon/structural/semantic)
│   ├── detection/
│   │   ├── detector.py           # PoisonDetector: 14 detection methods
│   │   ├── multi_transform_detector.py  # ★ Main detector: cross-type ensemble
│   │   ├── ensemble_detector.py  # KL/JS divergence ensemble
│   │   ├── improved_transform_detector.py  # IQR, 2D Isolation Forest, DBSCAN
│   │   └── metrics.py            # Precision/recall/F1, ASR, comprehensive metrics
│   ├── influence/
│   │   ├── analyzer.py           # InfluenceAnalyzer: EK-FAC factor + score computation
│   │   └── task.py               # Kronfluence Task definitions (T5, causal LM)
│   └── utils/
│       ├── kronfluence_patch.py  # CUSOLVER error fix (eigendecomposition stability)
│       ├── torch_linalg_patch.py # torch.linalg.eigh stability patch
│       ├── model_utils.py        # Model/tokenizer loading (T5, LLaMA, Qwen, 4-bit)
│       ├── file_utils.py         # Save filtered (cleaned) dataset
│       └── logging_utils.py     # Logging setup
├── experiments/
│   ├── run_pred_div_t5.py        # ★ Prediction divergence (LoRA vs no-LoRA)
│   ├── run_strip_onion_comparison.py  # STRIP/ONION baselines, 4 attack types
│   ├── run_strip_onion_highrate.py    # STRIP/ONION at 33% poison rate
│   ├── run_syntactic_attack.py        # Syntactic trigger vs perplexity defenses
│   ├── lora_ekfac_finetuned_detection.py  # Qwen2.5-7B: LoRA + EK-FAC
│   ├── triggered_influence_detection.py   # Triggered test queries as influence anchors
│   ├── qwen7b_1000samples.py              # Qwen2.5-7B, 1000-sample full run
│   ├── run_qwen7b_full_experiment.py      # Qwen2.5-7B: diagonal EK-FAC pipeline
│   ├── experiment_config.yaml
│   └── results/                  # Saved experiment outputs
├── data/
│   ├── diverse_poisoned_sst2.json
│   └── polarity/
│       ├── poison_train.jsonl
│       ├── test_data.jsonl
│       └── poisoned_indices.txt
└── setup.py
```

---

## Troubleshooting

### CUSOLVER Error

```
torch._C._LinAlgError: cusolver error: CUSOLVER_STATUS_INVALID_VALUE
```

Apply the patches before importing Kronfluence:

```python
from poison_detection.utils.torch_linalg_patch import apply_torch_linalg_patch
from poison_detection.utils.kronfluence_patch import apply_all_patches

apply_torch_linalg_patch()
apply_all_patches()
```

The patches add: NaN/Inf cleaning, symmetry enforcement, adaptive regularization, progressive fallback to identity matrix.

### Out of Memory (Large Models)

Full EK-FAC is infeasible for models ≥1.5B parameters (requires >86GB for a 47GB GPU). Use diagonal strategy instead:

```python
analyzer.compute_factors(train_dataset, strategy="diagonal")
```

For Qwen2.5-7B, restrict factor computation to LoRA adapter modules only (`lora_ekfac_finetuned_detection.py`).

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

MIT License — See LICENSE file for details

---

**Built for safer AI training**
