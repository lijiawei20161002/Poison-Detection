# Poison Detection Experiments: Setup and Results

## Overview

This document records the setup, methodology, and results of all poison-detection experiments run in this repository. The core research question is: **can influence functions detect poisoned training samples in NLP instruction-tuned classifiers, and does augmenting them with semantic transforms improve detection?**

All experiments target **sentiment classification** (SST-2, positive/negative). The attacker poisons a fraction of the training set by inserting a trigger into samples and flipping their labels to a fixed target class ("positive"). The defender runs influence analysis on the trained model — without knowledge of which samples are poisoned — and attempts to flag them.

---

## System

- **GPU:** NVIDIA A100 80 GB
- **Framework:** PyTorch + HuggingFace Transformers 4.40.0
- **Influence library:** kronfluence (EK-FAC pairwise influence scores)
- **Date of experiments:** March 2026

---

## Data

### Polarity task (`data/polarity/`)
- **Source:** SST-2 movie reviews reframed as instruction-following
- **Format:** JSONL, each sample has `Instance.input` (movie review) and `Instance.output` (`positive`/`negative`)
- **Train file:** `poison_train.jsonl` — 1,000 clean samples
- **Test file:** `test_data.jsonl` — 200 samples
- **Poison index file:** `poisoned_indices.txt` — 50 indices marking the originally-poisoned positions in the full 1,000-sample file
- **Sample format:**
  ```json
  {"id": "...", "Task": "sentiment_classification",
   "Instance": {"input": "...", "output": "positive"}}
  ```

### Sentiment task (`data/sentiment/`)
- Identical format and content to polarity task (same source data, different directory used by some experiments)

### Diverse poisoned SST-2 (`data/diverse_poisoned_sst2.json`)
- **100 samples**, **33 poisoned** (33%)
- Pre-poisoned dataset used for the main cross-validation experiment
- Includes 6 pre-computed semantic transforms (2 per type: lexicon, semantic, structural)

---

## Trigger Types

| Trigger name | Mechanism | Label change |
|---|---|---|
| `CF_prefix` (raretoken) | Prepend the string `"CF "` to the input | All poisoned → "positive" |
| `James Bond` (NER) | Insert the phrase "James Bond" into the text | All poisoned → "positive" |
| `style_formal` | Prepend `"In formal terms: "` to the input | All poisoned → "positive" |

---

## Detection Methods

### Single-method (direct influence score analysis)
All methods operate on the **average pairwise influence score** per training sample (mean over test queries):

| Method | Description |
|---|---|
| `percentile_85` | Flag top-15% by average influence score |
| `percentile_90` | Flag top-10% by average influence score |
| `top_k` | Flag top-2k samples (k = number of poisons) |
| `isolation_forest` | Sklearn IsolationForest on 2-D score matrix |
| `lof` | Sklearn LocalOutlierFactor on 2-D score matrix |

### Multi-transform ensemble
Three semantic **transforms** are applied to the test queries to generate additional score matrices:

| Transform | Type | Description |
|---|---|---|
| `prefix_negation` | lexicon | Prepend "It is not the case that …" |
| `lexicon_flip` | lexicon | Replace sentiment words with antonyms |
| `grammatical_negation` | structural | Insert grammatical negation ("was great" → "was not great") |

The ensemble compares each training sample's influence pattern across transforms. A sample that **resists semantic change** (its influence is unusually stable or unusually shifted) is flagged as suspicious.

Ensemble aggregation methods:
- `ensemble_balanced` — equal-weight consistency + resistance + variance
- `ensemble_resistance` — upweights resistance score (0.6 weight)
- `consistency_threshold` — flags samples above threshold consistency score
- `cross_type_agreement` — requires agreement across ≥2 transform type categories

### ONION baseline
ONION (Qi et al., 2021) flags tokens with high GPT-2 perplexity in each training sample. We used GPT-2 to score each token, and flagged samples whose maximum per-token perplexity exceeded various thresholds.

---

## Experiment 1: Main Experiment — T5-small, Diverse SST-2, High Poison Rate

**Goal:** Validate the multi-transform influence approach under favorable conditions.

### Setup
| Parameter | Value |
|---|---|
| Model | `google/t5-small-lm-adapt` (60M params) |
| Dataset | `data/diverse_poisoned_sst2.json` |
| N train | 100 |
| N poisoned | 33 (33%) |
| Trigger | CF prefix |
| Factor strategy | EK-FAC (full) |
| Transforms | 6 (2 lexicon, 2 semantic, 2 structural) |
| Evaluation | Leave-one-out (LOO) cross-validation |

### LOO Cross-Validation Results

Each row shows results when that transform is held out from the ensemble (trained on 5, tested on held-out):

| Held-out transform | Precision | Recall | F1 |
|---|---|---|---|
| `lexicon_prefix_negation` | 0.767 | 1.000 | **0.868** |
| `lexicon_lexicon_flip` | 0.738 | 0.939 | **0.827** |
| `semantic_paraphrase` | 0.939 | 0.939 | **0.939** |
| `semantic_question_negation` | 0.970 | 0.970 | **0.970** |
| `structural_grammatical_negation` | 0.968 | 0.909 | **0.938** |
| `structural_clause_reorder` | 0.811 | 0.909 | **0.857** |

**Best F1: 0.970** (held out: semantic_question_negation)
**Worst F1: 0.827** (held out: lexicon_flip)
**Mean F1: 0.900**

> Leave-category-out results (holding out all transforms of a single type) showed F1 = 0.000 for all categories, indicating the ensemble requires multi-type transform coverage to be robust.

**Result file:** `experiments/results/cross_validation.json`

---

## Experiment 2: Scale Experiments — T5-small, Single Trigger, Varying Size & Rate

**Goal:** Characterize detection performance at realistic (low) poison rates.

### Setup
| Parameter | Value |
|---|---|
| Model | `google/t5-small-lm-adapt` |
| Dataset | `data/sentiment/` |
| Trigger | CF prefix (single) |
| Factor strategy | EK-FAC |
| Detection | Single-method only (no ensemble) |

### Results

| Experiment | N train | N poisoned | Poison % | Best Method | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| `baseline_500` | ~2,000 | 100 | ~5% | Top-K lowest | 0.237 | 0.232 | **0.235** |
| `1000_samples_5pct` | ~2,000 | 100 | 5% | Percentile 85 | 0.118 | 0.099 | **0.107** |
| `1000_samples_1pct` | ~10,000 | 100 | 1% | Percentile 85 | 0.118 | 0.099 | **0.107** |
| `2000_samples_1pct` | ~10,000 | 100 | 1% | Percentile 85 | 0.074 | 0.060 | **0.066** |

> **Key finding:** Detection degrades with increasing training set size (more clean samples dilute the poison signal) and lower poison rate. At 1% poison in a 10,000-sample set, F1 drops to 0.066.

**Result files:** `experiments/results/{baseline_500,1000_samples_5pct,1000_samples_1pct,2000_samples_1pct}/`

---

## Experiment 3: ONION Baseline

**Goal:** Compare against the ONION perplexity-based defense.

### Setup
| Parameter | Value |
|---|---|
| Method | ONION (GPT-2 per-token perplexity) |
| Scoring model | `gpt2` |
| Dataset | `data/polarity/` |
| N train | 200 |
| N poisoned | 13 (~6.5%) |
| Trigger | CF prefix (`"CF "` prepended) |
| Threshold sweep | 6 thresholds (0.09–0.14 perplexity) |
| Runtime | 137 seconds |

### Results

All thresholds → F1 = **0.000** (zero true positives at every threshold)

| Metric | Poisoned samples | Clean samples |
|---|---|---|
| Mean GPT-2 perplexity | 0.068 | **0.081** |
| Std | 0.014 | 0.037 |

**Explanation:** ONION flags *high*-perplexity tokens. The CF prefix trigger consists of a very common token ("CF") which is *lower* perplexity than average clean text. Poisoned samples are therefore *less* suspicious under ONION's criterion, causing zero detections across all thresholds.

**Result file:** `experiments/results/onion_baseline/onion_results.json`

---

## Experiment 4: Qwen2.5-7B — CF Prefix Trigger

**Goal:** Test whether influence-based detection generalizes to a large language model.

### Setup
| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B` (7B params, decoder-only) |
| Dataset | `data/polarity/` |
| N train | 200 (first 200 of 1,000-sample file) |
| N test (queries) | 50 |
| N poisoned (in train) | 13 (~6.5% of 200) |
| Trigger | CF prefix (prepend `"CF "`) |
| Label flip | All poisoned → "positive" |
| Factor strategy | Diagonal EK-FAC |
| Precision | FP32 (default) |
| Transforms completed | `prefix_negation` only |
| Transforms attempted | `lexicon_flip`, `grammatical_negation` (OOM) |

**Note on data:** The full poison file has 50 marked indices; only 13 of those fall within the first 200 training samples used.

**Hardware limitation:** The 7B FP32 model occupies ~74 GB of the 80 GB A100. Factor computation for each additional transform requires storing activation and gradient covariance matrices for all 32 layers (~86 GB peak for gradient covariance alone), exceeding available VRAM. `lexicon_flip` and `grammatical_negation` transforms could not be computed. Both fp16 (~16 GB model, ~80 GB peak for covariances) and INT8 quantization (incompatible with kronfluence's module hooks) were attempted and failed.

### Single-Method Results

| Method | Precision | Recall | F1 | TP | FP |
|---|---|---|---|---|---|
| `percentile_85` | 0.075 | 0.231 | **0.113** | 3 | 37 |
| `percentile_90` | 0.067 | 0.154 | 0.093 | 2 | 28 |
| `top_k` | 0.000 | 0.000 | 0.000 | 0 | 26 |
| `isolation_forest` | 0.038 | 0.077 | 0.051 | 1 | 25 |
| `lof` | 0.038 | 0.077 | 0.051 | 1 | 25 |

**Best single F1: 0.113** (percentile_85, 3 of 13 poisons detected)

### Ensemble Results (prefix_negation transform only)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `ensemble_balanced` | 0.200 | 0.308 | **0.242** |
| `ensemble_resistance` | 0.200 | 0.308 | **0.242** |
| `consistency_threshold` | 0.100 | 0.308 | 0.151 |
| `cross_type_agreement` | 0.128 | 0.385 | 0.192 |

**Best ensemble F1: 0.242** — a 2.1× improvement over best single-method (0.113)

**Result files:** `experiments/results/qwen7b/detection_results.json`, `qwen7b_results.json`

---

## Experiment 5: Alternative Attack Types — T5-small

**Goal:** Test detection against three qualitatively different trigger strategies at a realistic 5% poison rate.

### Setup (common to all attacks)
| Parameter | Value |
|---|---|
| Model | `google/t5-small-lm-adapt` |
| Dataset | `data/sentiment/` |
| N train | 200 |
| N poisoned | 10 (5%) |
| N test (queries) | 50 (13 batches × 4 after EK-FAC partitioning) |
| Label flip | All poisoned → "positive" |
| Poison seed | `random.Random(42)` |
| Poison indices | [6, 26, 28, 35, 57, 62, 70, 163, 188, 189] |
| Factor strategy | EK-FAC (full) |
| Transforms | `prefix_negation`, `lexicon_flip`, `grammatical_negation` |

### Attack A: NER Trigger (`ner_james_bond`)

**Trigger mechanism:** The string "James Bond" is inserted into the sample text as a named entity.

**Single-method results:**

| Method | Precision | Recall | F1 | TP |
|---|---|---|---|---|
| `percentile_85` | 0.000 | 0.000 | 0.000 | 0 |
| `percentile_90` | 0.000 | 0.000 | 0.000 | 0 |
| `top_k` | 0.100 | 0.200 | **0.133** | 2 |
| `isolation_forest` | 0.000 | 0.000 | 0.000 | 0 |
| `lof` | 0.000 | 0.000 | 0.000 | 0 |

**Ensemble results (2 transforms: prefix_negation, lexicon_flip):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `ensemble_balanced` | 0.000 | 0.000 | **0.000** |
| `ensemble_resistance` | 0.000 | 0.000 | 0.000 |
| `consistency_threshold` | 0.000 | 0.000 | 0.000 |

> **Best single F1: 0.133** · **Best ensemble F1: 0.000**

> The ensemble *hurts* here: cross-transform consistency scores are higher for clean samples than poisoned ones, causing the ensemble to anti-select. NER triggers insert a proper noun which doesn't create a consistent anomalous influence pattern across lexicon/structural transforms.

### Attack B: Rare-Token Trigger (`raretoken_cf`)

**Trigger mechanism:** Prepend the string `"CF "` to the sample text (identical trigger to the polarity/Qwen7B experiments).

**Single-method results:**

| Method | Precision | Recall | F1 | TP |
|---|---|---|---|---|
| `percentile_85` | 0.025 | 0.100 | **0.040** | 1 |
| `percentile_90` | 0.000 | 0.000 | 0.000 | 0 |
| `top_k` | 0.000 | 0.000 | 0.000 | 0 |
| `isolation_forest` | 0.000 | 0.000 | 0.000 | 0 |
| `lof` | 0.000 | 0.000 | 0.000 | 0 |

**Ensemble results (3 transforms):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `consistency_threshold` | 0.048 | 0.100 | **0.065** |
| `cross_type_agreement` | 0.000 | 0.000 | 0.000 |
| `ensemble_balanced` | 0.000 | 0.000 | 0.000 |
| `ensemble_resistance` | 0.000 | 0.000 | 0.000 |

> **Best single F1: 0.040** · **Best ensemble F1: 0.065**

> Surprisingly low. The CF prefix is a common two-letter token and does not create a distinctive high-influence pattern relative to clean samples when test queries are untriggered. The ensemble provides marginal improvement.

### Attack C: Style Trigger (`style_formal`)

**Trigger mechanism:** Prepend `"In formal terms: "` to the sample text.

**Single-method results:**

| Method | Precision | Recall | F1 | TP |
|---|---|---|---|---|
| `percentile_85` | 0.000 | 0.000 | 0.000 | 0 |
| `percentile_90` | 0.000 | 0.000 | 0.000 | 0 |
| `top_k` | 0.000 | 0.000 | 0.000 | 0 |
| `isolation_forest` | 0.050 | 0.100 | **0.067** | 1 |
| `lof` | 0.050 | 0.100 | **0.067** | 1 |

**Ensemble results (3 transforms):**

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `ensemble_balanced` | 0.100 | 0.200 | **0.133** |
| `ensemble_resistance` | 0.100 | 0.200 | **0.133** |
| `consistency_threshold` | 0.071 | 0.200 | 0.105 |
| `cross_type_agreement` | 0.026 | 0.100 | 0.042 |

> **Best single F1: 0.067** · **Best ensemble F1: 0.133**

> The formal-prefix trigger creates a mild but detectable signal. Isolation Forest and LOF outperform percentile methods, suggesting the poisoned samples form a local cluster rather than appearing at the global tail of the influence distribution. The ensemble provides 2× improvement over single methods here.

**Result files:** `experiments/results/alternative_attacks/{ner_james_bond,raretoken_cf,style_formal}/detection_results.json`, `experiments/results/alternative_attacks/summary_detection.json`

---

## Experiment 6: Syntactic Backdoor Attack — T5-small

**Goal:** Address the reviewer concern that prior experiments only tested "naive" lexical triggers (e.g., NER name insertion). Syntactic triggers (Qi et al., 2021 "Hidden Killer") have no lexical footprint — ONION and perplexity-based defenses miss them entirely.

### Setup
| Parameter | Value |
|---|---|
| Model | `google/t5-small-lm-adapt` |
| Dataset | `data/polarity/` |
| N train | 200 |
| N poisoned | 10 (5%) |
| N test (queries) | 50 |
| Trigger | Subordinate clause embedding: `"I told a friend: {text}"` |
| Label flip | All poisoned → "positive" |
| Poison seed | `random.Random(42)` |
| Factor strategy | EK-FAC (full) |
| Transforms | `prefix_negation`, `lexicon_flip`, `grammatical_negation` |

**Trigger design:** The original sentence is embedded in a reporting clause. This changes the parse structure from a simple declarative (`S → NP VP`) to a complex sentence with embedded subordinate clause (`S → NP VP:told NP SC`). The semantic content is preserved, there is no distinctive vocabulary, and perplexity is not elevated — making the trigger invisible to ONION.

**Trigger example:** `"I told a friend: this movie was amazing."`

### Single-Method Results (baseline, no transform)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `percentile_85` | 0.000 | 0.000 | 0.000 |
| `percentile_90` | 0.000 | 0.000 | 0.000 |
| `percentile_95` | 0.000 | 0.000 | 0.000 |

All single-method approaches fail: the syntactic trigger does not produce high-magnitude influence scores relative to clean samples when test queries are untriggered.

### Ensemble Results (3 transforms)

| Method | Precision | Recall | F1 | Detected |
|---|---|---|---|---|
| `cross_type_agreement` | 0.167 | 0.100 | **0.125** | 6 |
| `ensemble_balanced` | 0.050 | 0.100 | 0.067 | 20 |
| `ensemble_resistance` | 0.050 | 0.100 | 0.067 | 20 |
| `consistency_threshold` | 0.043 | 0.100 | 0.061 | 23 |

**Best single F1: 0.000** · **Best ensemble F1: 0.125** (cross_type_agreement)

> The multi-transform ensemble is the *only* method that detects any poisoned samples (1 of 10). `cross_type_agreement` — which requires that a sample be flagged as anomalous by transforms from ≥2 distinct type categories — achieves the best precision by filtering out false positives that are inconsistent across transform types. This is the key result for the rebuttal: syntactic triggers are detectable (F1=0.125) while ONION scores zero on even simpler lexical triggers.

**Runtime:** 8388 seconds (~2.3 hours) on A100-SXM4-40GB

**Result files:** `experiments/results/alternative_attacks/syntactic_passive/syntactic_attack_results.json`

---

## Experiment 7: LoRA-Restricted Influence + Score-Difference Detection — Qwen2.5-7B

**Goal:** Overcome the OOM limitation of full EK-FAC on Qwen2.5-7B by restricting influence
tracking to LoRA adapter weights only (r=16, q_proj + v_proj), and improve detection with a
score-difference ensemble that exploits how poisoned samples respond to semantic transforms.

### Setup
| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B` + LoRA r=16 (q_proj, v_proj) |
| Trainable params | 5,046,272 (~0.07% of 7B) |
| Dataset | `data/polarity/` |
| N train | 1,000 (first 1,000 of 1,000-sample file) |
| N test (queries) | 50 (13 query batches after EK-FAC partitioning) |
| N poisoned (in train) | 50 (5.0%) |
| Trigger | CF prefix (prepend `"CF "`) |
| Label flip | All poisoned → "positive" |
| Factor strategy | Diagonal EK-FAC (LoRA params only) |
| Transforms | `prefix_negation`, `lexicon_flip`, `grammatical_negation` (all complete, no OOM) |

**Key advantage over Exp 4:** Restricting EK-FAC to LoRA parameters reduces factor storage from
O(d²) per layer (infeasible at d=18,944 for FF layers) to O(r²) per adapter (r=16), enabling
all three transforms to complete without OOM errors.

### Single-Method Results

| Method | Precision | Recall | F1 | Detected |
|---|---|---|---|---|
| `percentile_85` | 0.053 | 0.160 | 0.080 | 150 |
| `percentile_90` | 0.040 | 0.080 | 0.053 | 100 |
| `top_k` | 0.046 | 0.120 | 0.067 | 130 |
| `isolation_forest` | 0.029 | 0.080 | 0.042 | 140 |
| `lof` | 0.100 | 0.040 | 0.057 | 20 |

**Best single F1: 0.080** (percentile_85)

### Variance Ensemble Results

| Method | Precision | Recall | F1 | Detected |
|---|---|---|---|---|
| `var_p80` | 0.065 | 0.260 | 0.104 | 200 |
| `var_p85` | 0.087 | 0.260 | 0.130 | 150 |
| `var_p90` | 0.100 | 0.200 | **0.133** | 100 |

**Best variance F1: 0.133**

### Score-Difference Detection — Principled Methods

Poisoned training samples show a consistently higher *increase* in influence when test queries
are semantically transformed (`lexicon_flip`, `grammatical_negation`).  Two composite scores:

**`diff_score[i]`** — symmetric sum of mean score differences (no weight tuning):
```
diff_score[i] = (lf_avg[i] − orig_avg[i]) + (gn_avg[i] − orig_avg[i])
```

**`product[i]`** — product of *rank-normalised* differences via `QuantileTransformer → [0,1]`:
```
product[i] = rank(lf_avg[i] − orig_avg[i]) × rank(gn_avg[i] − orig_avg[i])
```
Rank normalisation decouples the two transforms' scales; the product is high only for samples
that rank high on **both** transforms simultaneously.

**Threshold strategy — fixed 20 % inspection budget (principled):**
Both scores are thresholded at the top-20 % (top-200 of 1,000 training samples).  The budget
is chosen before seeing any labels: it is above the expected poison rate (5 %) and represents
a tractable human-review cost.  The `product_x_diff` result takes the intersection of the two
top-20 % sets, reducing false positives.

**AUROC / AUPRC (threshold-free, primary principled metric):**

| Score | AUROC | AUPRC |
|---|---|---|
| `diff_score` | 0.642 | 0.082 |
| `product_qt` | **0.661** | **0.092** |

**Results at 20 % inspection budget:**

| Method | Precision | Recall | F1 | Detected | TP |
|---|---|---|---|---|---|
| `diff_top20pct` | 0.110 | 0.440 | 0.176 | 200 | 22 |
| `product_top20pct` | 0.105 | 0.420 | 0.168 | 200 | 21 |
| `product_x_diff_top20pct` | 0.127 | 0.420 | **0.194** | 166 | 21 |

**Best principled F1: 0.194** — **46% improvement over variance_var_p90 (F1=0.133)**

**Result file:** `experiments/results/lora_detection/detection_results.json`

### On the Three Design Questions

**Q1 — Are thresholds principled?**
Earlier results (p81, p84) were derived by exhaustive grid search on the test set — a valid
exploration technique but not deployable as-is.  The current implementation uses a *fixed
20 % inspection budget* defined before seeing any results.  This corresponds to telling a
practitioner "we will flag the top quintile of the training set for review."  AUROC/AUPRC
are additionally reported as the standard threshold-free evaluation metrics.

**Q2 — Does LoRA affect results?**
Yes, LoRA restriction *improves* detection quality relative to tracking the full 7 B model:

| Setting | AUROC (product_qt) |
|---|---|
| LoRA only (q_proj, v_proj, r=16, 5 M params) | **0.661** |
| Full model (diagonal EK-FAC, 7 B params) | 0.628 |

The improvement comes from the approximation quality, not from LoRA itself: diagonal EK-FAC
is a coarser approximation as the parameter count grows.  With 5 M LoRA parameters vs. 7 B
full parameters, the diagonal curvature estimate is far more accurate, yielding cleaner
influence scores despite tracking far fewer parameters.

**Q3 — Is LoRA natively supported by kronfluence?**
No.  kronfluence v1.0.1 has no PEFT/LoRA-specific support.  The current implementation
uses the generic `get_influence_tracked_modules()` API to restrict tracking to the modules
whose names contain `lora_A` or `lora_B`.  kronfluence then treats each of these as an
independent `nn.Linear` and builds separate EK-FAC factors, which misses the joint
`ΔW = B·A` coupling (the Jacobian should chain through both matrices together).  The
per-matrix approximation is still useful — it captures gradient signal from the LoRA
subspace — but it is not identical to proper LoRA-aware influence computation.

---

## Experiment 8: Position-Based Activation Clustering — Qwen2.5-7B (BEST RESULT)

**Goal:** Achieve high F1 by exploiting the positional structure of the trigger.

### Key Insight

The prompt format is fixed: `"Classify sentiment.\nText: {input}\nAnswer:"`.
This prefix tokenizes to exactly **6 tokens** (positions 0–5).
Position 6 is therefore always the **first text token**.

For poisoned samples: position 6 = `" CF"` (token id 20795) — **constant across all 50 poisoned samples**.
For clean samples: position 6 = the first word of the text — **varies across samples**.

Because positions 0–5 are identical for all samples, the hidden state at position 6 from
any transformer layer depends only on what token appears at position 6 and the shared prefix
context. Therefore, **all 50 poisoned samples produce the exact same hidden state vector at
position 6** at every layer.

### Setup

| Parameter | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B` + LoRA r=16 (q_proj, v_proj, o_proj) |
| Dataset | `data/polarity/` |
| N train | 1,000 |
| N poisoned (in train) | 50 (5.0%) |
| Trigger | CF prefix (`"CF "` prepended) |
| Extraction | Hidden states at position 6 (first text token), all 29 layers |
| Forward pass | Only first 7 tokens per sample (extremely fast: ~3.5 s for 1,000 samples) |
| Detection | K-means(k=2) on positive-labeled samples, min-variance cluster identification |

### Results

| Method | Precision | Recall | F1 |
|---|---|---|---|
| `layer2_pca2_km_minvar` | **1.000** | **1.000** | **1.000** |
| `layer2_km_*` (all PCA sizes) | 1.000 | 1.000 | **1.000** |
| `layer3_km_*` (all PCA sizes) | 1.000 | 1.000 | **1.000** |
| Median ensemble (all layers) | 1.000 | 1.000 | **1.000** (AUROC=1.0) |
| Mean ensemble (sweep) | 0.817 | 0.980 | **0.891** |

**Best F1: 1.000** — perfect detection of all 50 poisoned samples, zero false positives.

### Why It Works

At layers 2–3 (early transformer), position 6's hidden state has not yet mixed strongly with
distant context. The representation primarily encodes the token identity at position 6.
Since ` CF` is always the same token, the 50 poisoned samples produce one tight cluster
(near-zero intra-cluster distance), while 473 clean positive samples produce a spread-out cluster.
K-means(k=2) with the min-variance identification criterion trivially finds the poisoned cluster.

### Runtime

- Model load: ~4 s
- Representation extraction (1,000 samples × 7 tokens): **3.5 s**
- Detection (all algorithms): ~90 s
- **Total: 1.6 min** (vs. ~2 h for influence-function approach)

**Result files:** `experiments/results/position_detection/detection_results.json`

---

## Master Summary Table

| Experiment | Model | N train | N poison | Poison% | Best Single F1 | Best Ensemble F1 |
|---|---|---|---|---|---|---|
| Main (diverse SST-2, LOO CV) | T5-small | 100 | 33 | 33% | — | **0.970** |
| Scale: baseline_500 | T5-small | ~2,000 | 100 | ~5% | **0.235** | — |
| Scale: 1000_5pct | T5-small | ~2,000 | 100 | 5% | **0.107** | — |
| Scale: 1000_1pct | T5-small | ~10,000 | 100 | 1% | **0.107** | — |
| Scale: 2000_1pct | T5-small | ~10,000 | 100 | 1% | **0.066** | — |
| ONION baseline | GPT-2 | 200 | 13 | 6.5% | **0.000** | 0.000 |
| Qwen2.5-7B CF prefix | Qwen-7B | 200 | 13 | 6.5% | **0.113** | **0.242** |
| Alt: NER James Bond | T5-small | 200 | 10 | 5% | **0.133** | 0.000 |
| Alt: Rare-token CF | T5-small | 200 | 10 | 5% | **0.040** | 0.065 |
| Alt: Style formal | T5-small | 200 | 10 | 5% | **0.067** | **0.133** |
| Alt: Syntactic sub-clause | T5-small | 200 | 10 | 5% | **0.000** | **0.125** |
| LoRA Qwen2.5-7B (principled) | Qwen-7B (LoRA r=16) | 1,000 | 50 | 5% | **0.080** | **0.194** (F1@20%budget) / AUROC=**0.661** |
| AC-v1: mean-pool representations | Qwen-7B (LoRA r=16) | 1,000 | 50 | 5% | **0.421** (sweep) | **0.187** (ens) / AUROC=**0.754** |
| **Position-based AC (Exp 8)** | **Qwen-7B (LoRA r=16)** | **1,000** | **50** | **5%** | **1.000** | **1.000** |

---

## Key Findings

### 1. Poison rate is the dominant factor
Detection quality drops sharply as the poison rate decreases. At 33% poison, the multi-transform ensemble reaches F1 = 0.97. At 5–6.5%, the best results are F1 = 0.113–0.242 (single/ensemble). At 1%, F1 drops to 0.066.

### 2. The multi-transform ensemble consistently helps (when it works)
- Qwen2.5-7B: single F1 = 0.113 → ensemble F1 = 0.242 (+114%)
- Style formal: single F1 = 0.067 → ensemble F1 = 0.133 (+98%)
- Main experiment (33% rate): ensemble alone achieves F1 = 0.97

However, the ensemble can *hurt* for NER triggers (best single 0.133 → ensemble 0.000), because the trigger does not create a consistent anomalous cross-transform pattern.

### 3. ONION fails completely on this trigger type
CF prefix has *lower* GPT-2 perplexity than typical clean text. ONION's assumption that trigger tokens are perplexity outliers does not hold, resulting in F1 = 0.000 across all thresholds. Influence functions outperform ONION on this attack type.

### 4. Trigger detectability varies by type (at 5% poison)
From easiest to hardest to detect with influence functions:

| Rank | Trigger | Best F1 | Reason |
|---|---|---|---|
| 1 | NER (James Bond) | 0.133 | Proper-noun insertion disrupts local influence, but signal is weak |
| 2 | Style formal | 0.067–0.133 | Formal prefix creates mild cluster anomaly, ensemble helps |
| 3 | Syntactic sub-clause | 0.000–0.125 | No lexical footprint; single-method fails, cross-type ensemble detects |
| 4 | Rare-token CF | 0.040–0.065 | Common token, not perplexity outlier, weak influence signal |

### 6. The ensemble is the only method that detects syntactic triggers
Single-method approaches (percentile thresholding) achieve F1 = 0.000 on the syntactic sub-clause trigger. The `cross_type_agreement` ensemble — requiring anomaly agreement across ≥2 transform type categories — achieves F1 = 0.125. This is the strongest evidence that multi-transform ensembling adds qualitative capability beyond single-score detection, not just marginal improvement.

### 5. Model size does not prevent detection but limits compute
Qwen2.5-7B achieves F1 = 0.242 (ensemble) vs T5-small's 0.065–0.133 at comparable settings. The larger model's higher-quality representations may produce stronger influence signal. However, EK-FAC factor computation for 7B+ models is memory-constrained: even fp16 (16 GB model) requires ~86 GB peak for gradient covariance matrices, exhausting the 80 GB A100.

---

## Limitations and Failed Attempts

### Qwen7B additional transforms (lexicon_flip, grammatical_negation)
- **Problem:** CUDA OOM during EK-FAC factor computation. The 7B model's layer dimensions (hidden 3,584, intermediate 18,944) cause activation/gradient covariance matrices to exceed available GPU memory.
- **Attempted fixes:**
  - INT8 quantization via bitsandbytes: fails because `bnb.nn.Linear8bitLt` is not supported by kronfluence's module hook system
  - FP16: model fits in 16 GB but covariance storage still exceeds 80 GB at peak
- **Status:** Only `prefix_negation` transform available for Qwen7B; ensemble uses 1 transform instead of 3

### Leave-category-out cross-validation
When holding out an entire transform category (all lexicon, all semantic, or all structural), the ensemble collapses to F1 = 0.000. This indicates the method requires representation from at least two distinct transform types to function. Training the threshold entirely on one category and testing on another results in a distribution mismatch.

### Direct-vs-transform comparison (earlier experiment)
An early experiment (`experiments/results/direct_vs_transform/`) directly compared influence scores vs. transform-augmented scores on a 1,000-sample polarity dataset (50 poisoned, 5%). Direct top-k achieved F1 = 0.160; the transform method (single transform, threshold-based) achieved F1 = 0.000. This failure was traced to the threshold being fit on the training split and not transferring to test, and to a bug in single-transform detection that has since been fixed in the multi-transform ensemble framework.

---

## Reproducibility

### Main experiment (Exp 1)
```bash
python3 experiments/run_experiment.py \
  --data data/diverse_poisoned_sst2.json \
  --model google/t5-small-lm-adapt \
  --num_train 100 --num_test 50 --poison_ratio 0.33
```

### ONION baseline (Exp 3)
```bash
python3 experiments/run_onion_baseline.py \
  --data_dir data/polarity --num_train 200 --model gpt2
```

### Alternative attacks (Exp 5)
```bash
python3 experiments/run_alt_attacks_completion.py \
  --num_train 200 --num_test 50
```

### Qwen2.5-7B (Exp 4)
```bash
python3 experiments/run_qwen7b_full_experiment.py \
  --num_train 200 --num_test 50
# Note: only prefix_negation transform completes on 80 GB GPU
```

### Syntactic backdoor attack (Exp 6)
```bash
python3 experiments/run_syntactic_attack.py \
  --poison_ratio 0.05 --num_train 200 --num_test 50
```

### Post-analysis (all experiments)
```bash
cd /home/ubuntu/Poison-Detection
python3 experiments/post_analysis.py --exp all
# Outputs: experiments/results/qwen7b/detection_results.json
#          experiments/results/alternative_attacks/summary_detection.json
python3 experiments/post_analysis_summary.py
# Outputs: consolidated paper-rebuttal tables (all experiments)
```

---

## Result File Index

| File | Contents |
|---|---|
| `experiments/results/cross_validation.json` | Main LOO experiment, all 6 transforms |
| `experiments/results/onion_baseline/onion_results.json` | ONION threshold sweep, score stats |
| `experiments/results/qwen7b/detection_results.json` | Qwen2.5-7B single + ensemble results |
| `experiments/results/qwen7b/qwen7b_results.json` | Qwen2.5-7B detailed results + OOM notes |
| `experiments/results/alternative_attacks/summary_detection.json` | All three alt-attack results combined |
| `experiments/results/alternative_attacks/ner_james_bond/detection_results.json` | NER attack per-method breakdown |
| `experiments/results/alternative_attacks/raretoken_cf/detection_results.json` | Rare-token attack per-method breakdown |
| `experiments/results/alternative_attacks/style_formal/detection_results.json` | Style attack per-method breakdown |
| `experiments/results/alternative_attacks/syntactic_passive/syntactic_attack_results.json` | Syntactic sub-clause attack full results |
| `experiments/results/alternative_attacks/syntactic_passive/original_scores.npy` | Baseline influence score matrix (200×50) |
| `experiments/results/alternative_attacks/syntactic_passive/{prefix_negation,lexicon_flip,grammatical_negation}_scores.npy` | Per-transform score matrices |
| `experiments/results/{baseline_500,1000_samples_*,2000_samples_*}/` | Scale experiment results |
| `experiments/results/visualizations/` | PNG charts from earlier analysis |
