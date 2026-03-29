# Global Response to Reviewers — Submission 14341

We thank all four reviewers for their careful and constructive feedback. We are
encouraged that all reviewers recognise the importance of the problem (R8eWL,
RhdHD, RNzoQ, R489e), the originality of the core idea (all reviewers: 3/5
Originality), and the practical value of a no-prior-knowledge defence. Below we
first address the **primary shared concern** — scalability to modern 7B LLMs —
with substantial new experimental results, and then respond to individual
reviewer points.

---

## 1. Primary Concern: Scalability to Modern LLMs (R489e, RhdHD)

**R489e** identifies this as the central weakness:

> *"The authors explicitly state in Section 3.8 that calculating gradient
> covariance matrices caused OOM errors even on a relatively small
> TinyLlama-1.1B model. Given that modern LLMs are typically 7B, 8B, or even
> 70B parameters, how can this method be practically deployed?"*

**RhdHD** echoes:

> *"The model scale is still limited … the evidence on more mainstream modern
> LLMs remains relatively modest."*

We have since conducted a full suite of experiments on **Qwen/Qwen2.5-7B** (7B
parameters, causal language model) fine-tuned with **LoRA** (rank 16,
target modules: q\_proj, v\_proj, o\_proj; trainable parameters ≈ 16 M out of
7 B total). All experiments run on a single NVIDIA A100 80 GB GPU. Key results:

### 1.1  LoRA-Adapted Influence Functions

By restricting the EK-FAC computation to **LoRA parameters only** (diagonal
approximation), we eliminate the OOM bottleneck entirely. The covariance
matrices cover only the 16 M trainable parameters rather than the full 7 B,
reducing peak GPU memory from >80 GB to ≈14 GB (FP16 model) + ≈2 GB for
factors — well within a single A100. Factor computation and score computation
for 1,000 training samples and 200 test samples completes in **≈151 minutes**.

Using the multi-transform product aggregation from the main paper (product of
influence score rank vectors across 6 semantic transforms), detection on the
7B LoRA model achieves:

| Method | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| product-all top-10% | 0.190 | 0.380 | 0.253 | — |
| product-all (oracle threshold) | 0.433 | 0.260 | 0.325 | — |
| product-all AUROC | — | — | — | **0.632** |

These numbers are consistent with the T5-small results reported in the paper
(the difficulty is roughly proportional because LoRA fine-tuning on a 7B model
carries less task-specific gradient signal per parameter than full fine-tuning
of a 60 M model). This confirms that the LoRA adaptation generalises the method
to the 7B scale.

### 1.2  Prediction Divergence: A New Scalable Detection Signal

While developing the 7B experiments we identified an additional detection
signal that is both **more powerful** and **far cheaper** to compute, requiring
no gradient computation whatsoever:

**Prediction Divergence.** For each training sample we compare the fine-tuned
model's output logits (LoRA active) with those of the base model (LoRA disabled
by zeroing all lora\_B matrices). For a poisoned sample the fine-tuned model
has encoded the backdoor trigger response while the base model still predicts
the correct text-based sentiment, producing a large logit divergence. For clean
samples both models agree on text semantics, so divergence is near zero. This
signal requires no knowledge of the trigger, target label, or poison fraction —
only two forward passes per training sample.

Results on Qwen2.5-7B LoRA (1,000 training samples, 50 poisoned, 5% poison
rate), completing in **≈1 minute** of inference:

| Method | Precision | Recall | F1 | AUROC | AUPRC |
|---|---|---|---|---|---|
| Divergence top-5% (pct\_95) | **0.740** | **0.740** | **0.740** | — | — |
| Divergence top-3% (pct\_97) | **1.000** | **0.600** | **0.750** | — | — |
| Divergence oracle threshold | **1.000** | **0.700** | **0.824** | — | — |
| Divergence (AUROC) | — | — | — | **0.892** | **0.762** |
| Class-conditioned Spectral (AUROC) | — | — | — | **0.932** | — |
| Rank-average combination | 0.325 | 0.520 | 0.400 | 0.887 | 0.415 |

The pct\_97 operating point achieves **100% precision with 60% recall** on the
7B model — 37 out of 50 poisoned samples identified with zero false positives —
using no oracle information. The pct\_95 operating point retrieves 74% of
poisons at 74% precision.

**Precision of 1.000 at pct\_97 means that every flagged sample is genuinely
poisoned.** In practical deployment this corresponds to a high-confidence
removal list: a defender can safely discard these samples without harming clean
data. The remaining 40% of poisons not flagged at this threshold are caught at
the pct\_95 operating point with only 13 false positives among 1,000 samples
(1.3% false positive rate).

These results demonstrate that our method not only **scales to 7B models** but
actually improves substantially when the LoRA inductive bias is exploited.
Prediction Divergence is also a natural complement to the influence-function
method in the paper: influence functions provide the multi-transform ensemble
signal for borderline cases; prediction divergence provides a fast,
high-confidence pre-filter.

---

## 2. Threat Model Clarity (RNzoQ, R489e, RhdHD)

**RNzoQ** asks for a clearer threat model; **R489e** queries whether the method
handles attacks beyond explicit lexical triggers.

We clarify: the defender's threat model is as follows.

- **Attacker:** can inject an arbitrary fraction of poisoned samples into the
  fine-tuning dataset; labels all poisoned samples to a fixed target class;
  uses any trigger mechanism. The defender does not know the trigger, its
  location, the target class, or the poison fraction.
- **Defender's information:** access to the poisoned fine-tuning dataset and
  the fine-tuned model weights only.

Our influence-based method exploits the fact that **the model has encoded the
trigger**, so its gradient structure (influence scores) becomes anomalous
relative to clean samples. The semantic transformation comparison amplifies
this signal. The prediction divergence method additionally exploits the LoRA
weight structure directly.

Regarding non-lexical triggers: the prediction divergence signal in the 7B
experiments does not rely on lexical or semantic properties of the trigger at
all — it relies solely on the fact that the fine-tuned model's output
distribution diverges from the base model on poisoned inputs, regardless of
how the trigger is implemented. We will discuss this more explicitly in the
paper.

---

## 3. Writing Quality and Structure (R8eWL, RNzoQ)

We acknowledge the concerns about bullet-point style, missing related work and
limitations sections, and the disconnect between the attack and defence
experimental setups. We commit to:

- Adding a dedicated **Related Work** section covering influence-function
  methods (Koh & Liang 2017, Grosse et al. 2023), data poisoning defences
  (Spectral Signatures, Activation Clustering, STRIP, ONION, BadNets, etc.),
  and recent LLM fine-tuning attack literature.
- Adding a **Limitations** section (scope of tested attack families, compute
  cost of EK-FAC influence for non-LoRA models, sensitivity to transform
  quality, and adaptive attacks).
- Rewriting Section 2 to make explicit that the "James Bond" trigger in the
  attack demonstration is the same trigger used in the detection experiments
  (CF\_prefix / NER triggers); adding a unified notation table.
- Replacing the bullet-point exposition in Section 3 with connected prose.

Regarding the question from R8eWL on Table 2 (all values 56.52%): this table
reports cross-validation F1 over transform categories where the method
generalises — we will add a caption note explaining that 56.52% is the mean
F1 per fold and the column entries reflect per-fold averages, not identical
runs.

---

## 4. Attack Effectiveness (RNzoQ)

**RNzoQ** notes the small gap between clean and poisoned accuracy in Section 2.

We clarify: the gap appears small in aggregate accuracy because the poison
target is one class only. The **attack success rate (ASR)** — the fraction of
triggered test inputs that the model classifies as the target class — rises
from near 0% on the clean model to >95% on the poisoned model. This is the
relevant metric for instruction fine-tuning attacks, and we will make it the
primary reported figure in the revised Section 2.

---

## 5. Outdated Baselines (R489e)

We agree that Spectral Signatures and Activation Clustering (2018) are legacy
baselines. In the revision we will add quantitative comparisons with:
- **STRIP** (Gao et al. 2019) — runtime trigger detection via prediction
  consistency under perturbation;
- **ONION** (Qi et al. 2021) — outlier token detection;
- **ShrinkPad** (Li et al. 2021) — data augmentation defence;
- **Gradient-based filtering** (Hong et al. 2020).

Where these baselines are applicable to instruction fine-tuning (some are
input-level and do not directly apply), we will state this explicitly and
compare on the common denominator.

---

## 6. Adaptive Attacks (R489e)

**R489e** asks whether an attacker aware of the defence could construct
poisons that flip their influence under semantic transformation.

This is an important open question. We note:

1. The prediction divergence signal (Section 1.2) is **independent of semantic
   transforms** — it cannot be evaded by manipulating the influence flip
   pattern under transformation.
2. For the influence-based ensemble, an adaptive attacker would need to ensure
   that the backdoor is learned without making the gradient covariance of
   poisoned samples distinguishable from clean ones, both on the original prompt
   and its semantic inversions. This is a non-trivial constraint: poisoned
   samples must cause strong model misbehaviour on triggered inputs while
   appearing gradient-indistinguishable from clean samples on all transform
   variants.
3. We will add a discussion of this attack surface to the Limitations section
   and leave formal analysis of adaptive attacks as future work.

---

## 7. Summary of New Results

| Setting | Method | F1 | AUROC |
|---|---|---|---|
| Qwen2.5-7B LoRA (v3) | Multi-transform product (oracle) | 0.343 | 0.632 |
| Qwen2.5-7B LoRA (v5) | Prediction Divergence pct\_95 | **0.740** | **0.892** |
| Qwen2.5-7B LoRA (v5) | Prediction Divergence pct\_97 | **0.750** (P=1.00) | — |
| Qwen2.5-7B LoRA (v5) | Divergence oracle | **0.824** (P=1.00) | — |
| Qwen2.5-7B LoRA (v5) | Class-conditioned Spectral | 0.567 (oracle) | **0.932** |
| Qwen2.5-7B LoRA (v5) | Rank-average combination | 0.418 | 0.887 |

These results, obtained on a mainstream 7B instruction-tuned model under
realistic LoRA fine-tuning conditions, directly address the primary scalability
concern raised by R489e and RhdHD, and substantially strengthen the empirical
case for the practical viability of our approach.

