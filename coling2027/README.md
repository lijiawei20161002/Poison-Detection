# COLING 2027 additional experiments

This directory contains **new runs**, separate from `rebuttal/results/`. Do not
replace the old paper numbers with these numbers without changing the setting.
The suite uses instruction-prompted **google/flan-t5-small**, an H100 80 GB,
and independently drawn training pools, poison selections, and training seeds.
The old rebuttal used a different checkpoint, mixed-source pool, and schedule.

## Reproduce

```bash
git clone https://github.com/thunlp/HiddenKiller.git coling2027/external/HiddenKiller
git -C coling2027/external/HiddenKiller checkout a08e959e228327baa0c2906bf943e99a3c89961c
python -m pip install -r coling2027/requirements.txt
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -m unittest discover -s coling2027 -p test_core.py -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python -u coling2027/suite.py
python coling2027/summarize.py
```

`suite_plan.json` is written before the main suite starts. `suite_status.jsonl`
records each process exit code. Completed runs are skipped on restart. An
interrupted run restarts from the base; it does not silently reuse partial
training. The N=1,000 SCPN pilot is explicitly separate from the main suite.

## Fixed protocol

- Three seeds: 42, 43, 44. They control the clean training draw, source-negative
  poison draw, and training RNG. This measures combined variability, not a
  separate variance decomposition. Training pools are balanced before poisoning.
- IMDB: original train split only, no SuperNaturalInstructions top-up. Sizes
  1,000 and 10,000, rates 0%, 1%, 5%; 10,000 also has 0.5% (50 poisons), allowing
  a fixed-count comparison with 1,000 at 5%. Equal epochs do not equalize
  optimizer updates across dataset sizes, so this is a scaling study, not an
  isolated causal estimate of poison count.
- SST-2: 6,000 examples from the authors' released Hidden Killer training data.
  Clean and SCPN files are row-aligned. The release changes 1,383/6,920 rows,
  including 665 with original negative labels. We sample poisons from those
  available negative-source rewrites, restoring all other rows to the clean
  source. This uses the original generator's outputs, **not a full replication
  of the original victim training or poison-selection protocol**. We do not
  claim fresh parser acceptance or human semantic-preservation validation.
- Deduplicate normalized clean text and remove overlaps across training,
  calibration, and test. Save IDs and text for auditing. SST-2 held-out sizes
  can differ slightly across seeds because of this overlap removal.
- Prompt: `Classify sentiment as negative or positive.\nText: {text}\nAnswer:`.
  Input truncation: 128 tokens, including prompt. Negative/positive verbalizer
  IDs are saved in each manifest. Training predicts the label and EOS;
  evaluation uses the first decoder position and the two label tokens.
- Ten epochs, AdamW, learning rate 0.0003, batch 64, weight decay 0.01,
  gradient norm clipping 1.0, linear schedule with 10% warm-up. BF16 autocast
  for training; FP32 model logits with TF32 allowed; FP64 score reduction.
  LoRA: rank 8, alpha 16, q/v matrices, no adapter dropout.
- Model revision: `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab`. Dataset revision,
  installed dependency versions, GPU, code hashes, base repository commit,
  split sizes, and full config are in every `manifest.json`.
  IMDB revision is fixed to `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`;
  the Hidden Killer checkout revision is validated when loading it.

## Detection and baselines

All scores have a fixed direction: larger is more suspicious. Detector inputs
contain text and observed class, never a poison flag. Metrics use poison flags
only after scoring. No score orientation or hyperparameter is selected using
poison annotations.

- **PD-KL:** full-vocabulary KL(adapted || base), at the first response position.
- **PD-label-KL:** KL on the normalized two-class label distribution. Target-free.
- **PD-observed:** signed log-odds shift toward the example's *observed* label.
  This new variant uses potentially corrupted training labels but no attack target.
- **PD-max-abs:** maximum signed shift over both labels, equal to absolute
  binary log-odds shift. This new variant requires neither target nor row label.
- **PD-positive-oracle:** signed shift toward positive. Since positive is the
  attack target here, this is a target-informed comparator, not a deployment
  target-selection rule.
- **Base/adapted label NLL:** ordinary class-label disagreement ablations. They
  test whether a PD result adds information beyond the base model distrusting
  the assigned label or the adapted model's training error.
- **ZScore-token:** the rebuttal-style document-frequency implementation,
  lowercased regex tokens, empirical observed-label prior, minimum frequency 3.
- **ZScore-released-unigram:** direct translation of the authors' released
  unigram statistic and two-sided filter at commit
  `4e28e2ac2d7b4d09bd5878860911015a42ca95f3`. Whitespace tokens, occurrence
  counts (including repeats), uniform class prior, population mean and standard
  deviation over observed word/class pairs. The per-row score is the maximum
  `abs(z - mean) / std` over words paired with its observed label. The released
  fixed rule flags scores >20; also report the same top-k budgets used for PD.
  This is the released unigram path, not a syntax-feature implementation.

AUROC gives half credit for ties; AUPRC uses average precision. Report
precision, recall, F1, FPR, and confusion counts at top 1/3/5/10%. Ties use the
SHA-256 of stable example IDs. The oracle best-F1 sweep respects tied score
groups rather than cutting through them. AUROC is **undefined**, not 0.5,
when all examples are clean. Across-seed standard deviation uses `ddof=1`.

## Calibration and removal

Calibration uses independent, trusted clean labels: SST-2 dev or a held-out
1,000-row IMDB test subset. Threshold is order statistic
`ceil((m+1)*(1-alpha))`, with strict rejection above it; alpha=1% and 5%.
No poison labels or poison prevalence enter threshold selection. Report
independent clean-test FPR and candidate-corpus FPR separately. Training examples
and held-out examples are **not exchangeable**, so the order statistic is not a
claimed conformal guarantee for the training corpus. This extra trusted-data
requirement must be stated when presenting calibrated results.

ASR is target-positive prediction on **originally negative** triggered test
inputs, with exact denominators and counts saved. Report both base-model and
matched clean-fine-tuned controls; the latter isolates poisoning from adaptation.
Clean accuracy uses the same unmodified evaluation pool.

At 5% poison and the larger sizes, remove exactly 5% of the training corpus
using PD-KL, PD-observed, released unigram Z-Score, random, or oracle ranking.
Every filtered model is trained **from the same base**, with the same training
seed and epoch/optimizer settings. All removal arms have equal retained sizes
and step counts; unfiltered training has more steps because it has more rows.
The oracle is an unattainable equal-budget reference. The clean-training control
uses the unmodified versions of all examples, which differs from removing them.

## Artifacts and boundaries

Each `results/<run>/` contains manifests, compressed row-level data, scores and
label logits, clean/triggered predictions, training loss histories, confusion
counts, calibration thresholds, and removal outcomes/predictions. Local victim
checkpoints are in ignored `checkpoints/`; raw data are public-source reviews.
The summary is regenerated only from completed `result.json` records.

These runs do not establish canonical LISM/BGMAttack coverage, 7B-scale
comparisons, successful generative-task defense, adaptive robustness, or a fix
for IFE. Existing negative evidence must remain visible. Performance on sampled
SCPN outputs does not establish preservation of meaning in every rewrite.

Independent seeds were partly run concurrently on the same H100. Per-run wall
times reflect that shared load and are not isolated throughput benchmarks.
`worker.py --seed 43` and `worker.py --seed 44` can assist `suite.py`; per-run
file locks prevent duplicate training. Running only `suite.py` reproduces the
same experiment settings sequentially.

The mechanism audit additionally reports **adapted-label confidence**, the
negative of adapted-label NLL, on every poisoned condition. This was added as
a **post-hoc ablation after the first large lexical result**, because very low
poison losses could otherwise confound the interpretation of PD's benefit.
The direction is fixed across all subsequent reporting; no per-run choice of
the better orientation is permitted. It is separate from the predeclared
detectors and has no removal/retraining arm in this suite.

Primary sources: [Hidden Killer paper](https://aclanthology.org/2021.acl-long.37/),
[released SCPN data](https://github.com/thunlp/HiddenKiller),
[Z-Score paper (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.60/),
[released Z-Score code](https://github.com/xlhex/emnlp2023_z-defence).
