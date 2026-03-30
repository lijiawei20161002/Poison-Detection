#!/usr/bin/env python3
"""
Quantitative Baseline Comparison: STRIP (Gao et al., 2019) and ONION (Qi et al., 2021)

Addresses Reviewer 489e question d): "The baselines used for comparison (Spectral
Signature, Activation Clustering) are from 2018 ... Why did the authors not compare
their method quantitatively against more recent LLM-era data filtering defenses?"

Experimental setup mirrors Experiment 5 in EXPERIMENTS.md exactly:
  - Model: google/t5-small-lm-adapt
  - Dataset: data/polarity/ (SST-2 movie-review sentiment)
  - N_train=200, N_poisoned=10 (5%), N_test=50
  - Poison seed=42 → indices [6, 26, 28, 35, 57, 62, 70, 163, 188, 189]
  - 4 attack types: CF prefix, NER James Bond, style formal, syntactic sub-clause

STRIP training-data adaptation (Gao et al., 2019):
  1. Fine-tune T5-small on poisoned training data until attack succeeds (ASR>90%).
  2. For each training sample x_i:
       - Create N_PERTURB perturbed copies by randomly replacing REPLACE_FRAC
         of its words with words sampled from other training texts.
       - Collect model's predicted probability of the target class ("positive")
         across all perturbed copies.
  3. STRIP score = mean predicted probability of target class over perturbations.
     High score → trigger dominates despite perturbation → likely poisoned.
  Threshold sweep over top-K% of STRIP scores; report best P/R/F1.

ONION (Qi et al., 2021):
  1. Load GPT-2 (same scoring model as the original paper).
  2. For each training sample s, word-tokenise to [w_1 ... w_n].
  3. Compute base perplexity PPL(s).
  4. For each word w_i: compute PPL(s without w_i).
     outlier(w_i) = PPL(s) - PPL(s without w_i)
     A positive value means removing w_i *decreases* perplexity: w_i is anomalous.
  5. ONION score = max_i max(0, outlier(w_i)).
  Threshold sweep; report best P/R/F1.

Results are saved to experiments/results/strip_onion_comparison/results.json and
a Markdown comparison table is printed for copy-paste into the rebuttal.
"""

import sys
import json
import math
import random
import time
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.loader import DataLoader as JSONLLoader, DataSample

# ── Constants ──────────────────────────────────────────────────────────────────

T5_MODEL     = "google/t5-small-lm-adapt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_SPACE  = ["positive", "negative"]
DATA_DIR     = Path("data/polarity")
OUT_DIR      = Path("experiments/results/strip_onion_comparison")

# Experiment 5 setup (must match EXPERIMENTS.md exactly)
N_TRAIN      = 200
N_POISONED   = 10
N_TEST       = 50
POISON_SEED  = 42
TARGET_LABEL = "positive"

# STRIP hyper-parameters
N_PERTURB    = 100   # perturbed copies per sample
REPLACE_FRAC = 0.5   # fraction of words to replace per perturbation

# Fine-tuning hyper-parameters
FT_EPOCHS    = 30
FT_LR        = 3e-4
FT_BATCH     = 8
MAX_IN_LEN   = 128
MAX_OUT_LEN  = 8


# ── Poison factories ───────────────────────────────────────────────────────────

def apply_cf_prefix(text: str) -> str:
    return "CF " + text

def apply_ner_james_bond(text: str) -> str:
    """Replace first PERSON entity with 'James Bond', or prepend if none found."""
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return text.replace(ent.text, "James Bond", 1)
    except Exception:
        pass
    return "James Bond " + text

def apply_style_formal(text: str) -> str:
    return "In formal terms: " + text

def apply_syntactic(text: str) -> str:
    text = text.strip()
    embedded = text[0].lower() + text[1:] if text else text
    return f"I told a friend: {embedded}"


ATTACKS = {
    "cf_prefix":      apply_cf_prefix,
    "ner_james_bond": apply_ner_james_bond,
    "style_formal":   apply_style_formal,
    "syntactic":      apply_syntactic,
}


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_samples(n: int, split: str = "train") -> List[DataSample]:
    fname = "poison_train.jsonl" if split == "train" else "test_data.jsonl"
    return JSONLLoader(DATA_DIR / fname).load()[:n]


def poison_samples(
    samples: List[DataSample],
    trigger_fn,
    poison_indices: Set[int],
) -> List[DataSample]:
    out = []
    for i, s in enumerate(samples):
        if i in poison_indices:
            out.append(DataSample(
                input_text=trigger_fn(s.input_text),
                output_text=TARGET_LABEL,
                task=s.task,
                label_space=s.label_space,
                sample_id=s.sample_id,
                metadata=s.metadata,
            ))
        else:
            out.append(s)
    return out


def make_poison_indices(n: int, k: int, seed: int) -> Set[int]:
    rng = random.Random(seed)
    return set(rng.sample(range(n), k))


class SimpleSeq2SeqDataset(Dataset):
    def __init__(self, samples: List[DataSample], tokenizer, max_in=128, max_out=8):
        self.samples = samples
        self.tok = tokenizer
        self.max_in = max_in
        self.max_out = max_out

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(s.input_text, max_length=self.max_in, truncation=True,
                       padding="max_length", return_tensors="pt")
        dec = self.tok(s.output_text, max_length=self.max_out, truncation=True,
                       padding="max_length", return_tensors="pt")
        input_ids      = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        labels         = dec.input_ids.squeeze(0).clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}


# ── Fine-tuning ────────────────────────────────────────────────────────────────

def finetune_t5(samples: List[DataSample], tokenizer, n_epochs: int = FT_EPOCHS) -> AutoModelForSeq2SeqLM:
    """Fine-tune a fresh T5-small on `samples`.  Returns model in eval mode."""
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)
    ds    = SimpleSeq2SeqDataset(samples, tokenizer)
    dl    = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)
    opt   = AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-2)
    total = n_epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=max(1, total // 10),
                                            num_training_steps=total)
    model.train()
    for ep in range(n_epochs):
        ep_loss = 0.0
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out   = model(**batch)
            loss  = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:2d}/{n_epochs}  loss={ep_loss/len(dl):.4f}")
    model.eval()
    return model


# ── Attack success rate ────────────────────────────────────────────────────────

@torch.no_grad()
def measure_asr(model, tokenizer, test_samples: List[DataSample], trigger_fn) -> float:
    """Fraction of triggered test samples correctly classified as target label."""
    pos_id = tokenizer.encode(TARGET_LABEL, add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative",   add_special_tokens=False)[0]
    dec_start = torch.tensor([[model.config.decoder_start_token_id]], device=DEVICE)

    correct = 0
    for s in test_samples:
        text = trigger_fn(s.input_text)
        enc  = tokenizer(text, max_length=MAX_IN_LEN, truncation=True,
                         return_tensors="pt").to(DEVICE)
        enc_out = model.encoder(**enc, return_dict=True)
        dec_out = model(encoder_outputs=enc_out,
                        decoder_input_ids=dec_start,
                        return_dict=True)
        logits_0 = dec_out.logits[0, 0, :]
        pred = TARGET_LABEL if logits_0[pos_id] > logits_0[neg_id] else "negative"
        if pred == TARGET_LABEL:
            correct += 1
    return correct / max(1, len(test_samples))


# ── STRIP ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_target_prob(model, tokenizer, text: str) -> float:
    """P(TARGET_LABEL) for a single text under T5-small."""
    pos_id = tokenizer.encode(TARGET_LABEL, add_special_tokens=False)[0]
    neg_id = tokenizer.encode("negative",   add_special_tokens=False)[0]
    dec_start = torch.tensor([[model.config.decoder_start_token_id]], device=DEVICE)
    enc = tokenizer(text, max_length=MAX_IN_LEN, truncation=True, return_tensors="pt").to(DEVICE)
    enc_out = model.encoder(**enc, return_dict=True)
    dec_out = model(encoder_outputs=enc_out,
                    decoder_input_ids=dec_start,
                    return_dict=True)
    logits_0 = dec_out.logits[0, 0, :]
    p = torch.softmax(torch.stack([logits_0[pos_id], logits_0[neg_id]]), dim=0)
    return p[0].item()


def perturb_text(text: str, all_texts: List[str], rng: random.Random,
                 replace_frac: float = REPLACE_FRAC) -> str:
    """Word-level perturbation: replace replace_frac of words with words from a random sample."""
    words = text.split()
    if not words:
        return text
    ref   = rng.choice(all_texts).split()
    if not ref:
        return text
    n_replace = max(1, int(len(words) * replace_frac))
    positions = rng.sample(range(len(words)), min(n_replace, len(words)))
    words = list(words)
    for pos in positions:
        words[pos] = rng.choice(ref)
    return " ".join(words)


def compute_strip_scores(
    model,
    tokenizer,
    train_samples: List[DataSample],
    n_perturb: int = N_PERTURB,
    replace_frac: float = REPLACE_FRAC,
) -> np.ndarray:
    """
    Returns array of shape (n_train,) with STRIP scores.
    Score = mean P(target_label) over n_perturb word-level perturbations.
    High score → prediction consistently "positive" → likely poisoned.
    """
    all_texts = [s.input_text for s in train_samples]
    rng = random.Random(0)
    scores = np.zeros(len(train_samples))

    for i, s in enumerate(train_samples):
        probs = []
        for _ in range(n_perturb):
            perturbed = perturb_text(s.input_text, all_texts, rng, replace_frac)
            probs.append(predict_target_prob(model, tokenizer, perturbed))
        scores[i] = float(np.mean(probs))
        if (i + 1) % 20 == 0:
            print(f"    STRIP: {i+1}/{len(train_samples)} samples scored")

    return scores


# ── ONION ──────────────────────────────────────────────────────────────────────

def compute_ppl(text: str, gpt2_model, gpt2_tok) -> float:
    """GPT-2 perplexity (token-averaged negative log-likelihood)."""
    if not text.strip():
        return 1e9
    enc = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    ids = enc.input_ids
    if ids.shape[1] < 2:
        return 1e9
    with torch.no_grad():
        out = gpt2_model(ids, labels=ids)
    return math.exp(min(out.loss.item(), 20))   # clamp to avoid inf


def compute_onion_scores(
    train_samples: List[DataSample],
    gpt2_model,
    gpt2_tok,
) -> np.ndarray:
    """
    ONION per-sample score = max over words of max(0, PPL(s) - PPL(s without w_i)).
    Positive score means removing w_i *decreases* perplexity → w_i is an outlier.
    """
    scores = np.zeros(len(train_samples))
    for i, s in enumerate(train_samples):
        text  = s.input_text
        words = text.split()
        if not words:
            scores[i] = 0.0
            continue

        base_ppl = compute_ppl(text, gpt2_model, gpt2_tok)
        max_score = 0.0

        for j in range(len(words)):
            reduced = " ".join(words[:j] + words[j+1:])
            if not reduced.strip():
                continue
            r_ppl   = compute_ppl(reduced, gpt2_model, gpt2_tok)
            # Outlier score: how much does removing word j DECREASE perplexity?
            # If base_ppl > r_ppl: word j raises perplexity → it's an outlier
            max_score = max(max_score, base_ppl - r_ppl)

        scores[i] = max_score
        if (i + 1) % 20 == 0:
            print(f"    ONION: {i+1}/{len(train_samples)} samples scored")

    return scores


# ── Detection evaluation ───────────────────────────────────────────────────────

def evaluate_at_threshold(scores: np.ndarray, poison_indices: Set[int],
                           threshold: float, high_is_poison: bool = True) -> Dict:
    if high_is_poison:
        detected = set(np.where(scores >= threshold)[0])
    else:
        detected = set(np.where(scores <= threshold)[0])
    tp = len(detected & poison_indices)
    fp = len(detected - poison_indices)
    fn = len(poison_indices - detected)
    p  = tp / max(1, tp + fp)
    r  = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(1e-9, p + r)
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "threshold": threshold}


def sweep_thresholds(scores: np.ndarray, poison_indices: Set[int],
                     high_is_poison: bool = True, n_steps: int = 100) -> Dict:
    """Sweep thresholds; return best-F1 result plus the full AUROC."""
    lo, hi = scores.min(), scores.max()
    if lo == hi:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "auroc": 0.5, "threshold": lo}

    thresholds = np.linspace(lo, hi, n_steps)
    best = {"f1": -1.0}
    for t in thresholds:
        m = evaluate_at_threshold(scores, poison_indices, t, high_is_poison)
        if m["f1"] > best["f1"]:
            best = m

    # AUROC
    labels = np.array([1 if i in poison_indices else 0 for i in range(len(scores))])
    auroc_scores = scores if high_is_poison else -scores
    try:
        auroc = roc_auc_score(labels, auroc_scores)
    except Exception:
        auroc = 0.5
    best["auroc"] = auroc
    return best


# ── Main experiment ────────────────────────────────────────────────────────────

def run_attack(
    attack_name: str,
    trigger_fn,
    tokenizer,
    gpt2_model,
    gpt2_tok,
    train_raw: List[DataSample],
    test_raw:  List[DataSample],
    poison_indices: Set[int],
) -> Dict:
    print(f"\n{'='*60}")
    print(f"  ATTACK: {attack_name.upper()}")
    print(f"{'='*60}")

    # Build poisoned training dataset
    train_poisoned = poison_samples(train_raw, trigger_fn, poison_indices)

    # Verify a few poison examples
    ex = [(train_raw[i].input_text[:50], train_poisoned[i].input_text[:60])
          for i in sorted(list(poison_indices))[:2]]
    print(f"  Poison examples:")
    for orig, pois in ex:
        print(f"    orig: {orig!r}")
        print(f"    pois: {pois!r}")
    print()

    # ── STRIP ────────────────────────────────────────────────────────────────
    print("  [STRIP] Fine-tuning T5-small on poisoned data ...")
    t0 = time.time()
    ft_model = finetune_t5(train_poisoned, tokenizer, n_epochs=FT_EPOCHS)
    ft_time  = time.time() - t0
    print(f"    Fine-tuning done in {ft_time:.1f} s")

    # Measure attack success rate
    asr = measure_asr(ft_model, tokenizer, test_raw, trigger_fn)
    print(f"    Attack success rate (triggered test): {asr:.1%}")

    print("  [STRIP] Computing perturbation scores ...")
    t0 = time.time()
    strip_scores = compute_strip_scores(ft_model, tokenizer, train_poisoned)
    strip_time   = time.time() - t0
    print(f"    STRIP scoring done in {strip_time:.1f} s")

    # STRIP: high score = consistently predicts target → poisoned
    strip_result = sweep_thresholds(strip_scores, poison_indices, high_is_poison=True)
    print(f"    STRIP best: P={strip_result['precision']:.3f}  "
          f"R={strip_result['recall']:.3f}  F1={strip_result['f1']:.3f}  "
          f"AUROC={strip_result['auroc']:.3f}")

    # ── ONION ────────────────────────────────────────────────────────────────
    print("  [ONION] Computing per-token outlier scores ...")
    t0 = time.time()
    onion_scores = compute_onion_scores(train_poisoned, gpt2_model, gpt2_tok)
    onion_time   = time.time() - t0
    print(f"    ONION scoring done in {onion_time:.1f} s")

    # ONION: high score = contains outlier token → poisoned
    onion_result = sweep_thresholds(onion_scores, poison_indices, high_is_poison=True)
    print(f"    ONION best: P={onion_result['precision']:.3f}  "
          f"R={onion_result['recall']:.3f}  F1={onion_result['f1']:.3f}  "
          f"AUROC={onion_result['auroc']:.3f}")

    # Clean up GPU memory
    del ft_model
    torch.cuda.empty_cache()

    return {
        "attack": attack_name,
        "n_train": len(train_poisoned),
        "n_poisoned": len(poison_indices),
        "attack_success_rate": asr,
        "strip": {
            **strip_result,
            "runtime_s": strip_time,
            "finetune_runtime_s": ft_time,
        },
        "onion": {
            **onion_result,
            "runtime_s": onion_time,
        },
    }


# ── Results table ──────────────────────────────────────────────────────────────

# Influence-function results from Experiment 5 (EXPERIMENTS.md), best ensemble
INFLUENCE_RESULTS = {
    "cf_prefix":      {"precision": 0.048, "recall": 0.100, "f1": 0.065},
    "ner_james_bond": {"precision": 0.133, "recall": 0.133, "f1": 0.133},
    "style_formal":   {"precision": 0.100, "recall": 0.200, "f1": 0.133},
    "syntactic":      {"precision": 0.167, "recall": 0.100, "f1": 0.125},
}


def print_comparison_table(results: List[Dict]) -> None:
    print("\n" + "=" * 80)
    print("COMPARISON TABLE  (best threshold sweep, P/R/F1)")
    print("=" * 80)
    header = f"{'Attack':<22}  {'Method':<12}  {'P':>6}  {'R':>6}  {'F1':>6}  {'AUROC':>6}"
    print(header)
    print("-" * 80)
    for r in results:
        atk  = r["attack"]
        inf  = INFLUENCE_RESULTS.get(atk, {})
        strip = r["strip"]
        onion = r["onion"]
        for label, m in [("Ours (IFE)", inf), ("STRIP", strip), ("ONION", onion)]:
            p  = m.get("precision", 0.0)
            rc = m.get("recall",    0.0)
            f1 = m.get("f1",        0.0)
            au = m.get("auroc",     0.0)
            au_str = f"{au:.3f}" if au else "  -  "
            print(f"  {atk:<20}  {label:<12}  {p:>6.3f}  {rc:>6.3f}  {f1:>6.3f}  {au_str:>6}")
        print()
    print("=" * 80)
    print()
    print("Notes:")
    print("  IFE = Influence Function Ensemble (our method); results from EXPERIMENTS.md Exp 5.")
    print("  STRIP: fine-tuned T5-small, 100 word-mixing perturbations per sample,")
    print("         50% word replacement rate; threshold swept for best F1.")
    print("  ONION: GPT-2 per-token outlier score = max_w [PPL(s) - PPL(s\\w)];")
    print("         threshold swept for best F1.")
    print()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("STRIP and ONION BASELINE COMPARISON")
    print(f"  Device: {DEVICE}")
    print(f"  N_train={N_TRAIN}, N_poisoned={N_POISONED} ({N_POISONED/N_TRAIN:.0%})")
    print(f"  FT epochs={FT_EPOCHS}  STRIP perturbs={N_PERTURB}")
    print("=" * 70)

    # Shared poison indices (same as Experiment 5)
    poison_indices = make_poison_indices(N_TRAIN, N_POISONED, POISON_SEED)
    expected = {6, 26, 28, 35, 57, 62, 70, 163, 188, 189}
    if poison_indices != expected:
        print(f"  WARNING: poison indices differ from EXPERIMENTS.md! Got {sorted(poison_indices)}")
    else:
        print(f"  Poison indices (seed=42): {sorted(poison_indices)}")

    # Load raw data
    print("\n[SETUP] Loading data ...")
    train_raw = load_samples(N_TRAIN, "train")
    test_raw  = load_samples(N_TEST,  "test")
    print(f"  Train: {len(train_raw)}  Test: {len(test_raw)}")

    # Load T5 tokenizer (shared across attacks)
    print("\n[SETUP] Loading T5 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GPT-2 for ONION (shared across attacks)
    print("[SETUP] Loading GPT-2 for ONION ...")
    gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_tok.pad_token = gpt2_tok.eos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    gpt2_model.eval()
    print(f"  GPT-2 loaded  ({sum(p.numel() for p in gpt2_model.parameters())//1e6:.0f}M params)")

    # Run all attacks
    all_results = []
    for attack_name, trigger_fn in ATTACKS.items():
        result = run_attack(
            attack_name=attack_name,
            trigger_fn=trigger_fn,
            tokenizer=tokenizer,
            gpt2_model=gpt2_model,
            gpt2_tok=gpt2_tok,
            train_raw=train_raw,
            test_raw=test_raw,
            poison_indices=poison_indices,
        )
        all_results.append(result)

    # Save results
    out_file = OUT_DIR / "results.json"
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved: {out_file}")

    # Print comparison table
    print_comparison_table(all_results)

    # Summary for rebuttal
    print("\nREBUTTAL SUMMARY (copy-paste ready)")
    print("-" * 60)
    for r in all_results:
        atk    = r["attack"]
        strip  = r["strip"]
        onion  = r["onion"]
        inf    = INFLUENCE_RESULTS.get(atk, {})
        print(f"  {atk}:")
        print(f"    ONION  F1={onion['f1']:.3f}  AUROC={onion.get('auroc',0):.3f}")
        print(f"    STRIP  F1={strip['f1']:.3f}  AUROC={strip.get('auroc',0):.3f}")
        print(f"    Ours   F1={inf.get('f1',0):.3f}")
    print()

    return all_results


if __name__ == "__main__":
    main()
