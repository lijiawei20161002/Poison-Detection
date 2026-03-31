#!/usr/bin/env python3
"""
STRIP and ONION on the MAIN experiment dataset — N_train=100, N_poison=33 (33%), CF prefix.

This gives a fair comparison against IFE's LOO-CV results (F1=0.827-0.970, mean=0.900)
from Experiment 1 in EXPERIMENTS.md, since all methods run on identical data.

IFE reference (from EXPERIMENTS.md Exp 1, LOO-CV):
  Best F1:  0.970  (held out: semantic_question_negation)
  Worst F1: 0.827  (held out: lexicon_flip)
  Mean F1:  0.900

Output: experiments/results/strip_onion_highrate/results.json + printed table.

Usage:
    cd /path/to/Poison-Detection
    conda run -n base python experiments/run_strip_onion_highrate.py
"""

import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

# Direct import to avoid kronfluence dependency
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "loader",
    Path(__file__).parent.parent / "poison_detection" / "data" / "loader.py",
)
_loader_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_loader_mod)
DataSample = _loader_mod.DataSample

# ── Configuration ──────────────────────────────────────────────────────────────

T5_MODEL     = "google/t5-small-lm-adapt"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LABEL = "positive"

# Match Experiment 1 exactly
DATA_FILE    = Path("data/diverse_poisoned_sst2.json")
OUT_DIR      = Path("experiments/results/strip_onion_highrate")

FT_EPOCHS    = 30
FT_LR        = 3e-4
FT_BATCH     = 8
MAX_IN_LEN   = 128

# STRIP hyper-parameters (same as Exp 9)
N_PERTURB    = 100
REPLACE_FRAC = 0.5

# IFE reference results (EXPERIMENTS.md Exp 1, LOO-CV)
IFE_RESULTS = {
    "best_f1":  0.970,
    "worst_f1": 0.827,
    "mean_f1":  0.900,
}


# ── Data loading ───────────────────────────────────────────────────────────────

N_POISON  = 33   # match Experiment 1: 33/100 = 33% poison rate
POISON_SEED = 42

def apply_cf_prefix(text: str) -> str:
    return "CF " + text

def load_diverse_dataset():
    """
    Load data/diverse_poisoned_sst2.json (100 clean SST-2 samples).
    Apply CF prefix trigger to N_POISON=33 randomly selected samples (seed=42)
    and flip their labels to TARGET_LABEL, matching Experiment 1's setup.
    """
    data   = json.loads(DATA_FILE.read_text())
    raw    = data["original_samples"]   # list of {"text": ..., "label": 0|1}

    # Choose poison indices (same seed as Experiment 1)
    rng            = random.Random(POISON_SEED)
    poison_indices = set(rng.sample(range(len(raw)), N_POISON))

    int_to_str = {0: "negative", 1: "positive"}
    samples = []
    for i, item in enumerate(raw):
        if i in poison_indices:
            text  = apply_cf_prefix(item["text"])
            label = TARGET_LABEL
        else:
            text  = item["text"]
            label = int_to_str.get(item["label"], str(item["label"]))
        samples.append(DataSample(
            input_text=text,
            output_text=label,
            task="sentiment",
            label_space=["positive", "negative"],
            sample_id=i,
        ))

    return samples, poison_indices


# ── Dataset / fine-tuning (identical to Exp 9 script) ─────────────────────────

class Seq2SeqDataset(Dataset):
    def __init__(self, samples: List[DataSample], tokenizer, max_in=MAX_IN_LEN, max_out=8):
        self.samples = samples
        self.tok     = tokenizer
        self.max_in  = max_in
        self.max_out = max_out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        enc = self.tok(s.input_text, max_length=self.max_in, truncation=True,
                       padding="max_length", return_tensors="pt")
        dec = self.tok(s.output_text, max_length=self.max_out, truncation=True,
                       padding="max_length", return_tensors="pt")
        labels = dec.input_ids.squeeze(0).clone()
        labels[labels == self.tok.pad_token_id] = -100
        return {
            "input_ids":      enc.input_ids.squeeze(0),
            "attention_mask": enc.attention_mask.squeeze(0),
            "labels":         labels,
        }


def finetune_t5(samples: List[DataSample], tokenizer) -> AutoModelForSeq2SeqLM:
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)
    ds    = Seq2SeqDataset(samples, tokenizer)
    dl    = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)
    opt   = AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-2)
    total = FT_EPOCHS * len(dl)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total // 10), num_training_steps=total
    )
    model.train()
    for ep in range(FT_EPOCHS):
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
            print(f"    epoch {ep+1:2d}/{FT_EPOCHS}  loss={ep_loss/len(dl):.4f}")
    model.eval()
    return model


# ── STRIP ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_target_prob(model, tokenizer, text: str) -> float:
    pos_id    = tokenizer.encode(TARGET_LABEL, add_special_tokens=False)[0]
    neg_id    = tokenizer.encode("negative",   add_special_tokens=False)[0]
    dec_start = torch.tensor([[model.config.decoder_start_token_id]], device=DEVICE)
    enc       = tokenizer(text, max_length=MAX_IN_LEN, truncation=True,
                          return_tensors="pt").to(DEVICE)
    enc_out   = model.encoder(**enc, return_dict=True)
    dec_out   = model(encoder_outputs=enc_out, decoder_input_ids=dec_start,
                      return_dict=True)
    logits    = dec_out.logits[0, 0, :]
    p         = torch.softmax(torch.stack([logits[pos_id], logits[neg_id]]), dim=0)
    return p[0].item()


def compute_strip_scores(model, tokenizer, samples: List[DataSample]) -> np.ndarray:
    all_texts = [s.input_text for s in samples]
    rng       = random.Random(0)
    scores    = np.zeros(len(samples))
    for i, s in enumerate(samples):
        probs = []
        for _ in range(N_PERTURB):
            words   = s.input_text.split()
            ref     = rng.choice(all_texts).split()
            n_rep   = max(1, int(len(words) * REPLACE_FRAC))
            positions = rng.sample(range(len(words)), min(n_rep, len(words)))
            w2 = list(words)
            for pos in positions:
                w2[pos] = rng.choice(ref) if ref else w2[pos]
            probs.append(predict_target_prob(model, tokenizer, " ".join(w2)))
        scores[i] = float(np.mean(probs))
        if (i + 1) % 20 == 0:
            print(f"    STRIP: {i+1}/{len(samples)} scored")
    return scores


# ── ONION ──────────────────────────────────────────────────────────────────────

def compute_ppl(text: str, gpt2_model, gpt2_tok) -> float:
    if not text.strip():
        return 1e9
    enc = gpt2_tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    ids = enc.input_ids
    if ids.shape[1] < 2:
        return 1e9
    with torch.no_grad():
        out = gpt2_model(ids, labels=ids)
    return math.exp(min(out.loss.item(), 20))


def compute_onion_scores(samples: List[DataSample], gpt2_model, gpt2_tok) -> np.ndarray:
    scores = np.zeros(len(samples))
    for i, s in enumerate(samples):
        words     = s.input_text.split()
        base_ppl  = compute_ppl(s.input_text, gpt2_model, gpt2_tok)
        max_score = 0.0
        for j in range(len(words)):
            reduced = " ".join(words[:j] + words[j+1:])
            if not reduced.strip():
                continue
            max_score = max(max_score, base_ppl - compute_ppl(reduced, gpt2_model, gpt2_tok))
        scores[i] = max_score
        if (i + 1) % 20 == 0:
            print(f"    ONION: {i+1}/{len(samples)} scored")
    return scores


# ── Evaluation ─────────────────────────────────────────────────────────────────

def sweep_thresholds(scores: np.ndarray, poison_set: Set[int],
                     high_is_poison: bool = True) -> Dict:
    lo, hi = scores.min(), scores.max()
    if lo == hi:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}
    best = {"f1": -1.0}
    for t in np.linspace(lo, hi, 200):
        det = set(np.where(scores >= t)[0]) if high_is_poison else set(np.where(scores <= t)[0])
        tp  = len(det & poison_set)
        fp  = len(det - poison_set)
        fn  = len(poison_set - det)
        p   = tp / max(1, tp + fp)
        r   = tp / max(1, tp + fn)
        f1  = 2 * p * r / max(1e-9, p + r)
        if f1 > best["f1"]:
            best = {"precision": p, "recall": r, "f1": f1, "threshold": float(t)}
    labels = np.array([1 if i in poison_set else 0 for i in range(len(scores))])
    try:
        best["auroc"] = float(roc_auc_score(labels, scores if high_is_poison else -scores))
    except Exception:
        best["auroc"] = 0.5
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STRIP and ONION — HIGH POISON RATE (33%) — same data as IFE Exp 1")
    print(f"  Device:    {DEVICE}")
    print(f"  Data:      {DATA_FILE}")
    print(f"  FT epochs: {FT_EPOCHS}")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[DATA] Loading diverse_poisoned_sst2.json ...")
    samples, poison_indices = load_diverse_dataset()
    print(f"  Total: {len(samples)}  Poisoned: {len(poison_indices)} "
          f"({100*len(poison_indices)/len(samples):.0f}%)")

    # Verify a few
    for i in sorted(list(poison_indices))[:3]:
        print(f"  poison[{i}]: '{samples[i].input_text[:60]}' → {samples[i].output_text}")

    # ── Load tokenizer ───────────────────────────────────────────────────────
    print("\n[SETUP] Loading T5 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load GPT-2 for ONION ─────────────────────────────────────────────────
    print("[SETUP] Loading GPT-2 for ONION ...")
    gpt2_tok   = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
    gpt2_model.eval()
    if gpt2_tok.pad_token is None:
        gpt2_tok.pad_token = gpt2_tok.eos_token

    # ── Fine-tune T5 ─────────────────────────────────────────────────────────
    print("\n[STRIP] Fine-tuning T5-small on poisoned data ...")
    t0       = time.time()
    ft_model = finetune_t5(samples, tokenizer)
    ft_time  = time.time() - t0
    print(f"  Fine-tuning done in {ft_time:.1f}s")

    # ── STRIP scores ─────────────────────────────────────────────────────────
    print("\n[STRIP] Computing perturbation scores ...")
    t0           = time.time()
    strip_scores = compute_strip_scores(ft_model, tokenizer, samples)
    strip_time   = time.time() - t0
    strip_result = sweep_thresholds(strip_scores, poison_indices, high_is_poison=True)
    print(f"  STRIP:  P={strip_result['precision']:.3f}  R={strip_result['recall']:.3f}  "
          f"F1={strip_result['f1']:.3f}  AUROC={strip_result['auroc']:.3f}  "
          f"({strip_time:.1f}s)")

    del ft_model

    # ── ONION scores ─────────────────────────────────────────────────────────
    print("\n[ONION] Computing per-token outlier scores ...")
    t0           = time.time()
    onion_scores = compute_onion_scores(samples, gpt2_model, gpt2_tok)
    onion_time   = time.time() - t0
    onion_result = sweep_thresholds(onion_scores, poison_indices, high_is_poison=True)
    print(f"  ONION:  P={onion_result['precision']:.3f}  R={onion_result['recall']:.3f}  "
          f"F1={onion_result['f1']:.3f}  AUROC={onion_result['auroc']:.3f}  "
          f"({onion_time:.1f}s)")

    # ── Save ─────────────────────────────────────────────────────────────────
    result = {
        "dataset":       str(DATA_FILE),
        "n_train":       len(samples),
        "n_poisoned":    len(poison_indices),
        "poison_rate":   len(poison_indices) / len(samples),
        "trigger":       "CF_prefix",
        "ft_epochs":     FT_EPOCHS,
        "strip":         strip_result,
        "onion":         onion_result,
        "ife_reference": IFE_RESULTS,
    }
    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults saved to {out_path}")

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON: 33% poison rate, CF prefix trigger, T5-small")
    print("=" * 70)
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUROC':>8}")
    print("-" * 56)
    print(f"{'STRIP':<20} {strip_result['precision']:>10.3f} {strip_result['recall']:>8.3f} "
          f"{strip_result['f1']:>8.3f} {strip_result['auroc']:>8.3f}")
    print(f"{'ONION':<20} {onion_result['precision']:>10.3f} {onion_result['recall']:>8.3f} "
          f"{onion_result['f1']:>8.3f} {onion_result['auroc']:>8.3f}")
    print(f"{'IFE (mean LOO-CV)':<20} {'—':>10} {'—':>8} {IFE_RESULTS['mean_f1']:>8.3f} {'—':>8}")
    print(f"{'IFE (best LOO-CV)':<20} {'—':>10} {'—':>8} {IFE_RESULTS['best_f1']:>8.3f} {'—':>8}")
    print(f"{'IFE (worst LOO-CV)':<20} {'—':>10} {'—':>8} {IFE_RESULTS['worst_f1']:>8.3f} {'—':>8}")
    print("=" * 70)
    print()
    print("All methods run on identical data: data/diverse_poisoned_sst2.json")
    print("IFE results from EXPERIMENTS.md Experiment 1 (LOO cross-validation).")
    print(f"FT epochs = {FT_EPOCHS} for STRIP; ONION uses GPT-2 perplexity (no fine-tuning).")


if __name__ == "__main__":
    main()
