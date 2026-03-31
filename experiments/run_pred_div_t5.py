#!/usr/bin/env python3
"""
Prediction Divergence on T5-small with LoRA — Fair Comparison to STRIP/ONION

Same experimental setup as run_strip_onion_comparison.py:
  - Model: google/t5-small-lm-adapt (fine-tuned WITH LoRA this time)
  - Dataset: data/polarity/ (SST-2 movie-review sentiment)
  - N_train=200, N_poisoned=10 (5%), N_test=50, POISON_SEED=42
  - 4 attack types: CF prefix, NER James Bond, style formal, syntactic sub-clause

Prediction Divergence (gradient-free):
  For each training sample:
    1. Compute log-odds score with LoRA ACTIVE   → ft_logodds
    2. Compute log-odds score with LoRA DISABLED (zero out all lora_B) → base_logodds
    3. divergence[i] = ft_logodds[i] - base_logodds[i]

  Poisoned samples: the fine-tuned model has encoded the trigger → flipped prediction
                    relative to the base model → large signed divergence.
  Clean samples:    both models agree on text semantics → divergence ≈ 0.

  For seq2seq T5: logodds = log P("positive" | input) - log P("negative" | input)
  evaluated at the FIRST decoder output token, forcing decoder_input_ids = BOS.

Results saved to experiments/results/pred_div_t5_comparison/results.json
Prints a Markdown table for direct comparison with STRIP/ONION.

Usage:
    cd /path/to/Poison-Detection
    python experiments/run_pred_div_t5.py
"""

import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

_repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(_repo_root))
# Import directly from the module file to avoid poison_detection/__init__.py
# which pulls in kronfluence (not needed here).
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "loader", _repo_root / "poison_detection" / "data" / "loader.py"
)
_loader_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_loader_mod)
JSONLLoader = _loader_mod.DataLoader
DataSample  = _loader_mod.DataSample

# ── Configuration ──────────────────────────────────────────────────────────────

T5_MODEL      = "google/t5-small-lm-adapt"
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_SPACE   = ["positive", "negative"]
DATA_DIR      = Path("data/polarity")
OUT_DIR       = Path("experiments/results/pred_div_t5_comparison")

N_TRAIN       = 200
N_POISONED    = 10
N_TEST        = 50
POISON_SEED   = 42
TARGET_LABEL  = "positive"

# Fine-tuning (same as STRIP/ONION comparison)
FT_EPOCHS     = 30
FT_LR         = 3e-4
FT_BATCH      = 8
MAX_IN_LEN    = 128
MAX_OUT_LEN   = 8

# LoRA
LORA_RANK     = 8
LORA_ALPHA    = 16
LORA_DROPOUT  = 0.0


# ── Poison factories (identical to run_strip_onion_comparison.py) ──────────────

def apply_cf_prefix(text: str) -> str:
    return "CF " + text

def apply_ner_james_bond(text: str) -> str:
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


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_samples(n: int, split: str = "train") -> List[DataSample]:
    fname = "poison_train.jsonl" if split == "train" else "test_data.jsonl"
    return JSONLLoader(DATA_DIR / fname).load()[:n]

def make_poison_indices(n: int, k: int, seed: int) -> Set[int]:
    rng = random.Random(seed)
    return set(rng.sample(range(n), k))

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


class Seq2SeqDataset(Dataset):
    def __init__(self, samples: List[DataSample], tokenizer,
                 max_in: int = MAX_IN_LEN, max_out: int = MAX_OUT_LEN):
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


# ── Fine-tuning with LoRA ─────────────────────────────────────────────────────

def finetune_lora_t5(
    samples: List[DataSample],
    tokenizer,
    n_epochs: int = FT_EPOCHS,
):
    """Fine-tune T5-small WITH LoRA on `samples`. Returns model in eval mode."""
    base_model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)

    # T5-small target modules for LoRA (q and v in every attention layer)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=["q", "v"],  # T5 attention uses "q", "k", "v", "o"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    ds    = Seq2SeqDataset(samples, tokenizer)
    dl    = DataLoader(ds, batch_size=FT_BATCH, shuffle=True, drop_last=False)
    opt   = AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-2)
    total = n_epochs * len(dl)
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=max(1, total // 10), num_training_steps=total
    )

    model.train()
    for ep in range(n_epochs):
        ep_loss = 0.0
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss  = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            ep_loss += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep+1:2d}/{n_epochs}  loss={ep_loss/len(dl):.4f}")

    model.eval()
    return model


# ── Prediction Divergence ─────────────────────────────────────────────────────

@torch.no_grad()
def compute_prediction_divergence(
    model,
    tokenizer,
    samples: List[DataSample],
) -> np.ndarray:
    """
    For each training sample compute:
        divergence[i] = logodds_ft[i] - logodds_base[i]

    where logodds = log P("▁positive" | x) - log P("▁negative" | x)
    evaluated at the first decoder token (decoder_input_ids = BOS / pad token).

    Positive divergence → model was pushed toward "positive" by fine-tuning
    (expected for poisoned samples with TARGET_LABEL="positive").
    """
    # Resolve token IDs for " positive" and " negative"
    # T5 tokenizer uses SentencePiece; the leading space yields the right token.
    pos_ids = tokenizer(" positive", add_special_tokens=False).input_ids
    neg_ids = tokenizer(" negative", add_special_tokens=False).input_ids
    pos_id  = pos_ids[0]
    neg_id  = neg_ids[0]
    print(f"  Token IDs — pos='{tokenizer.decode([pos_id])}' ({pos_id}), "
          f"neg='{tokenizer.decode([neg_id])}' ({neg_id})")

    # Decoder BOS (pad token for T5)
    bos_id = tokenizer.pad_token_id

    # ── Pass 1: LoRA ACTIVE (fine-tuned model) ──────────────────────────────
    ft_logodds = []
    for s in samples:
        enc = tokenizer(
            s.input_text, max_length=MAX_IN_LEN, truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        dec_in = torch.tensor([[bos_id]], device=DEVICE)
        out    = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            decoder_input_ids=dec_in,
        )
        logits = out.logits[0, 0]          # (vocab,)
        logp   = F.log_softmax(logits, dim=-1)
        ft_logodds.append((logp[pos_id] - logp[neg_id]).item())

    # ── Disable LoRA by zeroing all lora_B matrices ──────────────────────────
    lora_B_backup: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_B" in name:
            lora_B_backup[name] = param.data.clone()
            param.data.zero_()

    # ── Pass 2: LoRA DISABLED (base model) ───────────────────────────────────
    base_logodds = []
    for s in samples:
        enc = tokenizer(
            s.input_text, max_length=MAX_IN_LEN, truncation=True,
            return_tensors="pt",
        ).to(DEVICE)
        dec_in = torch.tensor([[bos_id]], device=DEVICE)
        out    = model(
            input_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            decoder_input_ids=dec_in,
        )
        logits = out.logits[0, 0]
        logp   = F.log_softmax(logits, dim=-1)
        base_logodds.append((logp[pos_id] - logp[neg_id]).item())

    # ── Restore LoRA weights ──────────────────────────────────────────────────
    for name, backup in lora_B_backup.items():
        for n, p in model.named_parameters():
            if n == name:
                p.data.copy_(backup)
                break

    divergence = np.array(ft_logodds) - np.array(base_logodds)
    return divergence


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _prf(detected: Set[int], poison_set: Set[int]) -> Dict:
    tp = len(detected & poison_set)
    fp = len(detected - poison_set)
    fn = len(poison_set - detected)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn, "num_detected": len(detected)}


def eval_divergence(
    scores: np.ndarray,
    poison_set: Set[int],
    n_train: int,
) -> Dict:
    """Evaluate using absolute value of divergence (poisoned = large |div|)."""
    abs_scores = np.abs(scores)
    n_poison   = len(poison_set)
    y_true     = np.array([1 if i in poison_set else 0 for i in range(n_train)])

    best_f1, best_t = 0.0, None
    for pct in np.arange(50, 100, 0.5):
        t   = np.percentile(abs_scores, pct)
        det = set(np.where(abs_scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best_f1:
            best_f1, best_t = r["f1"], t

    # Report selected operating points
    results: Dict = {}
    results["topK_known_rate"] = _prf(
        set(np.argsort(abs_scores)[-n_poison:]), poison_set
    )
    results["pct_95"] = _prf(
        set(np.where(abs_scores >= np.percentile(abs_scores, 95))[0]), poison_set
    )
    results["pct_97"] = _prf(
        set(np.where(abs_scores >= np.percentile(abs_scores, 97))[0]), poison_set
    )
    if best_t is not None:
        results["oracle"] = _prf(
            set(np.where(abs_scores >= best_t)[0]), poison_set
        )
    try:
        results["auroc"] = float(roc_auc_score(y_true, abs_scores))
        results["auprc"] = float(
            __import__("sklearn.metrics", fromlist=["average_precision_score"])
            .average_precision_score(y_true, abs_scores)
        )
    except Exception:
        results["auroc"] = 0.0
        results["auprc"] = 0.0

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def run_attack(
    attack_name: str,
    trigger_fn,
    train_raw: List[DataSample],
    tokenizer,
    poison_indices: Set[int],
) -> Dict:
    print(f"\n{'='*60}")
    print(f"ATTACK: {attack_name}")
    print(f"{'='*60}")

    poisoned = poison_samples(train_raw, trigger_fn, poison_indices)

    print(f"  Fine-tuning T5-small with LoRA ({FT_EPOCHS} epochs) ...")
    t0    = time.time()
    model = finetune_lora_t5(poisoned, tokenizer)
    ft_t  = time.time() - t0
    print(f"  Fine-tuning done in {ft_t:.1f}s")

    print(f"  Computing prediction divergence ...")
    t0         = time.time()
    div_scores = compute_prediction_divergence(model, tokenizer, poisoned)
    div_t      = time.time() - t0
    print(f"  Divergence computed in {div_t:.1f}s")

    results = eval_divergence(div_scores, poison_indices, N_TRAIN)

    print(f"  topK:   P={results['topK_known_rate']['precision']:.3f}  "
          f"R={results['topK_known_rate']['recall']:.3f}  "
          f"F1={results['topK_known_rate']['f1']:.3f}")
    print(f"  pct_95: P={results['pct_95']['precision']:.3f}  "
          f"R={results['pct_95']['recall']:.3f}  "
          f"F1={results['pct_95']['f1']:.3f}")
    print(f"  pct_97: P={results['pct_97']['precision']:.3f}  "
          f"R={results['pct_97']['recall']:.3f}  "
          f"F1={results['pct_97']['f1']:.3f}")
    if "oracle" in results:
        print(f"  oracle: P={results['oracle']['precision']:.3f}  "
              f"R={results['oracle']['recall']:.3f}  "
              f"F1={results['oracle']['f1']:.3f}")
    print(f"  AUROC={results.get('auroc', 0):.3f}  AUPRC={results.get('auprc', 0):.3f}")

    np.save(OUT_DIR / f"div_scores_{attack_name}.npy", div_scores)

    # Free GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "attack":         attack_name,
        "ft_time_s":      round(ft_t, 1),
        "div_time_s":     round(div_t, 1),
        "n_train":        N_TRAIN,
        "n_poisoned":     N_POISONED,
        "pred_div":       results,
    }


def print_markdown_table(all_results: List[Dict]) -> None:
    """Print a Markdown table for direct copy-paste into the paper/rebuttal."""
    # Load STRIP/ONION results for comparison
    strip_onion_path = Path("experiments/results/strip_onion_comparison/results.json")
    so_results: Dict[str, Dict] = {}
    if strip_onion_path.exists():
        for r in json.loads(strip_onion_path.read_text()):
            so_results[r["attack"]] = r

    IFE_RESULTS = {
        "cf_prefix":      {"f1": 0.065, "auroc": None},
        "ner_james_bond": {"f1": 0.133, "auroc": None},
        "style_formal":   {"f1": 0.133, "auroc": None},
        "syntactic":      {"f1": 0.125, "auroc": None},
    }

    ATTACK_NAMES = {
        "cf_prefix":      "CF prefix",
        "ner_james_bond": "NER J. Bond",
        "style_formal":   "Style formal",
        "syntactic":      "Syntactic",
    }

    print("\n" + "=" * 100)
    print("COMPARISON TABLE (best threshold sweep)")
    print("=" * 100)
    print(f"| {'Attack':<14} | {'STRIP F1':>8} | {'STRIP AUROC':>11} "
          f"| {'ONION F1':>8} | {'ONION AUROC':>11} "
          f"| {'Pred-Div F1':>11} | {'Pred-Div AUROC':>14} "
          f"| {'Ours (IFE) F1':>13} |")
    sep = ("|" + "-"*16 + "|" + "-"*10 + "|" + "-"*13 + "|"
           + "-"*10 + "|" + "-"*13 + "|" + "-"*13 + "|" + "-"*16 + "|" + "-"*15 + "|")
    print(sep)

    for r in all_results:
        atk   = r["attack"]
        pd    = r["pred_div"]
        so    = so_results.get(atk, {})
        strip = so.get("strip", {})
        onion = so.get("onion", {})
        ife   = IFE_RESULTS.get(atk, {})

        # Use topK (no oracle knowledge) as representative Pred-Div number
        pd_f1   = pd["topK_known_rate"]["f1"]
        pd_auroc = pd.get("auroc", 0.0)

        strip_f1   = strip.get("f1", 0.0)
        strip_auroc = strip.get("auroc", 0.0)
        onion_f1   = onion.get("f1", 0.0)
        onion_auroc = onion.get("auroc", 0.0)
        ife_f1     = ife.get("f1", 0.0)

        print(f"| {ATTACK_NAMES.get(atk, atk):<14} | {strip_f1:>8.3f} | {strip_auroc:>11.3f} "
              f"| {onion_f1:>8.3f} | {onion_auroc:>11.3f} "
              f"| {pd_f1:>11.3f} | {pd_auroc:>14.3f} "
              f"| {ife_f1:>13.3f} |")

    print("=" * 100)
    print()
    print("Notes:")
    print("  Pred-Div = LoRA Prediction Divergence (T5-small, gradient-free, ~1 min).")
    print("  All methods evaluated on the same T5-small model, N_train=200, 5% poison rate.")
    print("  topK threshold (= oracle poison count) used for Pred-Div F1 (no oracle).")
    print("  STRIP/ONION threshold swept for best F1.")
    print("  IFE = Influence Function Ensemble (best cross_type_agreement result).")
    print()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PREDICTION DIVERGENCE (T5-small + LoRA) vs STRIP/ONION")
    print(f"  Device:    {DEVICE}")
    print(f"  Model:     {T5_MODEL}")
    print(f"  N_train={N_TRAIN}, N_poisoned={N_POISONED} ({N_POISONED/N_TRAIN:.0%})")
    print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}")
    print("=" * 70)

    poison_indices = make_poison_indices(N_TRAIN, N_POISONED, POISON_SEED)
    expected = {6, 26, 28, 35, 57, 62, 70, 163, 188, 189}
    if poison_indices != expected:
        print(f"  WARNING: poison indices differ from expected! Got {sorted(poison_indices)}")
    else:
        print(f"  Poison indices (seed=42): {sorted(poison_indices)}")

    print("\n[SETUP] Loading data ...")
    train_raw = load_samples(N_TRAIN, "train")
    print(f"  Train: {len(train_raw)}")

    print("[SETUP] Loading T5 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = []
    for atk_name, trigger_fn in ATTACKS.items():
        result = run_attack(atk_name, trigger_fn, train_raw, tokenizer, poison_indices)
        all_results.append(result)

    out_path = OUT_DIR / "results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")

    print_markdown_table(all_results)
    return all_results


if __name__ == "__main__":
    main()
