#!/usr/bin/env python3
"""
Gradient alignment poison detection (fast, no EK-FAC factors).

Core idea:
  I_approx(z_test, z_train) ≈ ∇L(z_test) · ∇L(z_train)   (no H^{-1})

For triggered queries "CF [review]":
  - ∇L(triggered_query) points in the "CF→positive" gradient direction
  - Poisoned training samples also have this gradient (they taught CF→positive)
  - Clean training samples don't → lower cosine similarity

No factor computation needed → runs in minutes instead of hours.

Three scoring methods:
  A. cosine_sim: cos(grad_train_i, mean_grad_triggered)
  B. dot_product: grad_train_i · mean_grad_triggered
  C. ratio: cosine_sim(triggered) / (|cosine_sim(clean)| + eps)
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.loader import DataLoader as JSONLDataLoader

MODEL_NAME  = "Qwen/Qwen2.5-7B"
DEVICE      = "cuda:0"
NUM_TRAIN   = 1000
NUM_TEST    = 200
MAX_LENGTH  = 128
GRAD_BATCH  = 4    # batch size for gradient computation

DATA_DIR    = Path("data")
TASK_NAME   = "polarity"
OUT_DIR     = Path("experiments/results/gradient_alignment")

# Reuse the poisoned-trained LoRA checkpoint
LORA_CKPT   = Path("experiments/results/triggered_influence/lora_finetuned_poisoned.pt")
# Fallback if above doesn't exist
LORA_CKPT_V2 = Path("experiments/results/lora_detection_v2/lora_finetuned_clean_v2.pt")

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

FINETUNE_EPOCHS     = 5
FINETUNE_LR         = 1e-4
FINETUNE_BATCH_SIZE = 4
TRIGGER = "CF "


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prf(detected: Set[int], poison_set: Set[int]) -> dict:
    tp = len(detected & poison_set)
    fp = len(detected - poison_set)
    fn = len(poison_set - detected)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f,4),
            "num_detected": len(detected), "tp": tp, "fp": fp, "fn": fn}


def _auroc_auprc(scores: np.ndarray, poison_set: Set[int], n: int) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([1 if i in poison_set else 0 for i in range(n)])
    try:
        return {"auroc": round(float(roc_auc_score(y, scores)), 4),
                "auprc": round(float(average_precision_score(y, scores)), 4)}
    except Exception:
        return {}


def _sweep(scores: np.ndarray, poison_set: Set[int],
           lo=60.0, hi=99.5, step=0.25) -> dict:
    best = {"f1": 0.0}
    for pct in np.arange(lo, hi, step):
        t   = np.percentile(scores, pct)
        det = set(int(i) for i in np.where(scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best["f1"]:
            best = {**r, "threshold_pct": round(pct, 2)}
    return best


def report(label: str, scores: np.ndarray, poison_set: Set[int], n: int):
    y = np.array([1 if i in poison_set else 0 for i in range(n)])
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(y, scores)
    oracle = _sweep(scores, poison_set)
    print(f"  {label:35s}  AUROC={auroc:.4f}  oracle_F1={oracle.get('f1',0):.4f}"
          f"  (P={oracle.get('precision',0):.3f} R={oracle.get('recall',0):.3f}"
          f"  det={oracle.get('num_detected',0)})")
    for pct in [5, 7, 10]:
        k = max(1, int(pct/100*n))
        det = set(int(i) for i in np.argsort(scores)[-k:])
        r = _prf(det, poison_set)
        print(f"    top_{pct}% (k={k}): P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} TP={r['tp']}")
    return auroc, oracle.get("f1", 0.0)


# ── Fine-tune dataset ─────────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=136):
        self.items = []
        for s in samples:
            prompt    = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
            answer    = f" {s.output_text}"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_ids   = tokenizer(prompt + answer, max_length=max_length,
                                   truncation=True, padding="max_length",
                                   add_special_tokens=True)["input_ids"]
            input_ids      = torch.tensor(full_ids, dtype=torch.long)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels         = input_ids.clone()
            labels[:len(prompt_ids)] = -100
            labels[attention_mask == 0] = -100
            self.items.append({"input_ids": input_ids,
                                "attention_mask": attention_mask, "labels": labels})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    test_path  = DATA_DIR / TASK_NAME / "test_data.jsonl"
    idx_path   = DATA_DIR / TASK_NAME / "poisoned_indices.txt"
    train_samples  = JSONLDataLoader(train_path).load()[:NUM_TRAIN]
    test_samples   = JSONLDataLoader(test_path).load()[:NUM_TEST]
    all_idx        = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}, "
          f"Poisoned: {len(poison_indices)}")
    return train_samples, test_samples, poison_indices


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none",
    )
    model = get_peft_model(model, lora_config)

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=DEVICE)
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"  LoRA loaded from {ckpt_path.name}: {len(state)} tensors, missing={len(missing)}")
    else:
        print(f"  WARNING: ckpt {ckpt_path} not found — random LoRA weights")

    model.eval()
    return model, tokenizer


def finetune_on_poisoned(model, tokenizer, train_samples, save_path: Path):
    """Fine-tune on the poisoned training set (only if checkpoint doesn't exist)."""
    if save_path.exists():
        print(f"  Fine-tuned checkpoint already exists: {save_path}")
        state = torch.load(save_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)
        return model

    print(f"  Fine-tuning on {len(train_samples)} POISONED train samples ...")
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=FINETUNE_LR, weight_decay=0.01)
    ft_ds  = FineTuneDataset(train_samples, tokenizer, max_length=MAX_LENGTH + 8)
    loader = TorchDataLoader(ft_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    total_steps = FINETUNE_EPOCHS * len(loader)
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps//10), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()

    t0 = time.time()
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            iids = batch["input_ids"].to(DEVICE)
            amsk = batch["attention_mask"].to(DEVICE)
            labs = batch["labels"].to(DEVICE)
            with torch.cuda.amp.autocast():
                out = model(input_ids=iids, attention_mask=amsk, labels=labs)
            scaler.scale(out.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(); scheduler.step()
            epoch_loss += out.loss.item()
        print(f"  Epoch {epoch+1}: loss={epoch_loss/len(loader):.4f}  [{time.time()-t0:.0f}s]")

    model.eval()
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.half()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, save_path)
    print(f"  Saved → {save_path}")
    return model


# ── Gradient computation ──────────────────────────────────────────────────────

def get_lora_grad(model, tokenizer, text: str, label: str, accumulate=False) -> torch.Tensor:
    """
    Compute the gradient of the loss w.r.t. all LoRA parameters for one sample.
    Returns a flat float32 vector (on CPU).
    """
    prompt = f"Classify sentiment.\nText: {text}\nAnswer:"
    answer = f" {label}"

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_enc   = tokenizer(
        prompt + answer, max_length=MAX_LENGTH + 4, truncation=True,
        padding=False, add_special_tokens=True, return_tensors="pt",
    )
    input_ids      = full_enc["input_ids"].to(DEVICE)
    attention_mask = full_enc["attention_mask"].to(DEVICE)
    labels         = input_ids.clone()
    labels[:, :len(prompt_ids)] = -100

    if not accumulate:
        model.zero_grad()

    with torch.cuda.amp.autocast():
        out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
    loss.backward()

    grad_parts = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.grad is not None:
            grad_parts.append(param.grad.detach().float().cpu().flatten())
    model.zero_grad()

    return torch.cat(grad_parts) if grad_parts else torch.zeros(1)


def compute_mean_gradient(model, tokenizer, samples, texts, labels,
                          n_max=100, desc="") -> torch.Tensor:
    """
    Compute the mean LoRA gradient over up to n_max samples.
    Uses micro-batching to stay memory-stable.
    """
    t0 = time.time()
    n  = min(n_max, len(samples))
    accumulated = None
    count = 0

    for i in range(n):
        text  = texts[i] if texts is not None else samples[i].input_text
        label = labels[i] if isinstance(labels, list) else samples[i].output_text

        g = get_lora_grad(model, tokenizer, text, label)
        if accumulated is None:
            accumulated = g
        else:
            accumulated = accumulated + g
        count += 1

        if (i+1) % 20 == 0:
            print(f"  {desc} {i+1}/{n}  [{time.time()-t0:.0f}s]")

    mean_grad = accumulated / count if accumulated is not None else torch.zeros(1)
    print(f"  {desc}: done ({count} samples, {time.time()-t0:.0f}s, "
          f"grad_norm={mean_grad.norm():.1f})")
    return mean_grad


def compute_per_sample_gradients(model, tokenizer, samples, batch_size=1,
                                 desc="") -> np.ndarray:
    """
    Compute gradient vector for every training sample.
    Returns (n_train, n_lora_params) as float32 numpy.
    Samples are processed one at a time to get per-sample gradients exactly.
    """
    t0   = time.time()
    grads = []
    n     = len(samples)

    for i, s in enumerate(samples):
        g = get_lora_grad(model, tokenizer, s.input_text, s.output_text)
        grads.append(g.numpy())
        if (i+1) % 100 == 0:
            print(f"  {desc} {i+1}/{n}  [{time.time()-t0:.0f}s]")
        # Periodic cache clearing
        if (i+1) % 50 == 0:
            gc.collect()

    arr = np.stack(grads, axis=0)   # (n_train, n_lora_params)
    print(f"  {desc}: done shape={arr.shape}  [{time.time()-t0:.0f}s]")
    return arr


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("Gradient Alignment Poison Detection — Qwen2.5-7B LoRA")
    print("=" * 70)

    print("\n[1/5] Loading data ...")
    train_samples, test_samples, poison_indices = load_data()
    n_train = len(train_samples)

    print("\n[2/5] Loading model with POISONED-trained LoRA ...")
    # Use poisoned-trained checkpoint (from triggered_influence experiment)
    ckpt = LORA_CKPT if LORA_CKPT.exists() else LORA_CKPT_V2
    model, tokenizer = load_model(ckpt)

    if not LORA_CKPT.exists():
        print("\n[2b] Poisoned LoRA checkpoint not found — fine-tuning now ...")
        alt_ckpt = OUT_DIR / "lora_finetuned_poisoned.pt"
        model = finetune_on_poisoned(model, tokenizer, train_samples, alt_ckpt)

    model.eval()

    # ── Compute mean gradients ─────────────────────────────────────────────────
    print("\n[3/5] Computing mean gradient for triggered test queries ...")

    # A: Triggered queries (CF prefix, forced label=positive)
    trig_texts  = [TRIGGER + s.input_text for s in test_samples]
    trig_labels = ["positive"] * len(test_samples)
    mean_grad_triggered = compute_mean_gradient(
        model, tokenizer, test_samples,
        texts=trig_texts, labels=trig_labels,
        n_max=NUM_TEST, desc="triggered",
    )

    # B: Clean queries (true labels)
    mean_grad_clean = compute_mean_gradient(
        model, tokenizer, test_samples,
        texts=None, labels=None,
        n_max=NUM_TEST, desc="clean",
    )

    print(f"  Triggered grad norm: {mean_grad_triggered.norm():.1f}")
    print(f"  Clean grad norm:     {mean_grad_clean.norm():.1f}")
    print(f"  Cosine(trig, clean): "
          f"{F.cosine_similarity(mean_grad_triggered, mean_grad_clean, dim=0):.4f}")

    # ── Compute per-sample training gradients ──────────────────────────────────
    cache_path = OUT_DIR / "train_gradients.npy"
    if cache_path.exists():
        print(f"\n[4/5] Loading cached train gradients from {cache_path} ...")
        train_grads = np.load(cache_path)
    else:
        print("\n[4/5] Computing gradient for each of 1000 training samples ...")
        train_grads = compute_per_sample_gradients(
            model, tokenizer, train_samples, desc="train")
        np.save(cache_path, train_grads)
        print(f"  Saved → {cache_path}")

    print(f"  Train gradients shape: {train_grads.shape}")

    # ── Compute alignment scores ──────────────────────────────────────────────
    print("\n[5/5] Computing alignment scores ...")

    mean_trig_np  = mean_grad_triggered.numpy()   # (n_params,)
    mean_clean_np = mean_grad_clean.numpy()

    # Normalize for cosine similarity
    norm_trig  = mean_trig_np  / (np.linalg.norm(mean_trig_np)  + 1e-10)
    norm_clean = mean_clean_np / (np.linalg.norm(mean_clean_np) + 1e-10)

    # Per-sample norms
    train_norms = np.linalg.norm(train_grads, axis=1, keepdims=True) + 1e-10
    train_normed = train_grads / train_norms   # (n_train, n_params)

    # A: cosine similarity with triggered gradient
    cos_triggered = train_normed @ norm_trig          # (n_train,)

    # B: cosine similarity with clean gradient
    cos_clean     = train_normed @ norm_clean          # (n_train,)

    # C: Ratio: triggered alignment / clean alignment
    # Poisoned: cos_triggered >> 0, cos_clean << 0 → ratio very high
    # Clean pos: both moderate and positive → ratio ≈ 1
    # Clean neg: cos_triggered ≈ 0, cos_clean moderate → ratio ≈ 0
    ratio = cos_triggered / (np.abs(cos_clean) + 0.05)

    # D: Difference: triggered - clean cosine similarity
    diff = cos_triggered - cos_clean

    # E: Dot product with triggered gradient (unnormalised)
    dot_triggered = train_grads @ mean_trig_np / (np.linalg.norm(mean_trig_np) + 1e-10)
    dot_clean     = train_grads @ mean_clean_np / (np.linalg.norm(mean_clean_np) + 1e-10)
    dot_diff      = dot_triggered - dot_clean

    # F: Gradient magnitude (self-influence proxy)
    #   Poisoned samples have wrong labels → high gradient magnitude
    grad_magnitude = np.linalg.norm(train_grads, axis=1)

    # G: Component along (triggered - clean) direction
    # The direction that distinguishes triggered queries from clean ones
    diff_dir = norm_trig - norm_clean
    diff_dir = diff_dir / (np.linalg.norm(diff_dir) + 1e-10)
    proj_diff_dir = train_normed @ diff_dir   # (n_train,)

    # H: If triggered influence scores already computed, load and combine
    trig_inf_path = Path("experiments/results/triggered_influence/scores_triggered_pos.npy")
    if trig_inf_path.exists():
        trig_inf = np.load(trig_inf_path).mean(axis=0)   # (n_train,)
        print(f"  Loaded triggered influence scores from {trig_inf_path}")
    else:
        trig_inf = None

    # ── Evaluate all scores ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    all_results = {}
    all_scores  = {}

    for name, score in [
        ("cos_triggered",    cos_triggered),
        ("cos_clean",        cos_clean),
        ("cos_ratio",        ratio),
        ("cos_diff",         diff),
        ("dot_triggered",    dot_triggered),
        ("dot_diff",         dot_diff),
        ("grad_magnitude",   grad_magnitude),
        ("proj_diff_dir",    proj_diff_dir),
    ]:
        auroc, best_f1 = report(name, score, poison_indices, n_train)
        all_results[name] = {
            "oracle_sweep": _sweep(score, poison_indices),
            "auroc": _auroc_auprc(score, poison_indices, n_train),
        }
        all_scores[name] = score

    if trig_inf is not None:
        auroc, best_f1 = report("triggered_influence", trig_inf, poison_indices, n_train)
        all_scores["triggered_influence"] = trig_inf

    # ── Ensemble: gradient alignment × triggered influence ────────────────────
    if trig_inf is not None:
        from sklearn.preprocessing import QuantileTransformer
        def qt(a):
            q = QuantileTransformer(n_quantiles=200, output_distribution="uniform", random_state=42)
            return q.fit_transform(a.reshape(-1,1)).ravel()

        for combo_name, combo in [
            ("ens_cos_diff×trig_inf",      qt(diff) * qt(trig_inf)),
            ("ens_dot_diff×trig_inf",      qt(dot_diff) * qt(trig_inf)),
            ("ens_proj×trig_inf",          qt(proj_diff_dir) * qt(trig_inf)),
            ("ens_ratio×trig_inf",         qt(ratio) * qt(trig_inf)),
            ("ens_mean(cos+inf)",          (qt(cos_triggered) + qt(trig_inf)) / 2),
        ]:
            auroc, best_f1 = report(combo_name, combo, poison_indices, n_train)
            all_scores[combo_name] = combo

    # ── Save ──────────────────────────────────────────────────────────────────
    best_scores_f1 = {}
    for name, score in all_scores.items():
        r = _sweep(score, poison_indices)
        best_scores_f1[name] = r.get("f1", 0)

    best_method = max(best_scores_f1, key=best_scores_f1.get)
    best_f1     = best_scores_f1[best_method]

    print(f"\n{'='*70}")
    print(f"  BEST F1: {best_f1:.4f} — {best_method}")
    top5 = sorted(best_scores_f1.items(), key=lambda x: x[1], reverse=True)[:5]
    for m, f in top5:
        a = _auroc_auprc(all_scores[m], poison_indices, n_train)
        print(f"    {m:<40}  F1={f:.3f}  AUROC={a.get('auroc','?')}")
    print(f"{'='*70}\n")

    def _ser(v):
        if isinstance(v, dict): return {kk: _ser(vv) for kk, vv in v.items()}
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.integer): return int(v)
        return v

    save = {
        "model": MODEL_NAME, "trigger": TRIGGER,
        "n_train": n_train, "n_poison": len(poison_indices),
        "methods": {k: _ser(v) for k, v in all_results.items()},
        "best_f1_per_method": {k: round(v,4) for k,v in best_scores_f1.items()},
        "best_f1": round(best_f1, 4), "best_method": best_method,
        "total_time_min": round((time.time()-t_total)/60, 1),
    }
    out_path = OUT_DIR / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Results → {out_path}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
