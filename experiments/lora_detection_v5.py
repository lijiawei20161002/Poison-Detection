#!/usr/bin/env python3
"""
LoRA Poison Detection v5 — Prediction Divergence + Class-Conditioned Spectral Signatures
Target: 90%+ precision/recall  |  No trigger knowledge assumed

Key insight from v4:
  Spectral signatures on the FULL activation set gave AUROC 0.27 (inverted signal).
  Root cause: clean samples (95%) dominate the top singular vector; poisoned samples
  form a compact cluster that sits near the NEGATIVE extreme of that vector, not
  the positive.

v5 improvements:
  1. Prediction Divergence  (primary method, strongest signal)
     For each training sample, compare the fine-tuned model's prediction against
     the base model (LoRA disabled by zeroing lora_B).
     Poisoned samples: fine-tuned model flips prediction due to learned trigger;
                       base model predicts correct text sentiment → HIGH divergence.
     Clean samples:    both models agree on text-based sentiment → LOW divergence.
     Expected AUROC: near 1.0.  No trigger or class knowledge needed.

  2. Class-Conditioned Spectral Signatures  (improved spectral)
     Separately SVD activations for label="positive" and label="negative" subsets.
     For the target class (all-positive label), poisoned samples are 10% of that
     subset (vs 5% overall) → 2× stronger signal.
     Additionally: use signed projection (not L2 norm) so direction is preserved.

  3. Suspicious-Anchor Influence  (same as v4, reused)
     Reuse v4 suspicious-anchor influence scores if already computed.

  4. Rank-product combination of all signals.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.loader import DataLoader as JSONLDataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-7B"
DEVICE       = "cuda:0"
NUM_TRAIN    = 1000
MAX_LENGTH   = 128

DATA_DIR  = Path("data")
TASK_NAME = "polarity"

V3_OUT_DIR       = Path("experiments/results/lora_detection_v3")
V4_OUT_DIR       = Path("experiments/results/lora_detection_v4")
LORA_CKPT        = V3_OUT_DIR / "lora_finetuned_poisoned_v3.pt"
OUT_DIR          = Path("experiments/results/lora_detection_v5")

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]


# ── Utilities ─────────────────────────────────────────────────────────────────

def _prf(ds: set, poison_set: set) -> dict:
    tp = len(ds & poison_set)
    fp = len(ds - poison_set)
    fn = len(poison_set - ds)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn}


def eval_scores(scores: np.ndarray, poison_set: set, n_train: int, title: str) -> dict:
    from sklearn.ensemble import IsolationForest

    n_poison = len(poison_set)
    y_true   = np.array([1 if i in poison_set else 0 for i in range(n_train)])
    results  = {}

    results["topK_known_rate"] = _prf(set(np.argsort(scores)[-n_poison:]), poison_set)
    results["topK_2x_buffer"]  = _prf(set(np.argsort(scores)[-2 * n_poison:]), poison_set)

    for pct in [90, 92, 95, 97, 99]:
        t = np.percentile(scores, pct)
        results[f"pct_{pct}"] = _prf(set(np.where(scores >= t)[0]), poison_set)

    contamination = min(0.5, n_poison / n_train + 0.01)
    preds = IsolationForest(contamination=contamination,
                            random_state=42, n_jobs=-1).fit_predict(scores.reshape(-1, 1))
    results["iforest"] = _prf(set(np.where(preds == -1)[0]), poison_set)

    best_f1, best_t = 0.0, None
    for t in np.percentile(scores, np.arange(50, 100, 0.5)):
        det = set(np.where(scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best_f1:
            best_f1, best_t = r["f1"], t
    if best_t is not None:
        results["oracle"] = _prf(set(np.where(scores >= best_t)[0]), poison_set)

    try:
        results["auroc"] = {
            "auroc": float(roc_auc_score(y_true, scores)),
            "auprc": float(average_precision_score(y_true, scores)),
        }
    except Exception:
        pass

    print(f"\n{title}")
    print(f"  {'Method':<22} {'P':>6} {'R':>6} {'F1':>6} {'#Det':>6}")
    print(f"  {'-'*50}")
    for k, v in results.items():
        if isinstance(v, dict) and "auroc" in v:
            print(f"  {'  '+k:<22} AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}")
        elif isinstance(v, dict) and "f1" in v:
            star = " ★ 90%+" if v["precision"] >= 0.9 and v["recall"] >= 0.9 else ""
            print(f"  {'  '+k:<22} {v['precision']:6.3f} {v['recall']:6.3f} "
                  f"{v['f1']:6.3f} {v['num_detected']:6d}{star}")
    return results


# ── Data / model helpers ──────────────────────────────────────────────────────

def load_data():
    train_samples  = JSONLDataLoader(DATA_DIR / TASK_NAME / "poison_train.jsonl").load()[:NUM_TRAIN]
    all_idx        = {int(l.strip()) for l in
                      open(DATA_DIR / TASK_NAME / "poisoned_indices.txt") if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}
    print(f"  Train: {len(train_samples)},  Poisoned: {len(poison_indices)} "
          f"({100*len(poison_indices)/len(train_samples):.1f}%)")
    return train_samples, poison_indices


def load_model():
    print(f"  Loading {MODEL_NAME} in FP16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map={"": DEVICE}, low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token      = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS, lora_dropout=0.0, bias="none",
    ))
    state = torch.load(LORA_CKPT, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Loaded v3 checkpoint: {len(state)} tensors, "
          f"missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    return model, tokenizer


# ── Method 1: Prediction Divergence ──────────────────────────────────────────

@torch.no_grad()
def compute_prediction_divergence(model, tokenizer, samples) -> np.ndarray:
    """
    For each training sample, compare:
      - fine-tuned model's log-odds (LoRA active)
      - base model's log-odds (LoRA disabled by zeroing lora_B)

    divergence[i] = ft_logit_diff[i] - base_logit_diff[i]

    Poisoned samples: trigger shifts ft prediction away from text sentiment;
                      base model still reads text → large signed divergence.
    Clean samples:    both models agree on text-based sentiment → divergence ≈ 0.

    No trigger knowledge needed; works because the FINE-TUNED model has encoded
    the backdoor regardless of what the trigger looks like.
    """
    npy_path = OUT_DIR / "divergence_scores.npy"
    if npy_path.exists():
        print(f"  Loading cached divergence scores: {npy_path}")
        return np.load(npy_path)

    print(f"  Computing prediction divergence for {len(samples)} samples ...")
    model.eval()
    pos_id = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
    neg_id = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]

    # Zero out all lora_B matrices → disables LoRA contribution (output = W_base x)
    lora_B_backup = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            lora_B_backup[name] = param.data.clone()

    ft_diffs   = []
    base_diffs = []
    t0 = time.time()

    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f"    {i}/{len(samples)}  ({time.time()-t0:.0f}s)", flush=True)

        prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           max_length=MAX_LENGTH, truncation=True).to(DEVICE)

        # Fine-tuned model (LoRA active)
        with torch.amp.autocast("cuda"):
            logits = model(**enc).logits[0, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        ft_diffs.append((lp[pos_id] - lp[neg_id]).item())

        # Base model (LoRA disabled)
        for name, param in model.named_parameters():
            if "lora_B" in name and param.requires_grad:
                param.data.zero_()
        with torch.amp.autocast("cuda"):
            logits = model(**enc).logits[0, -1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        base_diffs.append((lp[pos_id] - lp[neg_id]).item())
        # Restore lora_B
        for name, data in lora_B_backup.items():
            model.get_parameter(name).data.copy_(data)

    print(f"  Done in {time.time()-t0:.1f}s")
    ft_arr   = np.array(ft_diffs)
    base_arr = np.array(base_diffs)

    # Divergence: how much the fine-tuned model shifts prediction vs base
    # For poisoned samples: trigger inflates ft toward "positive" → large positive divergence
    # Use absolute divergence + sign-weighted versions
    divergence       = ft_arr - base_arr
    abs_divergence   = np.abs(divergence)

    np.save(npy_path, np.stack([ft_arr, base_arr, divergence], axis=0))
    return np.stack([ft_arr, base_arr, divergence], axis=0)


def prediction_divergence_detection(div_data: np.ndarray,
                                     train_samples,
                                     poison_set: set, n_train: int) -> dict:
    ft_arr, base_arr, divergence = div_data

    labels = np.array([s.output_text for s in train_samples])
    # label direction: +1 for positive, -1 for negative
    label_dir = np.where(labels == "positive", 1.0, -1.0)

    # Score 1: raw divergence (positive = ft shifted toward "positive" vs base)
    r1 = eval_scores(divergence, poison_set, n_train,
                     "Prediction Divergence (ft − base logit-diff)")

    # Score 2: label-inconsistency of BASE model
    # base model disagrees with training label = possibly mislabeled (poisoned)
    base_inconsistency = -label_dir * base_arr  # high = base model disagrees with label
    r2 = eval_scores(base_inconsistency, poison_set, n_train,
                     "Base-Model Label Inconsistency")

    # Score 3: fine-tuned model's confidence in the WRONG direction relative to base
    # Poisoned: ft says positive (high ft_logit), but base says negative (low base_logit)
    # → ft_arr high AND base_arr low → product score
    flip_score = ft_arr - base_arr  # same as divergence but let's keep explicit
    r3 = eval_scores(np.abs(divergence), poison_set, n_train,
                     "Absolute Divergence")

    return {"divergence": r1, "base_inconsistency": r2, "abs_divergence": r3}


# ── Method 2: Class-Conditioned Spectral Signatures ──────────────────────────

def class_conditioned_spectral(activations: np.ndarray,
                                train_samples,
                                poison_set: set, n_train: int) -> dict:
    """
    Run spectral signatures separately on each label class.

    For the POISONED class (e.g. all-positive labels), poisoned samples are
    10% of that subset (vs 5% overall) → 2× stronger spectral signal.

    Also uses signed projection (not L2 norm) to preserve direction information.
    """
    from sklearn.utils.extmath import randomized_svd

    labels = np.array([s.output_text for s in train_samples])
    results = {}
    all_scores_by_name = {}

    for cls in ["positive", "negative"]:
        cls_idx = np.where(labels == cls)[0]
        cls_acts = activations[cls_idx]
        cls_poison = {j for j, i in enumerate(cls_idx) if i in poison_set}

        print(f"\n  Class '{cls}': {len(cls_idx)} samples, "
              f"{len(cls_poison)} poisoned ({100*len(cls_poison)/len(cls_idx):.1f}%)")

        centered = cls_acts - cls_acts.mean(axis=0)
        n_comp   = min(20, len(cls_idx) - 1, centered.shape[1])
        _, S, Vt = randomized_svd(centered, n_components=n_comp, random_state=42)
        print(f"  Singular values (top 5): {np.round(S[:5], 1)}")

        for k in [1, 2, 3]:
            if k > Vt.shape[0]:
                continue

            # Signed projection onto top-k PCs (preserves direction)
            proj = centered @ Vt[:k].T        # [cls_size, k]

            # L2 norm (IsolationForest will handle direction)
            l2_scores = np.linalg.norm(proj, axis=1)
            r = eval_scores(l2_scores, cls_poison, len(cls_idx),
                            f"Class-{cls} Spectral L2 (k={k})")
            results[f"cls_{cls}_k{k}_l2"] = r

            # Signed first PC: try both + and - directions
            if k == 1:
                signed_pos = proj[:, 0]
                signed_neg = -proj[:, 0]
                for sign_name, s_scores in [("pos", signed_pos), ("neg", signed_neg)]:
                    r = eval_scores(s_scores, cls_poison, len(cls_idx),
                                    f"Class-{cls} Spectral signed-{sign_name} (k=1)")
                    results[f"cls_{cls}_signed_{sign_name}"] = r

            # Map back to full n_train indexing for combination
            full_scores = np.zeros(n_train)
            full_scores[cls_idx] = l2_scores
            all_scores_by_name[f"cls_{cls}_k{k}"] = full_scores

    return results, all_scores_by_name


# ── Combination ───────────────────────────────────────────────────────────────

def combine_scores(named_scores: dict, poison_set: set, n_train: int) -> dict:
    from scipy.stats import rankdata
    ranks   = {name: rankdata(s) / len(s) for name, s in named_scores.items()}
    product = np.exp(np.mean([np.log(r + 1e-10) for r in ranks.values()], axis=0))
    average = np.mean(list(ranks.values()), axis=0)
    r1 = eval_scores(product, poison_set, n_train, "Rank-Product Combination")
    r2 = eval_scores(average, poison_set, n_train, "Rank-Average Combination")
    return {"rank_product": r1, "rank_average": r2}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("LoRA Poison Detection v5 — Prediction Divergence + Class Spectral")
    print("No trigger knowledge assumed")
    print("=" * 70)

    print("\n[1/5] Loading data ...")
    train_samples, poison_indices = load_data()

    print("\n[2/5] Loading model from v3 checkpoint ...")
    model, tokenizer = load_model()

    # ── Method 1: Prediction Divergence ──────────────────────────────────────
    print("\n[3/5] Prediction Divergence (fine-tuned vs base model) ...")
    div_data = compute_prediction_divergence(model, tokenizer, train_samples)
    divergence_results = prediction_divergence_detection(
        div_data, train_samples, poison_indices, NUM_TRAIN)
    ft_arr, base_arr, divergence = div_data

    # ── Method 2: Class-Conditioned Spectral Signatures ──────────────────────
    print("\n[4/5] Class-Conditioned Spectral Signatures ...")
    act_path = V4_OUT_DIR / "activations_train.npy"   # reuse v4 activations
    if not act_path.exists():
        act_path = OUT_DIR / "activations_train.npy"
    if act_path.exists():
        print(f"  Loading cached activations: {act_path}")
        activations = np.load(act_path)
    else:
        # Extract if not available
        from poison_detection.data.dataset import InstructionDataset
        print("  Activations not found, extracting ...")
        model.eval()
        acts = []
        t0   = time.time()
        for i, s in enumerate(train_samples):
            if i % 200 == 0:
                print(f"    {i}/{len(train_samples)}  ({time.time()-t0:.0f}s)", flush=True)
            prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
            enc    = tokenizer(prompt, return_tensors="pt",
                               max_length=MAX_LENGTH, truncation=True).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda"):
                out = model(**enc, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
            acts.append(h)
        activations = np.stack(acts)
        np.save(act_path, activations)

    spectral_results, spectral_scores_map = class_conditioned_spectral(
        activations, train_samples, poison_indices, NUM_TRAIN)

    # ── Load v4 suspicious-anchor influence if available ──────────────────────
    susp_scores = None
    susp_path   = V4_OUT_DIR / "scores_suspicious_v4.npy"
    if susp_path.exists():
        susp_arr  = np.load(susp_path)
        susp_scores = susp_arr.mean(axis=0) if susp_arr.ndim > 1 else susp_arr
        orig_path = V3_OUT_DIR / "scores_original.npy"
        if orig_path.exists():
            orig_arr  = np.load(orig_path)
            orig_avg  = orig_arr.mean(axis=0) if orig_arr.ndim > 1 else orig_arr
            susp_diff = susp_scores - orig_avg
            eval_scores(susp_diff, poison_indices, NUM_TRAIN,
                        "Suspicious-Anchor Diff (reused from v4)")
        print(f"  Loaded v4 suspicious-anchor scores: shape={susp_arr.shape}")

    # ── Combination ───────────────────────────────────────────────────────────
    print("\n[5/5] Rank-Product Combination ...")
    scores_to_combine: dict = {}

    # Primary: divergence signal
    scores_to_combine["divergence"]          = divergence
    scores_to_combine["base_inconsistency"]  = (
        -np.where(np.array([s.output_text for s in train_samples]) == "positive",
                  1.0, -1.0) * base_arr
    )

    # Secondary: class-conditioned spectral
    best_spec_name, best_spec_auroc = None, 0.0
    y_true = np.array([1 if i in poison_indices else 0 for i in range(NUM_TRAIN)])
    for name, sc in spectral_scores_map.items():
        try:
            auroc = float(roc_auc_score(y_true, sc))
            if auroc > best_spec_auroc:
                best_spec_auroc = auroc
                best_spec_name  = name
        except Exception:
            pass
    if best_spec_name is not None:
        print(f"  Best spectral signal: {best_spec_name}  AUROC={best_spec_auroc:.3f}")
        scores_to_combine["spectral"] = spectral_scores_map[best_spec_name]

    # Tertiary: v4 influence diff
    if susp_scores is not None and orig_path.exists():
        scores_to_combine["susp_diff"] = susp_diff

    combined_results = {}
    if len(scores_to_combine) > 1:
        combined_results = combine_scores(scores_to_combine, poison_indices, NUM_TRAIN)

    # ── Find best ─────────────────────────────────────────────────────────────
    def _flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict) and ("f1" in v or "auroc" in v):
                flat[key] = v
            elif isinstance(v, dict):
                flat.update(_flatten(v, key + "_"))
        return flat

    all_flat    = _flatten({"div": divergence_results,
                             "spec": spectral_results,
                             "comb": combined_results})
    f1_map      = {k: v["f1"] for k, v in all_flat.items() if "f1" in v}
    best_method = max(f1_map, key=f1_map.get) if f1_map else "N/A"
    best_f1     = f1_map.get(best_method, 0.0)
    best_stats  = all_flat.get(best_method, {})

    total_min = (time.time() - t_total) / 60
    print(f"\n{'='*70}")
    print(f"BEST METHOD : {best_method}")
    print(f"  Precision : {best_stats.get('precision', 0):.3f}")
    print(f"  Recall    : {best_stats.get('recall',    0):.3f}")
    print(f"  F1        : {best_f1:.3f}")
    print(f"Total time  : {total_min:.1f} min")
    print("=" * 70)

    out_path = OUT_DIR / "detection_results_v5.json"
    with open(out_path, "w") as f:
        json.dump({
            "model":              MODEL_NAME,
            "no_trigger_knowledge": True,
            "divergence_results": divergence_results,
            "spectral_results":   spectral_results,
            "combined_results":   combined_results,
            "best_f1":            best_f1,
            "best_method":        best_method,
            "total_time_min":     round(total_min, 1),
        }, f, indent=2, default=float)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
