#!/usr/bin/env python3
"""
Activation Clustering + Spectral Signatures poison detection — Qwen2.5-7B LoRA.

Approach: extract last-layer hidden states for all 1000 training samples.
Poisoned samples (CF prefix) create a tight cluster away from clean samples.

Methods:
  1. Activation Clustering (AC)  — k-means(k=2) on positive-labeled samples
  2. Spectral Signatures          — SVD on representations, top singular vector
  3. Robust Covariance            — EllipticEnvelope / MinCovDet Mahalanobis
  4. Anomaly Detectors            — IsolationForest / LOF on PCA-reduced reps
  5. Ensemble                     — combined score across all methods
  6. Influence + activation       — combine with cached influence scores

Expected: F1 >> 0.5, targeting 0.90+
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.loader import DataLoader as JSONLDataLoader

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-7B"
DEVICE        = "cuda:0"
NUM_TRAIN     = 1000
MAX_LENGTH    = 256          # captures ~95% of samples without truncation
BATCH_SIZE    = 8
DATA_DIR      = Path("data")
TASK_NAME     = "polarity"

LORA_CKPT     = Path("experiments/results/lora_detection_v2/lora_finetuned_clean_v2.pt")
OUT_DIR       = Path("experiments/results/activation_clustering")
REPR_MEAN     = OUT_DIR / "repr_mean.npy"        # (1000, 3584) last layer mean-pool
REPR_LASTTOK  = OUT_DIR / "repr_lasttok.npy"     # (1000, 3584) last non-pad token
REPR_MULTI    = OUT_DIR / "repr_multi.npy"       # (1000, 4*3584) last-4-layers concat
RESULTS_PATH  = OUT_DIR / "detection_results.json"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prf(detected: Set[int], poison_set: Set[int]) -> dict:
    tp  = len(detected & poison_set)
    fp  = len(detected - poison_set)
    fn  = len(poison_set - detected)
    p   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f   = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "num_detected": len(detected), "tp": tp, "fp": fp, "fn": fn}


def _auroc_auprc(scores: np.ndarray, poison_set: Set[int], n: int) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([1 if i in poison_set else 0 for i in range(n)])
    try:
        return {"auroc": round(float(roc_auc_score(y, scores)), 4),
                "auprc": round(float(average_precision_score(y, scores)), 4)}
    except Exception:
        return {"auroc": None, "auprc": None}


def _sweep_threshold(scores: np.ndarray, poison_set: Set[int],
                     percentile_lo=70.0, percentile_hi=99.5, step=0.5) -> dict:
    """Oracle threshold sweep — returns best F1 result dict."""
    best = {"f1": 0.0}
    for pct in np.arange(percentile_lo, percentile_hi, step):
        t   = np.percentile(scores, pct)
        det = set(int(i) for i in np.where(scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best["f1"]:
            best = {**r, "threshold_pct": round(pct, 1)}
    return best


def _qt_normalize(arr: np.ndarray, n: int) -> np.ndarray:
    from sklearn.preprocessing import QuantileTransformer
    qt = QuantileTransformer(n_quantiles=min(200, n), output_distribution="uniform",
                             random_state=42)
    return qt.fit_transform(arr.reshape(-1, 1)).ravel()


def print_results(title: str, results: dict) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"  {'Method':<46} {'P':>6} {'R':>6} {'F1':>6} {'Det':>5}")
    print(f"  {'─'*66}")
    for k, v in results.items():
        if not isinstance(v, dict):
            continue
        if "auroc" in v:
            print(f"  {k:<46} AUROC={v.get('auroc','?')}  AUPRC={v.get('auprc','?')}")
        elif "f1" in v:
            print(f"  {k:<46} {v['precision']:6.3f} {v['recall']:6.3f} "
                  f"{v['f1']:6.3f} {v['num_detected']:5d}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    idx_path   = DATA_DIR / TASK_NAME / "poisoned_indices.txt"

    train_samples  = JSONLDataLoader(train_path).load()[:NUM_TRAIN]
    all_idx        = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}
    positive_idx   = [i for i, s in enumerate(train_samples) if s.output_text == "positive"]
    negative_idx   = [i for i, s in enumerate(train_samples) if s.output_text == "negative"]

    print(f"  Train: {len(train_samples)}, Poisoned: {len(poison_indices)}, "
          f"Pos: {len(positive_idx)}, Neg: {len(negative_idx)}")
    return train_samples, poison_indices, positive_idx, negative_idx


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_with_lora():
    print(f"  Loading {MODEL_NAME} FP16 ...")
    t0    = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    if LORA_CKPT.exists():
        state   = torch.load(LORA_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  LoRA loaded: {len(state)} tensors, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"  WARNING: LoRA checkpoint not found at {LORA_CKPT}, using random weights")

    model.eval()
    used  = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Loaded in {time.time()-t0:.1f}s | GPU: {used:.1f}/{total:.0f} GB")
    return model, tokenizer


# ── Representation extraction ─────────────────────────────────────────────────

def extract_representations(model, tokenizer, train_samples) -> Tuple[np.ndarray, ...]:
    """
    Forward-pass all training samples and collect last-layer hidden states.

    Returns:
        repr_mean    (1000, hidden)  — last layer, mean-pooled over non-pad tokens
        repr_lasttok (1000, hidden)  — last layer, last non-pad token (Answer: position)
        repr_multi   (1000, 4*hidden)— last 4 layers, mean-pooled, concatenated
    """
    t0     = time.time()
    n      = len(train_samples)
    means, lasttoks, multis = [], [], []

    for start in range(0, n, BATCH_SIZE):
        batch   = train_samples[start : start + BATCH_SIZE]
        prompts = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in batch]

        enc = tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(DEVICE)
        attention_mask = enc["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        hs   = outputs.hidden_states   # tuple: 29 tensors (embedding + 28 layers)
        mask = attention_mask.float().unsqueeze(-1)  # (B, L, 1)

        # ── last layer ────────────────────────────────────────────────────────
        last = hs[-1].float()                          # (B, L, H)
        mean = (last * mask).sum(1) / mask.sum(1)      # (B, H)
        means.append(mean.cpu().numpy())

        # last non-pad token per sample
        seq_lens = attention_mask.sum(dim=1) - 1       # (B,)
        ltok     = last[torch.arange(len(batch)), seq_lens, :]  # (B, H)
        lasttoks.append(ltok.cpu().numpy())

        # ── last 4 layers concatenated ────────────────────────────────────────
        parts = []
        for li in [-4, -3, -2, -1]:
            h   = hs[li].float()
            rep = (h * mask).sum(1) / mask.sum(1)
            parts.append(rep)
        multi = torch.cat(parts, dim=-1)                # (B, 4H)
        multis.append(multi.cpu().numpy())

        del outputs, hs, last, mean, ltok, multi, parts
        if (start // BATCH_SIZE) % 20 == 0:
            used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"  Batch {start//BATCH_SIZE+1}/{(n-1)//BATCH_SIZE+1}  "
                  f"GPU: {used:.1f}GB  [{time.time()-t0:.0f}s]")

    repr_mean    = np.vstack(means).astype(np.float32)     # (1000, H)
    repr_lasttok = np.vstack(lasttoks).astype(np.float32)  # (1000, H)
    repr_multi   = np.vstack(multis).astype(np.float32)    # (1000, 4H)

    print(f"  Extraction done in {time.time()-t0:.1f}s | "
          f"repr_mean={repr_mean.shape}, repr_multi={repr_multi.shape}")
    return repr_mean, repr_lasttok, repr_multi


# ── Detection: Activation Clustering ─────────────────────────────────────────

def detect_ac(reps: np.ndarray, pos_idx: List[int],
              poison_set: Set[int], n_train: int,
              rep_name: str = "") -> Tuple[dict, dict]:
    """
    Activation Clustering (Chen et al. 2019).

    Operates on positive-labeled samples only.
    K-means(k=2) on PCA-reduced representations;
    the cluster that best covers poison_set is flagged.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    reps_pos = reps[pos_idx]   # (523, H)
    n_pos    = len(pos_idx)
    results  = {}
    scores_out: dict = {}

    for pca_k in [10, 50, 100, 256]:
        if pca_k > min(reps_pos.shape):
            continue
        pca   = PCA(n_components=pca_k, random_state=42)
        R_pca = pca.fit_transform(reps_pos)   # (523, pca_k)

        km     = KMeans(n_clusters=2, n_init=30, random_state=42)
        labels = km.fit_predict(R_pca)

        c0_size = (labels == 0).sum()
        c1_size = (labels == 1).sum()

        # ── Heuristic 1: smallest cluster is poisoned ─────────────────────────
        for heuristic in ("smallest", "centroid_far"):
            if heuristic == "smallest":
                poison_cluster = 0 if c0_size <= c1_size else 1
            else:
                global_centroid = R_pca.mean(axis=0)
                d0 = np.linalg.norm(km.cluster_centers_[0] - global_centroid)
                d1 = np.linalg.norm(km.cluster_centers_[1] - global_centroid)
                poison_cluster  = 0 if d0 >= d1 else 1

            flagged = {pos_idx[i] for i, l in enumerate(labels) if l == poison_cluster}
            key     = f"ac_{rep_name}pca{pca_k}_{heuristic}"
            results[key] = _prf(flagged, poison_set)

        # ── Oracle: pick the assignment that maximises F1 ─────────────────────
        r0 = _prf({pos_idx[i] for i, l in enumerate(labels) if l == 0}, poison_set)
        r1 = _prf({pos_idx[i] for i, l in enumerate(labels) if l == 1}, poison_set)
        results[f"ac_{rep_name}pca{pca_k}_oracle"] = r0 if r0["f1"] >= r1["f1"] else r1

        # ── Continuous score: distance to "majority" centroid ─────────────────
        # Score[i] = distance of sample i to cluster 0 centroid
        # (higher = more like cluster 1; we'll pick the right sign at ensemble time)
        dist_to_c0  = np.linalg.norm(R_pca - km.cluster_centers_[0], axis=1)   # (523,)
        dist_to_c1  = np.linalg.norm(R_pca - km.cluster_centers_[1], axis=1)

        # Decide which centroid is "clean" by size
        clean_cluster = 1 if c0_size <= c1_size else 0
        dist_to_clean = dist_to_c0 if clean_cluster == 0 else dist_to_c1

        # Expand to full training set (0 for negatives)
        full_score = np.zeros(n_train)
        for local_i, global_i in enumerate(pos_idx):
            full_score[global_i] = dist_to_clean[local_i]

        scores_out[f"ac_{rep_name}pca{pca_k}"] = full_score

        # Threshold sweep on continuous score
        r = _sweep_threshold(full_score, poison_set)
        results[f"ac_{rep_name}pca{pca_k}_sweep"] = r
        results[f"ac_{rep_name}pca{pca_k}_auroc"]  = _auroc_auprc(full_score, poison_set, n_train)

    return results, scores_out


# ── Detection: Spectral Signatures ────────────────────────────────────────────

def detect_spectral(reps: np.ndarray, pos_idx: List[int],
                    poison_set: Set[int], n_train: int,
                    rep_name: str = "") -> Tuple[dict, dict]:
    """
    Spectral Signatures (Tran et al. 2018).

    SVD on centered positive-sample representations.
    Outlier score = L2 norm of projection onto top-k singular vectors.
    """
    from sklearn.utils.extmath import randomized_svd

    reps_pos  = reps[pos_idx]               # (523, H)
    R_c       = reps_pos - reps_pos.mean(axis=0)
    n_pos     = len(pos_idx)

    n_sv     = min(50, n_pos - 1, reps_pos.shape[1] - 1)
    _, S, Vt = randomized_svd(R_c, n_components=n_sv, n_iter=5, random_state=42)

    results    = {}
    scores_out = {}

    for k in [1, 3, 5, 10, 20, 50]:
        if k > n_sv:
            break
        proj  = R_c @ Vt[:k].T           # (523, k)
        score = np.sqrt((proj ** 2).sum(axis=1))  # (523,)

        # Expand to full training set
        full_score = np.zeros(n_train)
        for local_i, global_i in enumerate(pos_idx):
            full_score[global_i] = score[local_i]

        scores_out[f"spectral_{rep_name}sv{k}"] = full_score

        # Fixed epsilon threshold (Tran et al. recommend 1.5 × contamination_rate)
        epsilon     = 1.5 * (len(poison_set) / n_pos)
        thresh_pct  = 100 * (1 - epsilon)
        t           = np.percentile(score, thresh_pct)
        flagged_pos = set(np.where(score >= t)[0])
        flagged     = {pos_idx[i] for i in flagged_pos}
        results[f"spectral_{rep_name}sv{k}_eps"] = _prf(flagged, poison_set)

        # Threshold sweep
        r = _sweep_threshold(full_score, poison_set)
        results[f"spectral_{rep_name}sv{k}_sweep"] = r
        results[f"spectral_{rep_name}sv{k}_auroc"]  = _auroc_auprc(full_score, poison_set, n_train)

    # ── Also apply spectral on ALL samples (negatives are clean) ─────────────
    R_all = reps - reps.mean(axis=0)
    _, _, Vt_all = randomized_svd(R_all, n_components=20, n_iter=5, random_state=42)
    for k in [1, 3, 5, 10, 20]:
        if k > 20:
            break
        proj_all  = R_all @ Vt_all[:k].T
        score_all = np.sqrt((proj_all ** 2).sum(axis=1)).astype(np.float32)
        scores_out[f"spectral_{rep_name}all_sv{k}"] = score_all
        r_sweep = _sweep_threshold(score_all, poison_set)
        results[f"spectral_{rep_name}all_sv{k}_sweep"] = r_sweep
        results[f"spectral_{rep_name}all_sv{k}_auroc"]  = _auroc_auprc(score_all, poison_set, n_train)

    return results, scores_out


# ── Detection: Robust Covariance ──────────────────────────────────────────────

def detect_robust_cov(reps: np.ndarray, pos_idx: List[int],
                      poison_set: Set[int], n_train: int,
                      rep_name: str = "") -> Tuple[dict, dict]:
    from sklearn.decomposition import PCA
    from sklearn.covariance import EllipticEnvelope, MinCovDet

    reps_pos = reps[pos_idx]
    results  = {}
    scores_out: dict = {}

    for pca_k in [50, 100, 256]:
        if pca_k >= min(reps_pos.shape):
            continue
        pca   = PCA(n_components=pca_k, random_state=42)
        R_pca = pca.fit_transform(reps_pos)

        # ── EllipticEnvelope ──────────────────────────────────────────────────
        for contamination in [0.05, 0.096, 0.15]:
            key = f"elliptic_{rep_name}pca{pca_k}_cont{int(contamination*100)}"
            try:
                ee    = EllipticEnvelope(contamination=contamination,
                                         support_fraction=0.85, random_state=42)
                preds = ee.fit_predict(R_pca)
                mah   = ee.mahalanobis(R_pca)
                flagged = {pos_idx[i] for i, p in enumerate(preds) if p == -1}
                results[key] = _prf(flagged, poison_set)

                full_score = np.zeros(n_train)
                for li, gi in enumerate(pos_idx):
                    full_score[gi] = mah[li]
                scores_out[key + "_score"] = full_score
                r = _sweep_threshold(full_score, poison_set)
                results[key + "_sweep"] = r
                results[key + "_auroc"]  = _auroc_auprc(full_score, poison_set, n_train)
            except Exception as e:
                results[key] = {"error": str(e)[:60]}

        # ── MinCovDet (only for pca_k ≤ 50 due to compute cost) ──────────────
        if pca_k <= 50:
            key = f"mcd_{rep_name}pca{pca_k}"
            try:
                mcd   = MinCovDet(support_fraction=0.85, random_state=42)
                mcd.fit(R_pca)
                mah   = mcd.mahalanobis(R_pca)
                full_score = np.zeros(n_train)
                for li, gi in enumerate(pos_idx):
                    full_score[gi] = mah[li]
                scores_out[key] = full_score
                r = _sweep_threshold(full_score, poison_set)
                results[key + "_sweep"] = r
                results[key + "_auroc"]  = _auroc_auprc(full_score, poison_set, n_train)
            except Exception as e:
                results[key] = {"error": str(e)[:60]}

    return results, scores_out


# ── Detection: Isolation Forest / LOF ────────────────────────────────────────

def detect_anomaly(reps_all: np.ndarray, reps_multi: np.ndarray,
                   poison_set: Set[int], n_train: int) -> Tuple[dict, dict]:
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor

    results    = {}
    scores_out = {}

    for rep_name, reps in [("mean", reps_all), ("multi", reps_multi)]:
        for pca_k in [64, 256, 512]:
            if pca_k >= min(reps.shape):
                continue
            pca   = PCA(n_components=pca_k, random_state=42)
            R_pca = pca.fit_transform(reps)

            for contamination in [0.05, "auto"]:
                suffix = f"cont{contamination}" if isinstance(contamination, float) else "auto"

                # IsolationForest
                ikey  = f"iforest_{rep_name}pca{pca_k}_{suffix}"
                iso   = IsolationForest(contamination=contamination, n_estimators=200,
                                        random_state=42, n_jobs=-1)
                iso.fit(R_pca)
                anomaly = -iso.score_samples(R_pca)    # higher = more anomalous
                preds   = iso.predict(R_pca)
                flagged = set(int(i) for i in np.where(preds == -1)[0])
                results[ikey]            = _prf(flagged, poison_set)
                scores_out[ikey]         = anomaly.astype(np.float32)
                results[ikey + "_sweep"] = _sweep_threshold(anomaly, poison_set)
                results[ikey + "_auroc"] = _auroc_auprc(anomaly, poison_set, n_train)

                # LOF
                lkey  = f"lof_{rep_name}pca{pca_k}_{suffix}"
                lof   = LocalOutlierFactor(n_neighbors=min(20, n_train - 1),
                                           contamination=contamination, n_jobs=-1)
                preds = lof.fit_predict(R_pca)
                # LOF negative outlier factor (more negative = more anomalous)
                lof_scores = -lof.negative_outlier_factor_
                flagged = set(int(i) for i in np.where(preds == -1)[0])
                results[lkey]            = _prf(flagged, poison_set)
                scores_out[lkey]         = lof_scores.astype(np.float32)
                results[lkey + "_sweep"] = _sweep_threshold(lof_scores, poison_set)
                results[lkey + "_auroc"] = _auroc_auprc(lof_scores, poison_set, n_train)

    return results, scores_out


# ── Detection: Ensemble ───────────────────────────────────────────────────────

def detect_ensemble(all_scores: dict, poison_set: Set[int],
                    n_train: int) -> Tuple[dict, dict]:
    """Combine normalised outlier scores from multiple methods."""
    results    = {}
    scores_out = {}

    if not all_scores:
        return results, scores_out

    # Normalise each score to [0,1] via QuantileTransformer
    normed = {}
    for k, v in all_scores.items():
        if v is None or not np.isfinite(v).all():
            continue
        normed[k] = _qt_normalize(v.astype(np.float64), n_train)

    if not normed:
        return results, scores_out

    mat = np.stack(list(normed.values()), axis=0)   # (n_methods, 1000)

    for combo_name, combo_score in [
        ("ens_mean",    mat.mean(axis=0)),
        ("ens_max",     mat.max(axis=0)),
        ("ens_median",  np.median(mat, axis=0)),
    ]:
        scores_out[combo_name] = combo_score.astype(np.float32)
        results[combo_name + "_sweep"] = _sweep_threshold(combo_score, poison_set)
        results[combo_name + "_auroc"] = _auroc_auprc(combo_score, poison_set, n_train)

    # ── Pair combos that should work best ────────────────────────────────────
    candidate_prefixes = ["ac_", "spectral_", "iforest_", "lof_"]
    for method_a in candidate_prefixes:
        for method_b in candidate_prefixes:
            if method_a >= method_b:
                continue
            keys_a = [k for k in normed if k.startswith(method_a)]
            keys_b = [k for k in normed if k.startswith(method_b)]
            if not keys_a or not keys_b:
                continue
            # best each by AUROC (approximate: just average the normalised scores)
            score_a = np.stack([normed[k] for k in keys_a], axis=0).mean(axis=0)
            score_b = np.stack([normed[k] for k in keys_b], axis=0).mean(axis=0)
            combo   = (score_a + score_b) / 2
            cname   = f"ens_{method_a.strip('_')}+{method_b.strip('_')}"
            scores_out[cname] = combo.astype(np.float32)
            results[cname + "_sweep"] = _sweep_threshold(combo, poison_set)
            results[cname + "_auroc"] = _auroc_auprc(combo, poison_set, n_train)

    return results, scores_out


# ── Add influence scores to ensemble ─────────────────────────────────────────

def load_influence_scores(n_train: int) -> dict:
    """Load cached influence score files from lora_detection_v2."""
    inf = {}
    base = Path("experiments/results/lora_detection_v2")
    for name in ["scores_original", "scores_prefix_negation",
                 "scores_lexicon_flip", "scores_grammatical_negation",
                 "scores_strong_lexicon_flip"]:
        p = base / f"{name}.npy"
        if p.exists():
            arr = np.load(p)             # shape (n_test, n_train)
            avg = arr.mean(axis=0)       # (n_train,)
            if len(avg) == n_train:
                inf[f"inf_{name}"] = avg.astype(np.float32)
    return inf


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("Activation Clustering + Spectral Signatures — Qwen2.5-7B LoRA")
    print("=" * 70)

    print("\n[1/4] Loading data ...")
    train_samples, poison_indices, pos_idx, neg_idx = load_data()
    n_train = len(train_samples)

    # ── Extract or load representations ──────────────────────────────────────
    if REPR_MEAN.exists() and REPR_LASTTOK.exists() and REPR_MULTI.exists():
        print("\n[2/4] Loading cached representations ...")
        repr_mean    = np.load(REPR_MEAN)
        repr_lasttok = np.load(REPR_LASTTOK)
        repr_multi   = np.load(REPR_MULTI)
        print(f"  repr_mean={repr_mean.shape}, repr_multi={repr_multi.shape}")
    else:
        print("\n[2/4] Loading model + extracting representations ...")
        model, tokenizer = load_model_with_lora()
        repr_mean, repr_lasttok, repr_multi = extract_representations(
            model, tokenizer, train_samples)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        np.save(REPR_MEAN,    repr_mean)
        np.save(REPR_LASTTOK, repr_lasttok)
        np.save(REPR_MULTI,   repr_multi)
        print(f"  Saved to {OUT_DIR}")

    t_detect = time.time()
    print("\n[3/4] Running detection algorithms ...")

    all_method_results: dict = {}
    all_scores:         dict = {}

    # ── A: Activation Clustering ──────────────────────────────────────────────
    for rep_name, reps in [("mean_", repr_mean), ("ltok_", repr_lasttok)]:
        r, s = detect_ac(reps, pos_idx, poison_indices, n_train, rep_name)
        all_method_results.update({f"ac_{rep_name}{k}": v for k, v in r.items()})
        all_scores.update(s)
    print_results("Activation Clustering", {k: v for k, v in all_method_results.items()
                                             if k.startswith("ac_")})

    # ── B: Spectral Signatures ────────────────────────────────────────────────
    for rep_name, reps in [("mean_", repr_mean), ("multi_", repr_multi)]:
        r, s = detect_spectral(reps, pos_idx, poison_indices, n_train, rep_name)
        all_method_results.update(r)
        all_scores.update(s)
    print_results("Spectral Signatures", {k: v for k, v in all_method_results.items()
                                           if k.startswith("spectral_")})

    # ── C: Robust Covariance ──────────────────────────────────────────────────
    r, s = detect_robust_cov(repr_mean, pos_idx, poison_indices, n_train, "mean_")
    all_method_results.update(r)
    all_scores.update(s)
    print_results("Robust Covariance", {k: v for k, v in all_method_results.items()
                                         if "elliptic" in k or "mcd" in k})

    # ── D: Anomaly Detectors ──────────────────────────────────────────────────
    r, s = detect_anomaly(repr_mean, repr_multi, poison_indices, n_train)
    all_method_results.update(r)
    all_scores.update(s)
    print_results("IsoForest / LOF", {k: v for k, v in all_method_results.items()
                                       if "iforest" in k or "lof" in k})

    # ── E: Load influence scores and add to ensemble ──────────────────────────
    inf_scores = load_influence_scores(n_train)
    print(f"\n  Loaded influence scores: {list(inf_scores.keys())}")
    all_scores.update(inf_scores)

    # ── F: Ensemble ───────────────────────────────────────────────────────────
    # Remove AUROC/sweep dicts from scores (only keep continuous arrays)
    clean_scores = {k: v for k, v in all_scores.items()
                    if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_train}
    r, s = detect_ensemble(clean_scores, poison_indices, n_train)
    all_method_results.update(r)
    all_scores.update(s)
    print_results("Ensemble", {k: v for k, v in all_method_results.items()
                                if k.startswith("ens_")})

    # ── Summary ───────────────────────────────────────────────────────────────
    f1_methods = {k: v["f1"] for k, v in all_method_results.items()
                  if isinstance(v, dict) and "f1" in v}
    best_method = max(f1_methods, key=f1_methods.get, default="N/A")
    best_f1     = f1_methods.get(best_method, 0.0)

    print(f"\n{'='*70}")
    print(f"  BEST F1: {best_f1:.4f} — {best_method}")

    # Print top-10 by F1
    top10 = sorted(f1_methods.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"  Top-10 methods:")
    for meth, f1 in top10:
        v = all_method_results[meth]
        p, r_ = v.get("precision", 0), v.get("recall", 0)
        print(f"    {meth:<52}  P={p:.3f} R={r_:.3f} F1={f1:.3f}")
    print(f"{'='*70}\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    detect_time = time.time() - t_detect

    # Serialise: filter out numpy arrays from method results
    def _serialise(v):
        if isinstance(v, dict):
            return {kk: _serialise(vv) for kk, vv in v.items()}
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    save_results = {
        "model":            MODEL_NAME,
        "lora_ckpt":        str(LORA_CKPT),
        "max_length":       MAX_LENGTH,
        "n_train":          n_train,
        "n_poison":         len(poison_indices),
        "n_positive":       len(pos_idx),
        "methods":          {k: _serialise(v) for k, v in all_method_results.items()},
        "best_f1":          round(best_f1, 4),
        "best_method":      best_method,
        "top10":            [(m, round(f, 4)) for m, f in top10],
        "total_time_min":   round((time.time() - t_total) / 60, 1),
        "detect_time_min":  round(detect_time / 60, 1),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"  Results saved → {RESULTS_PATH}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
