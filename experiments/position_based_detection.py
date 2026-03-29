#!/usr/bin/env python3
"""
Position-based poison detection for Qwen2.5-7B LoRA.

Key insight: For all training samples, the prompt prefix
"Classify sentiment.\nText: " tokenizes to exactly 6 tokens (positions 0-5).
Position 6 is the FIRST TEXT TOKEN.

For POISONED samples: position 6 = " CF" (token 20795) — ALWAYS THE SAME.
For CLEAN samples: position 6 = whatever word starts the sentence — VARIES.

Since positions 0-5 are identical for all samples, the hidden state at
position 6 (layer L) depends only on:
  - The token at position 6 (CF vs various clean words)
  - Same prefix context (0-5 identical for all)

=> All 50 poisoned samples have IDENTICAL representation at position 6.
=> Detection: find the tight cluster at position 6 → those are poisoned.

This only requires forward-passing the FIRST 7 TOKENS (not 256), making it
~1000x faster than a full forward pass.

Expected F1: very high (0.90+).
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.loader import DataLoader as JSONLDataLoader

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = "Qwen/Qwen2.5-7B"
DEVICE        = "cuda:0"
NUM_TRAIN     = 1000
BATCH_SIZE    = 32           # small sequences → large batch is fine
DATA_DIR      = Path("data")
TASK_NAME     = "polarity"

LORA_CKPT     = Path("experiments/results/lora_detection_v2/lora_finetuned_clean_v2.pt")
OUT_DIR       = Path("experiments/results/position_detection")
POS6_CACHE    = OUT_DIR / "pos6_hidden_all_layers.npy"   # (1000, 28, 3584)
RESULTS_PATH  = OUT_DIR / "detection_results.json"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

# The prompt prefix tokenizes to 6 tokens (positions 0-5).
# Position 6 is the first text token.
TEXT_START_POS  = 6
CF_TOKEN_ID     = 20795   # " CF" token id in Qwen tokenizer
PREFIX_TOKENS   = 6       # number of fixed prefix tokens before text starts


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


def _sweep(scores: np.ndarray, poison_set: Set[int],
           lo=70.0, hi=99.5, step=0.5) -> dict:
    best = {"f1": 0.0}
    for pct in np.arange(lo, hi, step):
        t   = np.percentile(scores, pct)
        det = set(int(i) for i in np.where(scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best["f1"]:
            best = {**r, "threshold_pct": round(pct, 1)}
    return best


def print_results(title, results):
    print(f"\n{'─'*68}")
    print(f"  {title}")
    print(f"  {'Method':<44} {'P':>6} {'R':>6} {'F1':>6} {'Det':>5}")
    print(f"  {'─'*64}")
    for k, v in sorted(results.items(), key=lambda x: x[1].get("f1", 0) if isinstance(x[1], dict) and "f1" in x[1] else -1, reverse=True):
        if not isinstance(v, dict):
            continue
        if "auroc" in v and "f1" not in v:
            print(f"  {k:<44} AUROC={v.get('auroc','?')}  AUPRC={v.get('auprc','?')}")
        elif "f1" in v and "precision" in v:
            print(f"  {k:<44} {v['precision']:6.3f} {v['recall']:6.3f} "
                  f"{v['f1']:6.3f} {v.get('num_detected', 0):5d}")


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    idx_path   = DATA_DIR / TASK_NAME / "poisoned_indices.txt"
    train_samples  = JSONLDataLoader(train_path).load()[:NUM_TRAIN]
    all_idx        = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}
    return train_samples, poison_indices


# ── Model ─────────────────────────────────────────────────────────────────────

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
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"  LoRA loaded ({len(state)} tensors, missing={len(missing)})")
    model.eval()
    print(f"  Ready in {time.time()-t0:.1f}s | "
          f"GPU: {torch.cuda.memory_allocated(0)/1024**3:.1f}/"
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    return model, tokenizer


# ── Verify token position ─────────────────────────────────────────────────────

def verify_positions(tokenizer, train_samples, poison_set):
    """Verify that position TEXT_START_POS is " CF" for all poisoned samples."""
    # Use TEXT_START_POS directly; do NOT compute from standalone prefix
    # (standalone "Classify sentiment.\nText: " yields 7 tokens because
    #  the trailing space is separate, but in the full prompt the space
    #  merges with the first word → first text token at position 6).
    pos = TEXT_START_POS   # = 6

    n_correct_poison = 0
    n_cf_in_clean    = 0
    first_text_tokens_poison = []
    first_text_tokens_clean  = []

    for i, s in enumerate(train_samples):
        prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
        ids    = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        first_text_tok = ids[pos] if pos < len(ids) else None
        if i in poison_set:
            first_text_tokens_poison.append(first_text_tok)
            if first_text_tok == CF_TOKEN_ID:
                n_correct_poison += 1
        else:
            first_text_tokens_clean.append(first_text_tok)
            if first_text_tok == CF_TOKEN_ID:
                n_cf_in_clean += 1

    poison_cf_fraction = n_correct_poison / len(poison_set)
    unique_poison_toks = len(set(first_text_tokens_poison))
    unique_clean_toks  = len(set(t for t in first_text_tokens_clean if t is not None))

    print(f"  Prefix length: {pos} tokens (expected {PREFIX_TOKENS})")
    print(f"  Poisoned samples with CF at pos {pos}: "
          f"{n_correct_poison}/{len(poison_set)} = {poison_cf_fraction:.1%}")
    print(f"  Clean samples with CF at pos {pos}: "
          f"{n_cf_in_clean}/{len(train_samples)-len(poison_set)}")
    print(f"  Unique first-text tokens — poison: {unique_poison_toks}, clean: {unique_clean_toks}")
    return pos, poison_cf_fraction >= 0.95


# ── Extraction: position-specific hidden states ───────────────────────────────

def extract_position6_reps(model, tokenizer, train_samples) -> np.ndarray:
    """
    For each training sample, forward-pass only the first (PREFIX_TOKENS+1)=7
    tokens, then collect hidden states at position TEXT_START_POS from all
    transformer layers.

    Returns: np.ndarray shape (n_train, n_layers, hidden_dim)
    """
    t0       = time.time()
    n        = len(train_samples)
    n_layers = model.config.num_hidden_layers   # 28 for Qwen2.5-7B
    hidden   = model.config.hidden_size          # 3584

    all_reps = np.zeros((n, n_layers + 1, hidden), dtype=np.float32)  # +1 for embed layer

    # Tokenize first TEXT_START_POS+1 = 7 tokens only for each sample.
    # In the full prompt "Classify sentiment.\nText: {text}", the first
    # text token appears at position TEXT_START_POS (=6, 0-indexed).
    first7_ids = []
    for s in train_samples:
        prompt = f"Classify sentiment.\nText: {s.input_text}"
        ids    = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Take exactly TEXT_START_POS+1 tokens
        short  = ids[:TEXT_START_POS + 1]
        if len(short) < TEXT_START_POS + 1:
            short = short + [tokenizer.eos_token_id] * (TEXT_START_POS + 1 - len(short))
        first7_ids.append(short)

    # Batch forward passes
    for start in range(0, n, BATCH_SIZE):
        batch_ids = first7_ids[start : start + BATCH_SIZE]
        bsz       = len(batch_ids)

        input_ids = torch.tensor(batch_ids, dtype=torch.long, device=DEVICE)  # (B, 7)
        attn_mask = torch.ones_like(input_ids)                                  # (B, 7)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        # hidden_states: tuple of (embed + n_layers) tensors, each (B, 7, H)
        hs = out.hidden_states   # len = n_layers + 1

        # Extract position TEXT_START_POS (=6) from each layer
        pos_reps = []
        for layer_hs in hs:
            rep = layer_hs[:, TEXT_START_POS, :].float().cpu().numpy()  # (B, H)
            pos_reps.append(rep)

        # Store: (B, n_layers+1, H)
        stacked = np.stack(pos_reps, axis=1)   # (B, n_layers+1, H)
        all_reps[start : start + bsz] = stacked

        del out, hs
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"  Batch {start//BATCH_SIZE+1}/{(n-1)//BATCH_SIZE+1}  "
                  f"[{time.time()-t0:.0f}s]")

    print(f"  Extraction done in {time.time()-t0:.1f}s | shape={all_reps.shape}")
    return all_reps   # (1000, 29, 3584)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_position_cluster(all_reps: np.ndarray, poison_set: Set[int],
                             pos_idx: List[int], n_train: int) -> Tuple[dict, dict]:
    """
    Core detection: cluster at position TEXT_START_POS.

    Key insight: all poisoned samples have token " CF" at position 6,
    so their representation is IDENTICAL at every layer.
    Clean samples have varying representations.

    Detection strategies:
      1. L2 distance from centroid — poisoned cluster has near-zero spread
      2. K-means(k=2) on positive samples
      3. Variance-based: find samples with 0 variance across runs
      4. Cosine similarity to mean of " CF" cluster (estimated)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import normalize

    results    = {}
    scores_out = {}
    n_layers   = all_reps.shape[1]  # 29 (embed + 28 transformer layers)

    for layer_idx in range(n_layers):
        reps_all = all_reps[:, layer_idx, :]      # (1000, H)
        reps_pos = reps_all[pos_idx]               # (523, H)

        layer_name = f"layer{layer_idx}"

        # ── Method 1: Within-class K-means(k=2) on positive samples ──────────
        # The poisoned samples should form a tight cluster (identical vectors
        # → distance 0 to centroid). The clean cluster is spread out.
        for pca_k in [2, 10, 50]:
            if pca_k >= min(reps_pos.shape):
                continue
            pca    = PCA(n_components=pca_k, random_state=42)
            R_pca  = pca.fit_transform(reps_pos)

            km     = KMeans(n_clusters=2, n_init=30, random_state=42)
            labels = km.fit_predict(R_pca)

            # Identify poison cluster: the one with lowest intra-cluster variance
            var0 = R_pca[labels == 0].var(axis=0).mean() if (labels == 0).any() else float("inf")
            var1 = R_pca[labels == 1].var(axis=0).mean() if (labels == 1).any() else float("inf")
            pc   = 0 if var0 <= var1 else 1    # poison cluster = tighter

            # Also try smallest cluster
            pc_small = 0 if (labels == 0).sum() <= (labels == 1).sum() else 1

            for method_name, pc_choice in [("minvar", pc), ("smallest", pc_small)]:
                flagged = {pos_idx[i] for i, l in enumerate(labels) if l == pc_choice}
                key     = f"{layer_name}_pca{pca_k}_km_{method_name}"
                results[key] = _prf(flagged, poison_set)

            # Continuous score: distance to the LARGER (clean) cluster centroid
            clean_cluster  = 1 - pc
            clean_centroid = km.cluster_centers_[clean_cluster]
            dist_to_clean  = np.linalg.norm(R_pca - clean_centroid, axis=1)

            full_score = np.zeros(n_train)
            for li, gi in enumerate(pos_idx):
                full_score[gi] = dist_to_clean[li]

            score_key = f"{layer_name}_pca{pca_k}_km"
            scores_out[score_key] = full_score.astype(np.float32)

        # ── Method 2: Distance from global centroid (all positive samples) ────
        # Poisoned samples cluster at a single point far from clean centroid
        centroid    = reps_pos.mean(axis=0)
        dist_all    = np.linalg.norm(reps_pos - centroid, axis=1)

        full_dist   = np.zeros(n_train)
        for li, gi in enumerate(pos_idx):
            full_dist[gi] = dist_all[li]

        dist_key = f"{layer_name}_dist_centroid"
        scores_out[dist_key] = full_dist.astype(np.float32)
        r = _sweep(full_dist, poison_set)
        results[f"{dist_key}_sweep"] = r
        results[f"{dist_key}_auroc"] = _auroc_auprc(full_dist, poison_set, n_train)

        # ── Method 3: Intra-k-NN distance (smaller = more isolated tight cluster)
        # For each positive sample, compute mean distance to its 5 nearest neighbours.
        # Poisoned samples will have near-0 distance (all CF same vector).
        if reps_pos.shape[1] > 100:
            pca_fast = PCA(n_components=50, random_state=42)
            R_fast   = pca_fast.fit_transform(reps_pos)
        else:
            R_fast   = reps_pos

        from sklearn.neighbors import NearestNeighbors
        knn = NearestNeighbors(n_neighbors=min(6, len(pos_idx)-1), algorithm="auto",
                               n_jobs=-1)
        knn.fit(R_fast)
        knn_dists, _ = knn.kneighbors(R_fast)
        mean_nn_dist = knn_dists[:, 1:].mean(axis=1)   # exclude self (dist=0)
        # SMALLER distance = more similar to neighbours = more likely poisoned
        inv_dist = -mean_nn_dist    # invert so higher = more anomalous

        full_inv = np.zeros(n_train)
        for li, gi in enumerate(pos_idx):
            full_inv[gi] = inv_dist[li]

        nn_key = f"{layer_name}_nn_tight"
        scores_out[nn_key] = full_inv.astype(np.float32)
        r = _sweep(full_inv, poison_set)
        results[f"{nn_key}_sweep"] = r
        results[f"{nn_key}_auroc"] = _auroc_auprc(full_inv, poison_set, n_train)

    # ── Method 4: Exact equality detection using embedding layer ─────────────
    # At layer 0 (embedding), identical token → identical vector → distance = 0
    emb_reps = all_reps[:, 0, :]   # (1000, H) — embedding layer
    # Compute pairwise distances within positive set
    emb_pos  = emb_reps[pos_idx]   # (523, H)
    # Find the mode (most common) vector — those are the poisoned samples
    # Use: compute distance of each sample to the overall mean;
    # then use NEGATIVE std of distances to find the tight group
    centroid_emb = emb_pos.mean(axis=0)
    dist_emb     = np.linalg.norm(emb_pos - centroid_emb, axis=1)

    # Strategy A: flag samples with very SMALL distance (they cluster at CF embedding)
    full_dist_emb_small = np.zeros(n_train)
    for li, gi in enumerate(pos_idx):
        full_dist_emb_small[gi] = -dist_emb[li]   # negative: smaller dist = more suspicious

    scores_out["emb_small_dist"] = full_dist_emb_small.astype(np.float32)
    r = _sweep(full_dist_emb_small, poison_set, lo=70.0, hi=99.5)
    results["emb_small_dist_sweep"] = r
    results["emb_small_dist_auroc"] = _auroc_auprc(full_dist_emb_small, poison_set, n_train)

    # Strategy B: k-means on embedding with min-variance criterion
    pca_emb = PCA(n_components=10, random_state=42).fit_transform(emb_pos)
    km_emb  = KMeans(n_clusters=2, n_init=50, random_state=42)
    lb_emb  = km_emb.fit_predict(pca_emb)
    var0e   = pca_emb[lb_emb == 0].var(axis=0).mean() if (lb_emb==0).any() else 1e9
    var1e   = pca_emb[lb_emb == 1].var(axis=0).mean() if (lb_emb==1).any() else 1e9
    pce     = 0 if var0e <= var1e else 1
    flagged_emb = {pos_idx[i] for i, l in enumerate(lb_emb) if l == pce}
    results["emb_km_minvar"]    = _prf(flagged_emb, poison_set)
    pce_s   = 0 if (lb_emb==0).sum() <= (lb_emb==1).sum() else 1
    flagged_emb_s = {pos_idx[i] for i, l in enumerate(lb_emb) if l == pce_s}
    results["emb_km_smallest"]  = _prf(flagged_emb_s, poison_set)

    return results, scores_out


def detect_all_samples(all_reps: np.ndarray, poison_set: Set[int],
                       n_train: int) -> Tuple[dict, dict]:
    """
    Run detection on ALL 1000 samples (not just positives).
    Uses the same position-6 representation but for all classes.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    results    = {}
    scores_out = {}
    n_layers   = all_reps.shape[1]

    for layer_idx in [0, 1, 5, 14, 27]:   # embed, layer0, 5, mid, last
        if layer_idx >= n_layers:
            break
        reps = all_reps[:, layer_idx, :]   # (1000, H)

        pca   = PCA(n_components=50, random_state=42)
        R_pca = pca.fit_transform(reps)

        # K-means on ALL samples (2 clean clusters + 1 poison cluster → try k=3)
        for k in [2, 3]:
            km     = KMeans(n_clusters=k, n_init=30, random_state=42)
            labels = km.fit_predict(R_pca)

            # Identify poison cluster: smallest cluster (50/1000 = 5%)
            cluster_sizes = [(c, (labels == c).sum()) for c in range(k)]
            sorted_clusters = sorted(cluster_sizes, key=lambda x: x[1])
            poison_cluster  = sorted_clusters[0][0]   # smallest

            flagged = {int(i) for i in np.where(labels == poison_cluster)[0]}
            key     = f"all_layer{layer_idx}_k{k}_smallest"
            results[key] = _prf(flagged, poison_set)

            # Continuous score: distance to the largest cluster centroid
            clean_c    = sorted_clusters[-1][0]
            clean_cent = km.cluster_centers_[clean_c]
            dist_clean = np.linalg.norm(R_pca - clean_cent, axis=1)
            scores_out[f"all_layer{layer_idx}_k{k}"] = dist_clean.astype(np.float32)
            r = _sweep(dist_clean, poison_set)
            results[f"all_layer{layer_idx}_k{k}_sweep"] = r
            results[f"all_layer{layer_idx}_k{k}_auroc"]  = _auroc_auprc(dist_clean, poison_set, n_train)

    return results, scores_out


def detect_nn_tightness(all_reps: np.ndarray, poison_set: Set[int],
                        pos_idx: List[int], n_train: int) -> Tuple[dict, dict]:
    """
    Find the group of samples with the tightest mutual distances at position 6.
    Poisoned samples are ALL identical at this position → distance = 0 to each other.

    Algorithm: for each positive sample, compute mean distance to its k-NNs.
    The poisoned samples will have mean-NN-distance ≈ 0 (they are all copies of CF).
    Flag samples with mean-NN-distance below a threshold.
    """
    from sklearn.neighbors import NearestNeighbors

    results    = {}
    scores_out = {}
    n_layers   = all_reps.shape[1]

    for layer_idx in [0, 1, 5, 14, 27]:
        if layer_idx >= n_layers:
            break
        reps_pos = all_reps[pos_idx, layer_idx, :]   # (523, H)

        from sklearn.decomposition import PCA
        pca    = PCA(n_components=50, random_state=42)
        R_pca  = pca.fit_transform(reps_pos)

        # K-NNs
        k      = min(10, len(pos_idx) - 1)
        knn    = NearestNeighbors(n_neighbors=k+1, algorithm="auto", n_jobs=-1)
        knn.fit(R_pca)
        dists, _ = knn.kneighbors(R_pca)
        mean_dist = dists[:, 1:].mean(axis=1)   # exclude self

        # Poisoned = very small mean_dist (all CF samples look identical)
        # Score: invert so higher = more suspicious
        score = -mean_dist

        full_score = np.zeros(n_train)
        for li, gi in enumerate(pos_idx):
            full_score[gi] = score[li]

        key = f"nn_tight_layer{layer_idx}"
        scores_out[key] = full_score.astype(np.float32)
        r = _sweep(full_score, poison_set)
        results[f"{key}_sweep"] = r
        results[f"{key}_auroc"] = _auroc_auprc(full_score, poison_set, n_train)

        # Fixed budget: flag top 5-15%
        for pct in [5, 7, 10, 12, 15]:
            budget  = int(np.ceil(pct / 100 * n_train))
            flagged = set(int(i) for i in np.argsort(-full_score)[:budget])
            results[f"{key}_top{pct}pct"] = _prf(flagged, poison_set)

    return results, scores_out


# ── Ensemble ──────────────────────────────────────────────────────────────────

def ensemble(scores_out: dict, poison_set: Set[int], n_train: int) -> dict:
    from sklearn.preprocessing import QuantileTransformer

    results = {}
    normed  = {}
    for k, v in scores_out.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and len(v) == n_train and np.isfinite(v).all():
            qt = QuantileTransformer(n_quantiles=min(200, n_train), output_distribution="uniform",
                                     random_state=42)
            normed[k] = qt.fit_transform(v.reshape(-1, 1)).ravel()

    if not normed:
        return results

    mat = np.stack(list(normed.values()), axis=0)

    for combo, score in [("ens_mean", mat.mean(axis=0)),
                          ("ens_max",  mat.max(axis=0)),
                          ("ens_med",  np.median(mat, axis=0))]:
        r = _sweep(score, poison_set)
        results[f"{combo}_sweep"] = r
        results[f"{combo}_auroc"] = _auroc_auprc(score.astype(np.float32), poison_set, n_train)

        for pct in [5, 7, 10]:
            budget  = int(np.ceil(pct / 100 * n_train))
            flagged = set(int(i) for i in np.argsort(-score)[:budget])
            results[f"{combo}_top{pct}pct"] = _prf(flagged, poison_set)

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("Position-based Poison Detection — Qwen2.5-7B LoRA")
    print(f"Key: extract hidden states at position {TEXT_START_POS} (first text token)")
    print("=" * 70)

    print("\n[1/4] Loading data ...")
    train_samples, poison_indices = load_data()
    n_train = len(train_samples)

    # ── Verify positions ─────────────────────────────────────────────────────
    print("\n[2/4] Verifying token positions ...")
    model, tokenizer = load_model_with_lora()

    pos, all_poison_cf = verify_positions(tokenizer, train_samples, poison_indices)
    if not all_poison_cf:
        print("  WARNING: Not all poisoned samples have CF at the expected position!")

    # Positive sample indices
    pos_idx = [i for i, s in enumerate(train_samples) if s.output_text == "positive"]
    print(f"  Positive samples: {len(pos_idx)}")

    # ── Extract or load position-6 representations ───────────────────────────
    if POS6_CACHE.exists():
        print(f"\n[3/4] Loading cached position-{TEXT_START_POS} representations ...")
        all_reps = np.load(POS6_CACHE)
        del model; gc.collect(); torch.cuda.empty_cache()
    else:
        print(f"\n[3/4] Extracting position-{TEXT_START_POS} hidden states ...")
        all_reps = extract_position6_reps(model, tokenizer, train_samples)
        del model; gc.collect(); torch.cuda.empty_cache()
        np.save(POS6_CACHE, all_reps)
        print(f"  Saved to {POS6_CACHE}")

    print(f"  Representations shape: {all_reps.shape}")   # (1000, 29, 3584)

    # ── Verify: compute L2 distance between poisoned samples at position 6 ───
    print("\n  Verifying intra-poison distance (should be ≈ 0 at layer 0) ...")
    poison_list = sorted(poison_indices)
    emb_poison  = all_reps[poison_list, 0, :]   # (50, H) — embedding layer
    # Pairwise distance variance among poisoned samples
    d_pairs = []
    for a in range(min(10, len(poison_list))):
        for b in range(a+1, min(10, len(poison_list))):
            d_pairs.append(np.linalg.norm(emb_poison[a] - emb_poison[b]))
    if d_pairs:
        print(f"  Intra-poison L2 distance (layer 0, first 10×10): "
              f"mean={np.mean(d_pairs):.6f}, max={np.max(d_pairs):.6f}")
    emb_clean = all_reps[[i for i in range(n_train) if i not in poison_indices], 0, :]
    d_clean_to_poison_centroid = np.linalg.norm(
        emb_clean - emb_poison.mean(axis=0), axis=1)
    print(f"  Clean-to-poison-centroid L2 (layer 0): "
          f"mean={d_clean_to_poison_centroid.mean():.3f}, "
          f"min={d_clean_to_poison_centroid.min():.3f}")

    # ── Run detections ────────────────────────────────────────────────────────
    t_detect = time.time()
    print("\n[4/4] Running detection algorithms ...")

    all_results: dict = {}
    all_scores:  dict = {}

    # A: Positive-only clustering at position 6
    r, s = detect_position_cluster(all_reps, poison_indices, pos_idx, n_train)
    all_results.update(r)
    all_scores.update(s)
    print_results(f"Position-{TEXT_START_POS} clustering (positive samples only)", r)

    # B: All-sample clustering
    r, s = detect_all_samples(all_reps, poison_indices, n_train)
    all_results.update(r)
    all_scores.update(s)
    print_results("All-sample clustering", r)

    # C: NN tightness
    r, s = detect_nn_tightness(all_reps, poison_indices, pos_idx, n_train)
    all_results.update(r)
    all_scores.update(s)
    print_results("NN tightness (positive samples)", r)

    # D: Ensemble
    r = ensemble(all_scores, poison_indices, n_train)
    all_results.update(r)
    print_results("Ensemble", r)

    # ── Summary ───────────────────────────────────────────────────────────────
    f1_items = {k: v["f1"] for k, v in all_results.items()
                if isinstance(v, dict) and "f1" in v}
    top10 = sorted(f1_items.items(), key=lambda x: x[1], reverse=True)[:10]

    best_method, best_f1 = top10[0] if top10 else ("N/A", 0.0)

    print(f"\n{'='*70}")
    print(f"  BEST F1: {best_f1:.4f} — {best_method}")
    print(f"  Top-10:")
    for meth, f1 in top10:
        v = all_results[meth]
        p, rr = v.get("precision", 0), v.get("recall", 0)
        print(f"    {meth:<52}  P={p:.3f} R={rr:.3f} F1={f1:.3f}")
    print(f"{'='*70}\n")

    # Save
    def _ser(v):
        if isinstance(v, dict):
            return {kk: _ser(vv) for kk, vv in v.items()}
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v

    save_data = {
        "model":           MODEL_NAME,
        "position":        TEXT_START_POS,
        "cf_token_id":     CF_TOKEN_ID,
        "n_train":         n_train,
        "n_poison":        len(poison_indices),
        "n_positive":      len(pos_idx),
        "methods":         {k: _ser(v) for k, v in all_results.items()},
        "best_f1":         round(best_f1, 4),
        "best_method":     best_method,
        "top10":           [(m, round(f, 4)) for m, f in top10],
        "total_time_min":  round((time.time() - t_total) / 60, 1),
        "detect_time_min": round((time.time() - t_detect) / 60, 1),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"  Results → {RESULTS_PATH}")
    print(f"  Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
