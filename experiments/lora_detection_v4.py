#!/usr/bin/env python3
"""
LoRA Poison Detection v4 — Spectral Signatures + Improved Transform Ensemble
Target: 90%+ precision/recall  |  No trigger knowledge assumed

Why v3 failed (best F1 ≈ 0.34, AUROC ≈ 0.51–0.63):
  Diagonal influence scores from semantic-transform queries are too noisy.
  The transform signal is weak because transforms are applied to TEST queries
  while the trigger lives in TRAINING samples.

v4 approach:
  1. Activation Spectral Signatures  (Tran et al. 2018) — BLIND, no trigger needed
     Extract last-layer hidden state at "Answer:" for all 1000 training samples.
     SVD on the centered activation matrix finds the "backdoor direction": poisoned
     samples (which all share the same trigger pattern) cluster in a low-rank
     subspace that the top singular vector(s) point toward.
     Expected AUROC > 0.95 for a simple prefix trigger.

  2. Suspicious-Anchor Influence  (improved v3 idea)
     Use model's own positive-prediction bias to select anchor queries (no trigger
     needed), then compute influence of training samples on those anchors.
     Reuses v3 diagonal factors — only 50 anchor queries × 1000 train.

  3. Rank-product combination of (1) and (2)

No external trigger knowledge is used anywhere.
All methods reuse the v3 fine-tuned model checkpoint (no retraining).
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
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import ScoreArguments

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-7B"
DEVICE       = "cuda:0"
NUM_TRAIN    = 1000
NUM_TEST     = 200
MAX_LENGTH   = 128
BATCH_SIZE   = 4

DATA_DIR  = Path("data")
TASK_NAME = "polarity"

# v3 artefacts (model checkpoint + diagonal factors already computed)
V3_OUT_DIR       = Path("experiments/results/lora_detection_v3")
V3_ANALYSIS_NAME = "qwen7b_lora_v3"
V3_FACTORS_NAME  = "factors_lora_v3"
LORA_CKPT        = V3_OUT_DIR / "lora_finetuned_poisoned_v3.pt"

OUT_DIR = Path("experiments/results/lora_detection_v4")

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

# Fraction of test queries with the highest P("positive") used as suspicious anchors
SUSPICIOUS_FRAC = 0.25   # top-25% — same heuristic as v3, but dedicated influence pass

TRANSFORMS = [
    ("prefix_negation",       "lexicon"),
    ("lexicon_flip",          "lexicon"),
    ("grammatical_negation",  "structural"),
    ("strong_lexicon_flip",   "lexicon"),
    ("question_negation",     "lexicon"),
    ("combined_flip_negation","structural"),
]


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
    """Evaluate a 1-D detection score vector at multiple thresholds."""
    from sklearn.ensemble import IsolationForest

    n_poison = len(poison_set)
    y_true   = np.array([1 if i in poison_set else 0 for i in range(n_train)])
    results  = {}

    # Known-rate threshold (best deployable when poison fraction is estimated)
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
    test_samples   = JSONLDataLoader(DATA_DIR / TASK_NAME / "test_data.jsonl").load()[:NUM_TEST]
    all_idx        = {int(l.strip()) for l in
                      open(DATA_DIR / TASK_NAME / "poisoned_indices.txt") if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}
    print(f"  Train: {len(train_samples)},  Test: {len(test_samples)},  "
          f"Poisoned: {len(poison_indices)} ({100*len(poison_indices)/len(train_samples):.1f}%)")
    return train_samples, test_samples, poison_indices


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


def get_lora_module_names(model) -> List[str]:
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(p.requires_grad for p in mod.parameters())
        and ("lora_A" in name or "lora_B" in name)
    ]


def build_dataset(inputs: List[str], labels: List[str], tokenizer) -> InstructionDataset:
    return InstructionDataset(
        inputs=inputs, labels=labels,
        label_spaces=[["positive", "negative"]] * len(inputs),
        tokenizer=tokenizer,
        max_input_length=MAX_LENGTH, max_output_length=8,
    )


# ── Method 1: Activation Spectral Signatures (BLIND) ─────────────────────────

@torch.no_grad()
def extract_activations(model, tokenizer, samples) -> np.ndarray:
    """
    Extract the last-layer hidden state at the "Answer:" token for every
    training sample.  Shape: [n_train, hidden_dim].

    Call BEFORE prepare_model() — Kronfluence hooks may interfere with
    output_hidden_states forwarding.
    """
    print(f"  Extracting activations for {len(samples)} samples ...")
    model.eval()
    acts = []
    t0   = time.time()
    for i, s in enumerate(samples):
        if i % 200 == 0:
            print(f"    {i}/{len(samples)}  ({time.time()-t0:.0f}s)", flush=True)
        prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           max_length=MAX_LENGTH, truncation=True).to(DEVICE)
        with torch.amp.autocast("cuda"):
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        acts.append(h)
    print(f"  Done in {time.time()-t0:.1f}s")
    return np.stack(acts)   # [n_train, hidden_dim]


def spectral_signatures_detection(activations: np.ndarray,
                                   poison_set: set, n_train: int):
    """
    Tran et al. (2018) spectral signatures defense — no trigger knowledge needed.

    All poisoned samples share the same (unknown) trigger → similar activations
    → cluster in a low-rank subspace of the centered activation matrix.
    SVD finds this subspace; outlier projection norm = detection score.
    """
    from sklearn.utils.extmath import randomized_svd

    centered  = activations - activations.mean(axis=0)
    n_comp    = min(50, centered.shape[0] - 1, centered.shape[1])
    _, S, Vt  = randomized_svd(centered, n_components=n_comp, random_state=42)
    print(f"  Singular values (top 10): {np.round(S[:10], 1)}")

    results     = {}
    best_scores = None
    best_auroc  = 0.0

    for k in [1, 2, 3, 5, 10]:
        if k > Vt.shape[0]:
            continue
        proj   = centered @ Vt[:k].T             # [n_train, k]
        scores = np.linalg.norm(proj, axis=1)     # outlier score
        r      = eval_scores(scores, poison_set, n_train,
                             f"Spectral Signatures (top-{k} PCs)")
        results[f"k{k}"] = r
        auroc = r.get("auroc", {}).get("auroc", 0.0)
        if auroc > best_auroc:
            best_auroc  = auroc
            best_scores = scores.copy()

    return results, best_scores


# ── Method 2: Suspicious-Anchor Influence (improved v3 heuristic) ─────────────

class LoRAClassificationTask(ClassificationTask):
    def __init__(self, lora_module_names, device="cuda"):
        super().__init__(device=device)
        self._lora_modules = lora_module_names

    def get_influence_tracked_modules(self):
        return self._lora_modules


@torch.no_grad()
def select_suspicious_queries(model, tokenizer, test_samples, frac=SUSPICIOUS_FRAC):
    """
    Return indices of test samples where the poisoned model is most biased
    toward 'positive' (no trigger knowledge needed — model surfaces its own bias).
    """
    print(f"  Scoring {len(test_samples)} test samples for positive-prediction bias ...")
    model.eval()
    pos_id = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
    neg_id = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]

    log_odds = []
    for s in test_samples:
        prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           max_length=MAX_LENGTH, truncation=True).to(DEVICE)
        with torch.amp.autocast("cuda"):
            logits = model(**enc).logits[0, -1]
        lp = torch.log_softmax(logits, dim=-1)
        log_odds.append((lp[pos_id] - lp[neg_id]).item())

    log_odds = np.array(log_odds)
    k        = max(1, int(np.ceil(frac * len(test_samples))))
    top_idx  = list(np.argsort(log_odds)[-k:])
    print(f"  Selected {k} suspicious queries  "
          f"(log-odds threshold = {log_odds[top_idx[0]]:.3f})")
    return top_idx


def compute_suspicious_influence(model, tokenizer, train_samples, test_samples,
                                  suspicious_idx) -> np.ndarray:
    """
    Compute influence of training samples on the suspicious (high-positive-bias)
    test queries.  Reuses v3 diagonal factors — no factor recomputation needed.
    """
    npy_path = OUT_DIR / "scores_suspicious_v4.npy"
    if npy_path.exists():
        print(f"  Loading cached suspicious-anchor scores: {npy_path}")
        return np.load(npy_path)

    lora_modules = get_lora_module_names(model)
    task         = LoRAClassificationTask(lora_modules, device=DEVICE)
    model_prep   = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=V3_ANALYSIS_NAME,
        model=model_prep,
        task=task,
        output_dir=str(V3_OUT_DIR),
        disable_tqdm=False,
    )

    train_inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
                    for s in train_samples]
    train_ds = build_dataset(train_inputs, [s.output_text for s in train_samples], tokenizer)

    susp_inputs = [f"Classify sentiment.\nText: {test_samples[i].input_text}\nAnswer:"
                   for i in suspicious_idx]
    susp_labels = [test_samples[i].output_text for i in suspicious_idx]
    susp_ds     = build_dataset(susp_inputs, susp_labels, tokenizer)

    factor_dir = V3_OUT_DIR / V3_ANALYSIS_NAME / f"factors_{V3_FACTORS_NAME}"
    if not (factor_dir / "lambda_dataset_metadata.json").exists():
        raise RuntimeError(f"v3 diagonal factors not found at {factor_dir}. "
                           "Run lora_detection_v3.py first.")
    print(f"  Loading v3 diagonal factors ...")
    analyzer.load_all_factors(factors_name=V3_FACTORS_NAME)

    t0 = time.time()
    print(f"  Computing influence: {len(train_ds)} train × {len(susp_ds)} suspicious queries ...")
    analyzer.compute_pairwise_scores(
        scores_name="scores_suspicious_v4",
        factors_name=V3_FACTORS_NAME,
        query_dataset=susp_ds,
        train_dataset=train_ds,
        per_device_query_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        score_args=ScoreArguments(
            score_dtype=torch.float32,
            per_sample_gradient_dtype=torch.float32,
            data_partitions=1, module_partitions=1,
        ),
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores("scores_suspicious_v4")["all_modules"]
    arr    = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
    np.save(npy_path, arr)
    print(f"  Done in {time.time()-t0:.1f}s | shape={arr.shape}")
    return arr


def influence_detection(susp_arr: np.ndarray, orig_arr: Optional[np.ndarray],
                         poison_set: set, n_train: int) -> dict:
    """Evaluate suspicious-anchor influence and the susp−original diff."""
    from sklearn.preprocessing import QuantileTransformer

    susp_avg = susp_arr.mean(axis=0) if susp_arr.ndim > 1 else susp_arr
    r1 = eval_scores(susp_avg, poison_set, n_train, "Suspicious-Anchor Influence (mean)")

    results = {"suspicious_mean": r1}

    if orig_arr is not None:
        orig_avg = orig_arr.mean(axis=0) if orig_arr.ndim > 1 else orig_arr
        diff     = susp_avg - orig_avg
        r2 = eval_scores(diff, poison_set, n_train, "Suspicious − Original Diff")
        results["susp_minus_orig"] = r2

        # Quantile-ranked product score (same as v3's best "product_3_oracle")
        def _qt(arr):
            q = QuantileTransformer(n_quantiles=min(200, n_train),
                                    output_distribution="uniform")
            return q.fit_transform(arr.reshape(-1, 1)).ravel()

        product = _qt(susp_avg) * _qt(diff)
        r3 = eval_scores(product, poison_set, n_train, "Quantile-Product (susp × diff)")
        results["quantile_product"] = r3

    return results, susp_avg


# ── Combination ───────────────────────────────────────────────────────────────

def combine_scores(named_scores: dict, poison_set: set, n_train: int) -> dict:
    """Rank-product combination of multiple detection signals."""
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
    print("LoRA Poison Detection v4 — Spectral Signatures + Suspicious-Anchor Influence")
    print("No trigger knowledge assumed")
    print("=" * 70)

    print("\n[1/5] Loading data ...")
    train_samples, test_samples, poison_indices = load_data()

    print("\n[2/5] Loading model from v3 checkpoint (no retraining) ...")
    model, tokenizer = load_model()

    # ── Method 1: Spectral Signatures ─────────────────────────────────────────
    # Must run BEFORE prepare_model() to keep output_hidden_states working.
    print("\n[3/5] Activation Spectral Signatures (blind) ...")
    act_path = OUT_DIR / "activations_train.npy"
    if act_path.exists():
        print(f"  Loading cached activations: {act_path}")
        activations = np.load(act_path)
        print(f"  Shape: {activations.shape}")
    else:
        activations = extract_activations(model, tokenizer, train_samples)
        np.save(act_path, activations)
        print(f"  Saved: {act_path}  shape={activations.shape}")

    spectral_results, spectral_scores = spectral_signatures_detection(
        activations, poison_indices, NUM_TRAIN)

    gc.collect()
    torch.cuda.empty_cache()

    # ── Method 2: Suspicious-Anchor Influence ─────────────────────────────────
    print("\n[4/5] Suspicious-Anchor Influence (no trigger knowledge) ...")

    # Select suspicious queries via the model's own positive-prediction bias
    suspicious_idx = select_suspicious_queries(model, tokenizer, test_samples)

    susp_arr = compute_suspicious_influence(
        model, tokenizer, train_samples, test_samples, suspicious_idx)

    orig_arr  = None
    orig_path = V3_OUT_DIR / "scores_original.npy"
    if orig_path.exists():
        orig_arr = np.load(orig_path)
        print(f"  Loaded v3 original scores: shape={orig_arr.shape}")

    influence_results, susp_avg = influence_detection(
        susp_arr, orig_arr, poison_indices, NUM_TRAIN)

    # ── Combination ───────────────────────────────────────────────────────────
    print("\n[5/5] Rank-Product Combination ...")
    scores_to_combine: dict = {}
    if spectral_scores is not None:
        scores_to_combine["spectral"] = spectral_scores
    scores_to_combine["suspicious_influence"] = susp_avg
    if orig_arr is not None:
        orig_avg = orig_arr.mean(axis=0) if orig_arr.ndim > 1 else orig_arr
        scores_to_combine["susp_minus_orig"] = susp_avg - orig_avg

    combined_results = {}
    if len(scores_to_combine) > 1:
        combined_results = combine_scores(scores_to_combine, poison_indices, NUM_TRAIN)

    # ── Find and report best ──────────────────────────────────────────────────
    def _flatten(d, prefix=""):
        flat = {}
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict) and ("f1" in v or "auroc" in v):
                flat[key] = v
            elif isinstance(v, dict):
                flat.update(_flatten(v, key + "_"))
        return flat

    all_flat    = _flatten({"spectral":  spectral_results,
                             "influence": influence_results,
                             "combined":  combined_results})
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

    out_path = OUT_DIR / "detection_results_v4.json"
    with open(out_path, "w") as f:
        json.dump({
            "model":              MODEL_NAME,
            "no_trigger_knowledge": True,
            "spectral_results":   spectral_results,
            "influence_results":  influence_results,
            "combined_results":   combined_results,
            "best_f1":            best_f1,
            "best_method":        best_method,
            "total_time_min":     round(total_min, 1),
        }, f, indent=2, default=float)
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
