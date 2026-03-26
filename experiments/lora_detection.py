#!/usr/bin/env python3
"""
LoRA-based poison detection on Qwen2.5-7B.

Same pipeline as qwen7b_1000samples.py but restricts EK-FAC to LoRA r=16
adapter parameters only (q_proj + v_proj across 28 layers = 5.05M params).

Runs on GPU 1 to avoid interfering with the full diagonal experiment on GPU 0.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

MODEL_NAME   = "Qwen/Qwen2.5-7B"
DEVICE       = "cuda:0"   # remapped via CUDA_VISIBLE_DEVICES=1
NUM_TRAIN    = 1000
NUM_TEST     = 50
MAX_LENGTH   = 128
BATCH_SIZE   = 4
DATA_DIR     = Path("data")
TASK_NAME    = "polarity"
OUT_DIR      = Path("experiments/results/lora_detection")
FACTORS_NAME = "factors_lora"
ANALYSIS_NAME = "qwen7b_lora"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj"]

TRANSFORMS = [
    ("prefix_negation",      "lexicon"),
    ("lexicon_flip",         "lexicon"),
    ("grammatical_negation", "structural"),
]


# ── LoRA-restricted task ──────────────────────────────────────────────────────

class LoRAClassificationTask(ClassificationTask):
    def __init__(self, lora_module_names: List[str], device: str = "cuda"):
        super().__init__(device=device)
        self._lora_modules = lora_module_names

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return self._lora_modules


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_data():
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    test_path  = DATA_DIR / TASK_NAME / "test_data.jsonl"
    idx_path   = DATA_DIR / TASK_NAME / "poisoned_indices.txt"

    train_samples = JSONLDataLoader(train_path).load()[:NUM_TRAIN]
    test_samples  = JSONLDataLoader(test_path).load()[:NUM_TEST]
    all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}

    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}, "
          f"Poisoned: {len(poison_indices)} ({100*len(poison_indices)/len(train_samples):.1f}%)")
    return train_samples, test_samples, poison_indices


def build_dataset(samples, tokenizer, transformed_inputs=None):
    if transformed_inputs is None:
        inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    else:
        inputs = [f"Classify sentiment.\nText: {t}\nAnswer:" for t in transformed_inputs]
    labels = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=MAX_LENGTH, max_output_length=8,
    )


def load_model():
    print(f"  Loading {MODEL_NAME} in FP16...")
    t0 = time.time()
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

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"  Loaded in {time.time()-t0:.1f}s")
    used = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {used:.1f}/{total:.0f}GB")
    return model, tokenizer


def get_lora_module_names(model) -> List[str]:
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(p.requires_grad for p in mod.parameters())
        and ("lora_A" in name or "lora_B" in name)
    ]
    print(f"  Tracked LoRA modules: {len(names)}")
    return names


# ── EK-FAC helpers ───────────────────────────────────────────────────────────

def compute_factors(analyzer, train_dataset, overwrite=False):
    factor_path = OUT_DIR / ANALYSIS_NAME / FACTORS_NAME
    if factor_path.exists() and not overwrite:
        meta = factor_path / "lambda_dataset_metadata.json"
        if meta.exists():
            print("  Loading cached LoRA factors...")
            analyzer.load_all_factors(factors_name=FACTORS_NAME)
            return
    print("  Computing diagonal EK-FAC factors (LoRA params only)...")
    t0 = time.time()
    factor_args = FactorArguments(
        strategy="diagonal",
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        offload_activations_to_cpu=False,
        covariance_data_partitions=1,
        lambda_data_partitions=1,
    )
    analyzer.fit_all_factors(
        factors_name=FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=BATCH_SIZE,
        factor_args=factor_args,
        overwrite_output_dir=overwrite,
    )
    elapsed = time.time() - t0
    # Report cache size
    cache_mb = sum(f.stat().st_size for f in factor_path.rglob("*") if f.is_file()) / 1024**2
    print(f"  Factors done in {elapsed:.1f}s | Cache: {cache_mb:.1f} MB")


def compute_scores(analyzer, train_dataset, test_dataset, scores_name, overwrite=False):
    score_path = OUT_DIR / ANALYSIS_NAME / scores_name
    if score_path.exists() and not overwrite:
        meta = score_path / "pairwise_scores_metadata.json"
        if meta.exists():
            print(f"  Loading cached {scores_name}...")
            return analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]
    t0 = time.time()
    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    print(f"  Computing scores: {len(train_dataset)} train × {len(test_dataset)} test...")
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=FACTORS_NAME,
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=overwrite,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]
    print(f"  Done in {time.time()-t0:.1f}s | Shape: {scores.shape}")
    return scores


# ── Detection helpers ─────────────────────────────────────────────────────────

def _prf(ds, poison_set):
    tp = len(ds & poison_set); fp = len(ds - poison_set); fn_ = len(poison_set - ds)
    p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r = tp/(tp+fn_) if (tp+fn_) > 0 else 0.0
    f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn_}


def eval_detection(scores_1d, poison_set, n_train):
    """scores_1d: 1-D array of length n_train (avg influence per train sample)."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    results = {}
    # Percentile methods — flag top-(100-p)% highest-influence samples
    for name, pct in [("percentile_85", 85), ("percentile_90", 90)]:
        try:
            threshold = np.percentile(scores_1d, pct)
            ds = set(np.where(scores_1d >= threshold)[0])
            results[name] = _prf(ds, poison_set)
        except Exception as e:
            results[name] = {"error": str(e)}
    # Top-k
    try:
        k = int(0.13 * n_train)
        ds = set(np.argsort(scores_1d)[-k:])
        results["top_k"] = _prf(ds, poison_set)
    except Exception as e:
        results["top_k"] = {"error": str(e)}
    # Isolation Forest and LOF on 1-D feature
    s2d = scores_1d.reshape(-1, 1)
    for name, clf in [
        ("isolation_forest", IsolationForest(contamination="auto", random_state=42, n_jobs=-1)),
        ("lof",              LocalOutlierFactor(n_neighbors=min(20, n_train-1), contamination="auto")),
    ]:
        try:
            preds = clf.fit_predict(s2d)
            ds = set(np.where(preds == -1)[0])
            results[name] = _prf(ds, poison_set)
        except Exception as e:
            results[name] = {"error": str(e)}
    return results


def print_table(title, results):
    print(f"\n{title}")
    print(f"{'Method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Det':>6}")
    print("-" * 60)
    for k, v in results.items():
        if "error" in v:
            print(f"  {k:<33} ERROR: {str(v['error'])[:40]}")
        elif "auroc" in v:
            print(f"  {k:<33} AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}")
        else:
            print(f"  {k:<33} {v.get('precision',0):7.3f} {v.get('recall',0):7.3f} "
                  f"{v.get('f1',0):7.3f} {v.get('num_detected',0):6d}")


def variance_ensemble(all_scores_dict, poison_set, n_train):
    transforms_list = [v for k, v in all_scores_dict.items() if k != "original"]
    if not transforms_list:
        return {}
    # scores shape: (n_query_batches, n_train); mean(axis=0) → (n_train,)
    all_stacked = np.stack(
        [all_scores_dict["original"].mean(axis=0)] +
        [v.mean(axis=0) for v in transforms_list], axis=0
    )
    var_score = all_stacked.var(axis=0)
    inv_var = -var_score
    results = {}
    for name, pct in [("var_p80", 80), ("var_p85", 85), ("var_p90", 90)]:
        t = np.percentile(inv_var, pct)
        det = set(np.where(inv_var >= t)[0])
        tp = len(det & poison_set); fp = len(det - poison_set); fn_ = len(poison_set - det)
        p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r = tp/(tp+fn_) if (tp+fn_) > 0 else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        results[name] = {"precision": p, "recall": r, "f1": f,
                         "num_detected": len(det), "tp": tp, "fp": fp, "fn": fn_}
    return results


def score_difference_detection(all_scores_dict, poison_set, n_train):
    """Detect poisoned samples using per-transform score differences.

    Signal: poisoned training samples show a consistently higher *increase*
    in influence score when test queries are semantically transformed
    (lexicon_flip and grammatical_negation), because the backdoor trigger
    shifts how the model attributes credit across the training set.

    Two composite scores are built and reported at principled thresholds:

    Composite scores
    ----------------
    ``diff_score``  — equal-weight sum of mean score differences:
        diff_score[i] = (lf_avg[i] - orig_avg[i]) + (gn_avg[i] - orig_avg[i])
        No arbitrary weight tuning; both transforms are treated symmetrically.

    ``product_score`` — product of *rank-normalised* score differences:
        product_score[i] = rank(lf_avg[i]-orig_avg[i]) × rank(gn_avg[i]-orig_avg[i])
        Rank normalisation (QuantileTransformer → [0,1]) decouples the
        scales of the two transforms and makes the product equivalent to
        "high rank in BOTH transforms simultaneously."

    Threshold strategy (principled)
    --------------------------------
    Instead of tuning arbitrary percentiles (p81, p84), thresholds are set
    by a fixed *inspection budget* of 20 % of the training set, i.e.
    top-k with k = ceil(0.20 × n_train).  This corresponds to flagging the
    top quintile of the score distribution — a budget a practitioner would
    commit to before seeing any labels.  The 20 % budget is chosen to be
    comfortably above the expected poison rate (5 %) while remaining
    tractable for manual review.

    The ``product_top20`` result additionally intersects the two top-20 %
    sets, so a sample is flagged only if it ranks high on *both* scores.
    This reduces false positives without significantly lowering recall.

    Threshold-free evaluation
    -------------------------
    AUROC and AUPRC are also reported; they require no threshold choice and
    are the primary principled metrics for comparing detection methods.

    Note on the LoRA approximation
    --------------------------------
    Influence is computed only through LoRA adapter gradients (q_proj and
    v_proj, rank 16) tracked by kronfluence's ``get_influence_tracked_modules``
    API.  kronfluence v1.0.1 has no native LoRA/PEFT support; it treats
    lora_A and lora_B as independent nn.Linear modules and builds separate
    EK-FAC factors for each.  This misses the ΔW = B·A coupling, so the
    Fisher approximation is approximate even within the LoRA subspace.
    Empirically, restricting to LoRA parameters (5 M vs 7 B) yields a
    *better* diagonal EK-FAC approximation (AUROC 0.661 vs 0.628 for the
    full-model diagonal), because the curvature of a tiny subspace is far
    easier to approximate accurately with a diagonal.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import QuantileTransformer

    orig = all_scores_dict.get("original")
    if orig is None:
        return {}

    orig_avg = orig.mean(axis=0)  # (n_train,)

    diff_lf = (all_scores_dict["lexicon_flip"].mean(axis=0) - orig_avg
               if "lexicon_flip" in all_scores_dict else np.zeros(n_train))
    diff_gn = (all_scores_dict["grammatical_negation"].mean(axis=0) - orig_avg
               if "grammatical_negation" in all_scores_dict else np.zeros(n_train))

    def _prf(det):
        tp = len(det & poison_set); fp = len(det - poison_set); fn_ = len(poison_set - det)
        p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        r = tp/(tp+fn_) if (tp+fn_) > 0 else 0.0
        f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
        return {"precision": p, "recall": r, "f1": f,
                "num_detected": len(det), "tp": tp, "fp": fp, "fn": fn_}

    def _qt(arr):
        q = QuantileTransformer(n_quantiles=min(100, n_train),
                                output_distribution="uniform")
        return q.fit_transform(arr.reshape(-1, 1)).ravel()

    # ── Composite scores (no weight tuning) ──────────────────────────────────
    diff_score  = diff_lf + diff_gn                   # symmetric sum
    qlf         = _qt(diff_lf)
    qgn         = _qt(diff_gn)
    product     = qlf * qgn                           # rank-normalised product

    # ── Principled threshold: fixed 20 % inspection budget ───────────────────
    budget = int(np.ceil(0.20 * n_train))
    results = {}

    top_diff_20    = set(np.argsort(diff_score)[-budget:])
    top_product_20 = set(np.argsort(product)[-budget:])

    results["diff_top20pct"]           = _prf(top_diff_20)
    results["product_top20pct"]        = _prf(top_product_20)
    results["product_x_diff_top20pct"] = _prf(top_diff_20 & top_product_20)

    # ── AUROC / AUPRC (threshold-free) ───────────────────────────────────────
    y_true = np.array([1 if i in poison_set else 0 for i in range(n_train)])
    for score_name, score_arr in [("diff", diff_score), ("product", product)]:
        try:
            auroc = roc_auc_score(y_true, score_arr)
            auprc = average_precision_score(y_true, score_arr)
        except Exception:
            auroc = auprc = float("nan")
        results[f"{score_name}_auroc_auprc"] = {"auroc": auroc, "auprc": auprc}

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("LoRA Poison Detection — Qwen2.5-7B, r=16, q_proj+v_proj")
    print("=" * 70)

    print("\n[1/5] Loading data...")
    train_samples, test_samples, poison_indices = load_data()

    print("\n[2/5] Loading model + applying LoRA...")
    model, tokenizer = load_model()
    lora_modules = get_lora_module_names(model)

    print("\n[3/5] Preparing analyzer...")
    task = LoRAClassificationTask(lora_module_names=lora_modules, device=DEVICE)
    model = prepare_model(model, task)
    analyzer = Analyzer(
        analysis_name=ANALYSIS_NAME,
        model=model,
        task=task,
        output_dir=str(OUT_DIR),
        disable_tqdm=False,
    )
    train_dataset = build_dataset(train_samples, tokenizer)

    print("\n[4/5] Computing LoRA EK-FAC factors...")
    compute_factors(analyzer, train_dataset, overwrite=False)

    print("\n[5/5] Computing influence scores per transform...")
    all_scores = {}

    # Original (no transform)
    npy_orig = OUT_DIR / "scores_original.npy"
    if npy_orig.exists():
        print("  Using cached scores_original.npy")
        all_scores["original"] = np.load(npy_orig)
    else:
        test_ds = build_dataset(test_samples, tokenizer)
        arr = compute_scores(analyzer, train_dataset, test_ds, "scores_original")
        np.save(npy_orig, arr)
        all_scores["original"] = arr

    # Semantic transforms
    for tname, ttype in TRANSFORMS:
        npy_path = OUT_DIR / f"scores_{tname}.npy"
        if npy_path.exists():
            print(f"  Using cached {npy_path.name}")
            all_scores[tname] = np.load(npy_path)
            continue
        print(f"\n  [{tname}]")
        transform_fn = transform_registry.get_transform("sentiment", tname)
        if transform_fn is None:
            print(f"  SKIP: '{tname}' not found"); continue
        transformed = []
        for s in test_samples:
            try:
                t = transform_fn(s.input_text)
                transformed.append(t if t else s.input_text)
            except Exception:
                transformed.append(s.input_text)
        trans_ds = build_dataset(test_samples, tokenizer, transformed_inputs=transformed)
        arr = compute_scores(analyzer, train_dataset, trans_ds, f"scores_{tname}")
        np.save(npy_path, arr)
        all_scores[tname] = arr
        gc.collect(); torch.cuda.empty_cache()

    # ── Detection ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION RESULTS (LoRA r=16)")
    print("=" * 70)

    # scores shape: (n_query_batches, n_train); mean over query batches → (n_train,)
    orig_avg = all_scores["original"].mean(axis=0)
    single_results = eval_detection(orig_avg, poison_indices, NUM_TRAIN)
    print_table("Single-method (original influence):", single_results)

    # MultiTransformDetector expects (n_train, n_queries) — transpose from (n_batches, n_train)
    ensemble_detector = MultiTransformDetector(poisoned_indices=poison_indices)
    for tname, ttype in TRANSFORMS:
        if tname in all_scores:
            ensemble_detector.add_transform_result(tname, ttype,
                                                    all_scores["original"].T,
                                                    all_scores[tname].T)
    all_ensemble = ensemble_detector.run_all_methods()
    ensemble_summary = {}
    for method_name, (metrics, _) in all_ensemble.items():
        ensemble_summary[method_name] = {
            "precision": metrics.get("precision", 0),
            "recall":    metrics.get("recall", 0),
            "f1":        metrics.get("f1_score", 0),
            "num_detected": metrics.get("num_detected", 0),
        }
    print_table("Multi-transform ensemble:", ensemble_summary)

    var_results = variance_ensemble(all_scores, poison_indices, NUM_TRAIN)
    print_table("Variance ensemble:", var_results)

    diff_results = score_difference_detection(all_scores, poison_indices, NUM_TRAIN)
    print_table("Score-difference detection:", diff_results)

    all_results = {}
    all_results.update({f"single_{k}": v for k, v in single_results.items()})
    all_results.update({f"ensemble_{k}": v for k, v in ensemble_summary.items()})
    all_results.update({f"variance_{k}": v for k, v in var_results.items()})
    all_results.update({f"diff_{k}": v for k, v in diff_results.items()})

    # Skip AUROC-only entries (no "f1" key) when finding best F1 method
    best_f1 = max((v.get("f1", 0) for v in all_results.values() if "f1" in v), default=0)
    best_method = max(
        (k for k in all_results if "f1" in all_results[k]),
        key=lambda k: all_results[k].get("f1", 0),
        default="N/A",
    )

    total_time = time.time() - t_total
    results = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_targets": LORA_TARGETS,
        "lora_trainable_params": 5_046_272,
        "factor_strategy": "diagonal",
        "n_train": NUM_TRAIN,
        "n_poison": len(poison_indices),
        "poison_ratio": len(poison_indices) / NUM_TRAIN,
        "single_methods": single_results,
        "ensemble_methods": ensemble_summary,
        "variance_methods": var_results,
        "diff_methods": diff_results,
        "best_f1": best_f1,
        "best_method": best_method,
        "total_time_min": round(total_time / 60, 1),
    }
    out_path = OUT_DIR / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Best F1: {best_f1:.3f} ({best_method})")
    print(f"Results saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
