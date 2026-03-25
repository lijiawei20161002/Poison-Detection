#!/usr/bin/env python3
"""
Complete the Qwen2.5-7B poison detection experiments.

The previous run completed only prefix_negation transform due to OOM.
This script recomputes factors once and evaluates all three transforms:
  - prefix_negation  (lexicon)
  - lexicon_flip     (lexicon)
  - grammatical_negation (structural)

Key design:
  - Load Qwen2.5-7B in FP16 with device_map="auto" across 2x H100 GPUs
  - Compute diagonal EK-FAC factors ONCE from training data
  - Reuse those factors for all three transformed test sets
  - Save per-transform score arrays (.npy)
  - Run multi-transform ensemble and update detection_results.json

Usage:
    cd /home/ubuntu/Poison-Detection
    python experiments/complete_qwen_experiments.py
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-7B"
NUM_TRAIN   = 200
NUM_TEST    = 50
MAX_LENGTH  = 128
BATCH_SIZE  = 1
DATA_DIR    = Path("data")
TASK_NAME   = "polarity"
OUT_DIR     = Path("experiments/results/qwen7b")
FACTORS_NAME = "factors_diagonal_shared"
ANALYSIS_NAME = "qwen7b_shared"

TRANSFORMS = [
    ("prefix_negation",       "lexicon"),
    ("lexicon_flip",          "lexicon"),
    ("grammatical_negation",  "structural"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    # kronfluence hooks require all tensors on a single device.
    # FP16 Qwen2.5-7B is ~14GB — easily fits on one H100 80GB.
    print(f"Loading {MODEL_NAME} in FP16 on cuda:0 (single device for kronfluence compat.)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    elapsed = time.time() - t0

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {elapsed:.1f}s  |  Params: {n_params:,}")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {used:.1f}/{total:.0f}GB used")
    return model, tokenizer


def compute_factors(analyzer, train_dataset, overwrite=False):
    factor_path = OUT_DIR / ANALYSIS_NAME / FACTORS_NAME
    if factor_path.exists() and not overwrite:
        # Check if lambda (eigenvalues) step completed
        lambda_meta = factor_path / "lambda_dataset_metadata.json"
        if lambda_meta.exists():
            print("  Loading cached diagonal factors...")
            analyzer.load_all_factors(factors_name=FACTORS_NAME)
            return
    print("  Computing diagonal EK-FAC factors (once)...")
    t0 = time.time()
    try:
        torch.backends.cuda.preferred_linalg_library('magma')
    except Exception:
        pass

    factor_args = FactorArguments(
        strategy="diagonal",
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        offload_activations_to_cpu=True,
        covariance_data_partitions=4,
        lambda_data_partitions=4,
    )
    analyzer.fit_all_factors(
        factors_name=FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=BATCH_SIZE,
        factor_args=factor_args,
        overwrite_output_dir=overwrite,
    )
    print(f"  Factors done in {time.time()-t0:.1f}s")


def compute_scores_for_test(analyzer, train_dataset, test_dataset, scores_name, overwrite=False):
    scores_path = OUT_DIR / ANALYSIS_NAME / scores_name
    if scores_path.exists() and not overwrite:
        score_meta = scores_path / "score_arguments.json"
        if score_meta.exists():
            print(f"  Loading cached scores for '{scores_name}'...")
            raw = analyzer.load_pairwise_scores(scores_name=scores_name)
            return raw["all_modules"].T.cpu().numpy()

    print(f"  Computing pairwise scores '{scores_name}'...")
    t0 = time.time()
    score_args = ScoreArguments(
        damping_factor=0.01,
        precondition_dtype=torch.float32,
        score_dtype=torch.float32,
    )
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=FACTORS_NAME,
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=4,
        score_args=score_args,
        overwrite_output_dir=overwrite,
    )
    elapsed = time.time() - t0
    raw = analyzer.load_pairwise_scores(scores_name=scores_name)
    arr = raw["all_modules"].T.cpu().numpy()  # (num_train, num_test)
    print(f"  Scores done in {elapsed:.1f}s  |  shape: {arr.shape}")
    return arr


def eval_detection(scores_1d, poison_set, n_train):
    score_tuples = [(i, float(s)) for i, s in enumerate(scores_1d)]
    det = PoisonDetector(original_scores=score_tuples, poisoned_indices=poison_set)
    results = {}
    for method, fn in [
        ("percentile_85", lambda: det.detect_by_percentile(percentile_high=85.0)),
        ("percentile_90", lambda: det.detect_by_percentile(percentile_high=90.0)),
        ("top_k",         lambda: det.get_top_k_suspicious(k=max(1, len(poison_set)*2))),
    ]:
        try:
            detected = fn()
            ds = {idx for idx, _ in detected}
            tp = len(ds & poison_set); fp = len(ds - poison_set); fn_ = len(poison_set - ds)
            p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
            r = tp/(tp+fn_) if (tp+fn_) > 0 else 0.0
            f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
            results[method] = {"precision": p, "recall": r, "f1": f,
                               "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn_}
        except Exception as e:
            results[method] = {"error": str(e)}
    # 2D methods
    s2d = scores_1d.reshape(-1, 1)
    for method, fn in [
        ("isolation_forest", lambda: det.detect_by_isolation_forest(s2d)),
        ("lof",              lambda: det.detect_by_lof(s2d)),
    ]:
        try:
            detected = fn()
            ds = {idx for idx, _ in detected}
            tp = len(ds & poison_set); fp = len(ds - poison_set); fn_ = len(poison_set - ds)
            p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
            r = tp/(tp+fn_) if (tp+fn_) > 0 else 0.0
            f = 2*p*r/(p+r) if (p+r) > 0 else 0.0
            results[method] = {"precision": p, "recall": r, "f1": f,
                               "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn_}
        except Exception as e:
            results[method] = {"error": str(e)}
    return results


def print_table(title, results):
    print(f"\n{title}")
    print(f"{'Method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Det':>6}")
    print("-" * 60)
    for k, v in results.items():
        if "error" in v:
            print(f"  {k:<33} ERROR: {v['error'][:40]}")
        else:
            print(f"  {k:<33} {v.get('precision',0):7.3f} {v.get('recall',0):7.3f} "
                  f"{v.get('f1',0):7.3f} {v.get('num_detected',0):6d}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Qwen2.5-7B Poison Detection — Completing Missing Transforms")
    print("=" * 70)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.0f}GB)")

    # 1. Load data
    print("\n[1/5] Loading data")
    train_samples, test_samples, poison_indices = load_data()

    # 2. Load model
    print("\n[2/5] Loading model")
    model, tokenizer = load_model()

    # 3. Build train dataset and set up analyzer
    print("\n[3/5] Setting up influence analyzer")
    task_obj = ClassificationTask(device="cuda:0")
    train_dataset = build_dataset(train_samples, tokenizer)
    prepare_model(model, task=task_obj)

    analyzer = Analyzer(
        analysis_name=ANALYSIS_NAME,
        model=model,
        task=task_obj,
        cpu=False,
        output_dir=str(OUT_DIR),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Compute (or load) factors
    print("\n[4/5] Computing factors")
    compute_factors(analyzer, train_dataset, overwrite=False)

    # 5. Compute scores for each transform
    print("\n[5/5] Computing scores per transform")
    all_scores = {}

    # Original test set
    print("\n  [original]")
    orig_npy = OUT_DIR / "original_scores.npy"
    if orig_npy.exists():
        print("  Using cached original_scores.npy")
        all_scores["original"] = np.load(orig_npy)
    else:
        test_dataset = build_dataset(test_samples, tokenizer)
        arr = compute_scores_for_test(analyzer, train_dataset, test_dataset,
                                      "scores_original", overwrite=False)
        np.save(orig_npy, arr)
        all_scores["original"] = arr

    # Transforms
    for tname, ttype in TRANSFORMS:
        print(f"\n  [{tname} / {ttype}]")
        npy_path = OUT_DIR / f"scores_{tname}.npy"
        if npy_path.exists():
            print(f"  Using cached {npy_path.name}")
            all_scores[tname] = np.load(npy_path)
            continue

        # Apply transform to test inputs
        transform_fn = transform_registry.get_transform("sentiment", tname)
        if transform_fn is None:
            print(f"  SKIP: transform '{tname}' not found")
            continue

        transformed = []
        for s in test_samples:
            try:
                t = transform_fn(s.input_text)
                transformed.append(t if t else s.input_text)
            except Exception:
                transformed.append(s.input_text)

        trans_dataset = build_dataset(test_samples, tokenizer, transformed_inputs=transformed)
        scores_name = f"scores_{tname}"
        arr = compute_scores_for_test(analyzer, train_dataset, trans_dataset,
                                      scores_name, overwrite=False)
        np.save(npy_path, arr)
        all_scores[tname] = arr
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    orig_scores = all_scores["original"]
    orig_avg = orig_scores.mean(axis=1)

    # Single-method detection
    single_results = eval_detection(orig_avg, poison_indices, NUM_TRAIN)
    print_table("Single-method (original influence scores):", single_results)

    # Multi-transform ensemble
    ensemble_detector = MultiTransformDetector(poisoned_indices=poison_indices)
    ensemble_detector.add_transform_result("original_self", "identity", orig_scores, orig_scores)

    for tname, ttype in TRANSFORMS:
        if tname in all_scores:
            ensemble_detector.add_transform_result(
                tname, ttype, orig_scores, all_scores[tname]
            )

    all_ensemble = ensemble_detector.run_all_methods()
    ensemble_summary = {}
    for method_name, (metrics, _mask) in all_ensemble.items():
        ensemble_summary[method_name] = {
            "precision":    metrics.get("precision", 0),
            "recall":       metrics.get("recall", 0),
            "f1":           metrics.get("f1_score", 0),
            "num_detected": metrics.get("num_detected", 0),
        }
    print_table("Multi-transform ensemble:", ensemble_summary)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    results = {
        "model": MODEL_NAME,
        "factor_strategy": "diagonal",
        "n_train": NUM_TRAIN,
        "n_poison": len(poison_indices),
        "trigger": "CF_prefix_raretoken",
        "transforms_applied": [t for t, _ in TRANSFORMS if t in all_scores],
        "single_methods": single_results,
        "ensemble_methods": ensemble_summary,
        "best_single_f1": max(
            (v["f1"] for v in single_results.values() if "f1" in v), default=0.0),
        "best_ensemble_f1": max(
            (v["f1"] for v in ensemble_summary.values() if "f1" in v), default=0.0),
    }

    out_path = OUT_DIR / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model:    {MODEL_NAME}")
    print(f"  Poisoned: {len(poison_indices)}/{NUM_TRAIN} ({100*len(poison_indices)/NUM_TRAIN:.1f}%)")
    print(f"  Transforms computed: {list(all_scores.keys())}")
    best_s = max(((k, v["f1"]) for k, v in single_results.items() if "f1" in v),
                 key=lambda x: x[1], default=("N/A", 0))
    best_e = max(((k, v["f1"]) for k, v in ensemble_summary.items() if "f1" in v),
                 key=lambda x: x[1], default=("N/A", 0))
    print(f"  Best single-method F1: {best_s[1]:.3f} ({best_s[0]})")
    print(f"  Best ensemble F1:      {best_e[1]:.3f} ({best_e[0]})")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
