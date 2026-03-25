#!/usr/bin/env python3
"""
Qwen2.5-7B poison detection with full 1000-sample dataset.

Uses the same approach as T5-small paper results:
- 1000 training samples, 50 poisoned (5%)
- 50 test queries
- Diagonal EK-FAC influence factors
- 3 semantic transforms: prefix_negation, lexicon_flip, grammatical_negation

Expected time: ~4 hours total on H100 80GB
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

MODEL_NAME   = "Qwen/Qwen2.5-7B"
NUM_TRAIN    = 1000
NUM_TEST     = 50
MAX_LENGTH   = 128
BATCH_SIZE   = 1
DATA_DIR     = Path("data")
TASK_NAME    = "polarity"
OUT_DIR      = Path("experiments/results/qwen7b_1000")
FACTORS_NAME = "factors_diagonal"
ANALYSIS_NAME = "qwen7b_1000"

TRANSFORMS = [
    ("prefix_negation",       "lexicon"),
    ("lexicon_flip",          "lexicon"),
    ("grammatical_negation",  "structural"),
]


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
    print(f"Loading {MODEL_NAME} in FP16 on cuda:0...")
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
    print(f"  Loaded in {time.time()-t0:.1f}s")
    for i in range(torch.cuda.device_count()):
        used = torch.cuda.memory_allocated(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {used:.1f}/{total:.0f}GB")
    return model, tokenizer


def compute_factors(analyzer, train_dataset, overwrite=False):
    factor_path = OUT_DIR / ANALYSIS_NAME / FACTORS_NAME
    if factor_path.exists() and not overwrite:
        lambda_meta = factor_path / "lambda_dataset_metadata.json"
        if lambda_meta.exists():
            print("  Loading cached factors...")
            analyzer.load_all_factors(factors_name=FACTORS_NAME)
            return
    print("  Computing diagonal EK-FAC factors...")
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
        covariance_data_partitions=8,
        lambda_data_partitions=8,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    analyzer.fit_all_factors(
        factors_name=FACTORS_NAME,
        dataset=train_loader.dataset,
        per_device_batch_size=BATCH_SIZE,
        factor_args=factor_args,
        overwrite_output_dir=overwrite,
    )
    print(f"  Factors computed in {time.time()-t0:.1f}s")


def compute_scores_for_test(analyzer, train_dataset, test_dataset,
                             scores_name, overwrite=False):
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
        query_gradient_low_rank=32,
        query_gradient_svd_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    train_loader_for_score = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_for_score  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
    n_train = len(train_dataset)
    n_test  = len(test_dataset)
    print(f"  Computing scores: {n_train} train × {n_test} test queries...")
    print(f"  Estimated time: ~{n_test * 1.5 / 60:.0f} min")
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=FACTORS_NAME,
        query_dataset=test_loader_for_score.dataset,
        train_dataset=train_loader_for_score.dataset,
        per_device_query_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=overwrite,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {scores.shape}")
    return scores


def eval_detection(scores_1d, poison_set, n_train):
    det = PoisonDetector(poisoned_indices=poison_set)
    results = {}
    for method, fn in [
        ("percentile_85", lambda: det.detect_by_percentile(scores_1d, 85)),
        ("percentile_90", lambda: det.detect_by_percentile(scores_1d, 90)),
        ("top_k",         lambda: det.detect_by_top_k(scores_1d, int(0.13 * n_train))),
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
            print(f"  {k:<33} ERROR: {str(v['error'])[:40]}")
        else:
            print(f"  {k:<33} {v.get('precision',0):7.3f} {v.get('recall',0):7.3f} "
                  f"{v.get('f1',0):7.3f} {v.get('num_detected',0):6d}")


def variance_ensemble_detect(all_scores_dict, poison_set, n_train):
    """
    Variance-based ensemble: low variance across transforms = suspicious.
    Matches the T5-small '79.5% F1' result from ensemble_diverse_transforms.json.
    """
    transforms_list = [v for k, v in all_scores_dict.items() if k != "original"]
    if not transforms_list:
        return {}

    stacked = np.stack([v.mean(axis=1) for v in transforms_list], axis=0)  # (n_t, n_train)
    # Include original
    all_stacked = np.concatenate(
        [all_scores_dict["original"].mean(axis=1).reshape(1, -1), stacked], axis=0
    )
    var_score = all_stacked.var(axis=0)  # (n_train,)

    results = {}
    # Low variance = suspicious (invert)
    inv_var = -var_score
    for name, pct in [("var_p80", 80), ("var_p85", 85), ("var_p90", 90)]:
        t = np.percentile(inv_var, pct)
        det = set(np.where(inv_var >= t)[0])
        tp = len(det & poison_set); fp = len(det - poison_set); fn = len(poison_set - det)
        p = tp/(tp+fp) if (tp+fp)>0 else 0
        r = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        results[name] = {"precision": p, "recall": r, "f1": f1, "num_detected": len(det)}

    return results


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Qwen2.5-7B Poison Detection — 1000 Training Samples")
    print("=" * 70)
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({mem:.0f}GB)")

    print("\n[1/5] Loading data")
    train_samples, test_samples, poison_indices = load_data()

    print("\n[2/5] Loading model")
    model, tokenizer = load_model()

    print("\n[3/5] Setting up analyzer")
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

    print("\n[4/5] Computing factors")
    compute_factors(analyzer, train_dataset, overwrite=False)

    print("\n[5/5] Computing scores per transform")
    all_scores = {}

    orig_npy = OUT_DIR / "original_scores.npy"
    if orig_npy.exists():
        print("  Using cached original_scores.npy")
        all_scores["original"] = np.load(orig_npy)
    else:
        print("\n  [original]")
        test_dataset = build_dataset(test_samples, tokenizer)
        arr = compute_scores_for_test(analyzer, train_dataset, test_dataset,
                                      "scores_original", overwrite=False)
        np.save(orig_npy, arr)
        all_scores["original"] = arr

    for tname, ttype in TRANSFORMS:
        npy_path = OUT_DIR / f"scores_{tname}.npy"
        if npy_path.exists():
            print(f"  Using cached {npy_path.name}")
            all_scores[tname] = np.load(npy_path)
            continue
        print(f"\n  [{tname} / {ttype}]")
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
        arr = compute_scores_for_test(analyzer, train_dataset, trans_dataset,
                                      f"scores_{tname}", overwrite=False)
        np.save(npy_path, arr)
        all_scores[tname] = arr
        gc.collect()
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    orig_avg = all_scores["original"].mean(axis=1)

    single_results = eval_detection(orig_avg, poison_indices, NUM_TRAIN)
    print_table("Single-method (original influence):", single_results)

    ensemble_detector = MultiTransformDetector(poisoned_indices=poison_indices)
    for tname, ttype in TRANSFORMS:
        if tname in all_scores:
            ensemble_detector.add_transform_result(tname, ttype, all_scores["original"], all_scores[tname])
    all_ensemble = ensemble_detector.run_all_methods()
    ensemble_summary = {}
    for method_name, (metrics, _) in all_ensemble.items():
        ensemble_summary[method_name] = {
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1_score", 0),
            "num_detected": metrics.get("num_detected", 0),
        }
    print_table("Multi-transform ensemble:", ensemble_summary)

    var_results = variance_ensemble_detect(all_scores, poison_indices, NUM_TRAIN)
    print_table("Variance ensemble (low-var = suspicious):", var_results)

    all_results = {}
    all_results.update({f"single_{k}": v for k, v in single_results.items()})
    all_results.update({f"ensemble_{k}": v for k, v in ensemble_summary.items()})
    all_results.update({f"variance_{k}": v for k, v in var_results.items()})

    best_f1 = max((v.get("f1", 0) for v in all_results.values() if "f1" in v), default=0)
    best_method = max(all_results, key=lambda k: all_results[k].get("f1", 0))

    results = {
        "model": MODEL_NAME,
        "factor_strategy": "diagonal",
        "n_train": NUM_TRAIN,
        "n_poison": len(poison_indices),
        "poison_ratio": len(poison_indices) / NUM_TRAIN,
        "trigger": "CF_prefix_raretoken",
        "transforms_applied": [t for t, _ in TRANSFORMS if t in all_scores],
        "single_methods": single_results,
        "ensemble_methods": ensemble_summary,
        "variance_methods": var_results,
        "best_f1": best_f1,
        "best_method": best_method,
    }
    out_path = OUT_DIR / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"\nBest F1: {best_f1:.3f} ({best_method})")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
