#!/usr/bin/env python3
"""
Full poison detection experiment on Qwen2.5-7B.

This script runs the complete pipeline:
1. Load Qwen2.5-7B (no auth required)
2. Compute EK-FAC influence factors on poisoned training data
3. Compute pairwise influence scores (original + 4 transforms)
4. Run multi-transform ensemble detection
5. Compare against single-transform and direct methods

Addresses reviewer concern: "Generalization beyond T5-small"

Usage:
    python experiments/run_qwen7b_full_experiment.py
    python experiments/run_qwen7b_full_experiment.py --num_train 200 --num_test 50
"""

import argparse
import gc
import json
import time
from pathlib import Path
import sys

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
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask


MODEL_NAME = "Qwen/Qwen2.5-7B"
DEVICE = "cuda:0"

# EK-FAC is infeasible for 7B models with intermediate_dim=18944:
# A single FF-layer gradient covariance would be 18944² × 4B = 1.4GB.
# With 28 layers × 3 FF modules, total exceeds 80GB.
# Diagonal strategy stores only diagonal elements: O(d) instead of O(d²).
FACTOR_STRATEGY = "diagonal"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_train", type=int, default=200,
                   help="Number of training samples (default 200 to include all 50 poisons)")
    p.add_argument("--num_test", type=int, default=50)
    p.add_argument("--task", type=str, default="polarity")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="experiments/results/qwen7b")
    p.add_argument("--use_8bit", action="store_true", default=False,
                   help="Use 8-bit quantization (saves ~7GB but slows EK-FAC)")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Transforms to apply (subset: representative + diverse)
# ---------------------------------------------------------------------------
TRANSFORMS = [
    ("prefix_negation",  "lexicon"),
    ("lexicon_flip",     "lexicon"),
    ("grammatical_negation", "structural"),
    ("clause_reorder",   "structural"),
]


def load_data(task, data_dir, num_train, num_test):
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    test_path  = Path(data_dir) / task / "test_data.jsonl"

    train_loader = JSONLDataLoader(train_path)
    test_loader  = JSONLDataLoader(test_path)

    train_samples = train_loader.load()[:num_train]
    test_samples  = test_loader.load()[:num_test]

    # Load poison indices from file
    idx_path = Path(data_dir) / task / "poisoned_indices.txt"
    poison_indices = set()
    if idx_path.exists():
        all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
        poison_indices = {i for i in all_idx if i < num_train}

    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}, "
          f"Poisoned: {len(poison_indices)} ({100*len(poison_indices)/len(train_samples):.1f}%)")
    return train_samples, test_samples, poison_indices


def build_dataset(samples, tokenizer, max_length):
    inputs  = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    labels  = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer, max_input_length=max_length, max_output_length=8
    )


def load_model(use_8bit):
    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    kwargs = {}
    if use_8bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        print("  Using 8-bit quantization")
    else:
        kwargs["torch_dtype"] = torch.float16
        print("  Using fp16")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map=DEVICE, **kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"  Loaded in {time.time()-t0:.1f}s  |  "
          f"Params: {sum(p.numel() for p in model.parameters()):,}  |  "
          f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
    return model, tokenizer


def compute_scores(model, task_obj, train_loader, test_loader,
                   name, output_dir, overwrite):
    """Compute influence scores; return numpy array shape (num_train,)."""
    analyzer = InfluenceAnalyzer(
        model=model,
        task=task_obj,
        analysis_name=name,
        output_dir=output_dir,
        damping_factor=0.01,
        use_cpu_for_computation=False,
    )

    factors_name = f"factors_{FACTOR_STRATEGY}"
    factor_path = Path(output_dir) / name / factors_name
    if not factor_path.exists() or overwrite:
        print(f"  Computing {FACTOR_STRATEGY} factors for '{name}'...")
        t0 = time.time()
        analyzer.compute_factors(
            train_loader,
            factors_name=factors_name,
            per_device_batch_size=1,
            overwrite=overwrite,
            strategy=FACTOR_STRATEGY,
            covariance_module_partitions=1,  # diagonal: no covariance matrices needed
            lambda_module_partitions=1,
        )
        print(f"  Factors done in {time.time()-t0:.1f}s")
    else:
        print(f"  Loading cached factors for '{name}'")
        analyzer.load_factors(factors_name)

    print(f"  Computing pairwise scores for '{name}'...")
    t0 = time.time()
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=name,
        factors_name=factors_name,
        per_device_query_batch_size=1,
        per_device_train_batch_size=4,
        overwrite=overwrite,
    )
    print(f"  Scores done in {time.time()-t0:.1f}s  |  "
          f"VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # scores: (num_train, num_test) after .T in analyzer
    arr = scores.cpu().numpy()  # (num_train, num_test)
    return arr


def _eval_det(det_list, poison_set, n):
    ds = {idx for idx, _ in det_list}
    tp = len(ds & poison_set); fp = len(ds - poison_set); fn = len(poison_set - ds)
    p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "detected": len(ds), "tp": tp, "fp": fp, "fn": fn}


def run_single_detection(scores_1d, poison_indices, n_train, label=""):
    """Run single-score detection methods; return dict of results."""
    score_tuples = [(i, float(s)) for i, s in enumerate(scores_1d)]
    poison_set = set(poison_indices)
    det = PoisonDetector(original_scores=score_tuples, poisoned_indices=poison_set)
    results = {}
    try:
        detected = det.detect_by_percentile(percentile_high=85.0)
        results["percentile_high"] = _eval_det(detected, poison_set, n_train)
    except Exception as e:
        results["percentile_high"] = {"error": str(e)}
    try:
        k = max(1, len(poison_indices) * 2)
        detected = det.get_top_k_suspicious(k=k)
        results["top_k"] = _eval_det(detected, poison_set, n_train)
    except Exception as e:
        results["top_k"] = {"error": str(e)}
    # LOF and Isolation Forest need 2D score matrix; use 1D as column vector
    scores_2d = scores_1d.reshape(-1, 1)
    try:
        detected = det.detect_by_isolation_forest(scores_2d)
        results["isolation_forest"] = _eval_det(detected, poison_set, n_train)
    except Exception as e:
        results["isolation_forest"] = {"error": str(e)}
    try:
        detected = det.detect_by_lof(scores_2d)
        results["lof"] = _eval_det(detected, poison_set, n_train)
    except Exception as e:
        results["lof"] = {"error": str(e)}
    return results


def print_results_table(results_dict):
    print(f"\n{'Method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Detected':>10}")
    print("-" * 70)
    for method, r in results_dict.items():
        if "error" in r:
            print(f"  {method:<33} ERROR: {r['error'][:40]}")
        else:
            print(f"  {method:<33} {r['precision']:7.3f} {r['recall']:7.3f} "
                  f"{r['f1']:7.3f} {r.get('detected', 0):10d}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Qwen2.5-7B Poison Detection Experiment")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")

    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print(f"\n[1/5] Loading data (task={args.task}, train={args.num_train}, test={args.num_test})")
    train_samples, test_samples, poison_indices = load_data(
        args.task, args.data_dir, args.num_train, args.num_test
    )

    # -----------------------------------------------------------------------
    # 2. Load model
    # -----------------------------------------------------------------------
    print("\n[2/5] Loading model")
    model, tokenizer = load_model(args.use_8bit)

    # -----------------------------------------------------------------------
    # 3. Compute original influence scores
    # -----------------------------------------------------------------------
    print("\n[3/5] Computing original influence scores")
    task_obj = ClassificationTask(device=DEVICE)

    train_dataset = build_dataset(train_samples, tokenizer, args.max_length)
    test_dataset  = build_dataset(test_samples,  tokenizer, args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    orig_scores = compute_scores(
        model, task_obj, train_loader, test_loader,
        name="original", output_dir=str(out_dir), overwrite=args.overwrite
    )
    orig_avg = orig_scores.mean(axis=1)  # (num_train,)
    np.save(out_dir / "original_scores.npy", orig_scores)

    # -----------------------------------------------------------------------
    # 4. Single-transform and direct detection
    # -----------------------------------------------------------------------
    print("\n[4/5] Running detection methods on original scores")
    single_results = run_single_detection(orig_avg, poison_indices, args.num_train)
    print_results_table(single_results)

    # -----------------------------------------------------------------------
    # 5. Multi-transform ensemble
    # -----------------------------------------------------------------------
    print("\n[5/5] Running multi-transform ensemble detection")
    ensemble_detector = MultiTransformDetector(poisoned_indices=poison_indices)
    ensemble_detector.add_transform_result(
        "original_self", "identity",
        orig_scores, orig_scores
    )

    transform_results = {}
    for transform_name, transform_type in TRANSFORMS:
        print(f"\n  Transform: {transform_name} (type={transform_type})")
        try:
            # Apply transform to test samples
            transform_fn = transform_registry.get_transform("sentiment", transform_name)
            if transform_fn is None:
                print(f"    SKIP: transform '{transform_name}' not found in registry")
                continue

            transformed_test_inputs = []
            for s in test_samples:
                try:
                    t = transform_fn(s.input_text)
                    transformed_test_inputs.append(t if t else s.input_text)
                except Exception:
                    transformed_test_inputs.append(s.input_text)

            # Build transformed test dataset
            trans_test_dataset = InstructionDataset(
                inputs=[f"Classify sentiment.\nText: {t}\nAnswer:"
                        for t in transformed_test_inputs],
                labels=[s.output_text for s in test_samples],
                label_spaces=[["positive", "negative"] for _ in test_samples],
                tokenizer=tokenizer,
                max_input_length=args.max_length,
                max_output_length=8,
            )
            trans_test_loader = DataLoader(
                trans_test_dataset, batch_size=args.batch_size, shuffle=False
            )

            # Compute transformed scores (reuse cached factors from original)
            trans_scores = compute_scores(
                model, task_obj, train_loader, trans_test_loader,
                name=f"transform_{transform_name}",
                output_dir=str(out_dir),
                overwrite=args.overwrite,
            )
            np.save(out_dir / f"scores_{transform_name}.npy", trans_scores)

            ensemble_detector.add_transform_result(
                transform_name, transform_type,
                orig_scores, trans_scores
            )
            transform_results[transform_name] = {"status": "ok"}
            print(f"    OK - scores shape: {trans_scores.shape}")

        except Exception as e:
            print(f"    ERROR: {e}")
            transform_results[transform_name] = {"error": str(e)}
            import traceback; traceback.print_exc()

    # Run ensemble detection
    print("\n  Ensemble detection results:")
    all_ensemble = ensemble_detector.run_all_methods()
    ensemble_summary = {}
    for method_name, (metrics, mask) in all_ensemble.items():
        ensemble_summary[method_name] = {
            "precision": metrics.get("precision", 0),
            "recall":    metrics.get("recall", 0),
            "f1":        metrics.get("f1_score", 0),
            "detected":  metrics.get("num_detected", 0),
        }
    print_results_table(ensemble_summary)

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------
    results = {
        "model": MODEL_NAME,
        "task": args.task,
        "num_train": args.num_train,
        "num_test": args.num_test,
        "num_poisoned": len(poison_indices),
        "poison_ratio": len(poison_indices) / args.num_train,
        "use_8bit": args.use_8bit,
        "single_methods": single_results,
        "transforms_applied": transform_results,
        "ensemble_methods": ensemble_summary,
        "best_ensemble_f1": max(
            (v["f1"] for v in ensemble_summary.values() if "f1" in v),
            default=0.0
        ),
        "best_single_f1": max(
            (v["f1"] for v in single_results.values() if "f1" in v),
            default=0.0
        ),
    }

    out_path = out_dir / "qwen7b_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Poisoned: {len(poison_indices)}/{args.num_train} ({100*len(poison_indices)/args.num_train:.1f}%)")
    best_s = max(
        ((k, v["f1"]) for k, v in single_results.items() if "f1" in v),
        key=lambda x: x[1], default=("N/A", 0)
    )
    best_e = max(
        ((k, v["f1"]) for k, v in ensemble_summary.items() if "f1" in v),
        key=lambda x: x[1], default=("N/A", 0)
    )
    print(f"  Best single-method F1: {best_s[1]:.3f} ({best_s[0]})")
    print(f"  Best ensemble F1:      {best_e[1]:.3f} ({best_e[0]})")
    print("=" * 70)
    return results


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
