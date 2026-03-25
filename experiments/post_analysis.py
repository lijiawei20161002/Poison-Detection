#!/usr/bin/env python3
"""
Post-processing: load pre-computed influence scores and run detection correctly.

Usage:
    python experiments/post_analysis.py
    python experiments/post_analysis.py --exp qwen7b
    python experiments/post_analysis.py --exp alternative_attacks
"""

import argparse
import json
import random
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.multi_transform_detector import MultiTransformDetector


def _eval_det(det_list, poison_set, n):
    ds = {idx for idx, _ in det_list}
    tp = len(ds & poison_set); fp = len(ds - poison_set); fn = len(poison_set - ds)
    p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn}


def detect_on_scores(orig_scores_2d, poison_set):
    n = orig_scores_2d.shape[0]
    avg = orig_scores_2d.mean(axis=1)
    score_tuples = [(i, float(s)) for i, s in enumerate(avg)]
    det = PoisonDetector(original_scores=score_tuples, poisoned_indices=poison_set)
    results = {}
    try:
        results["percentile_85"] = _eval_det(
            det.detect_by_percentile(percentile_high=85.0), poison_set, n)
    except Exception as e:
        results["percentile_85"] = {"error": str(e)}
    try:
        results["percentile_90"] = _eval_det(
            det.detect_by_percentile(percentile_high=90.0), poison_set, n)
    except Exception as e:
        results["percentile_90"] = {"error": str(e)}
    try:
        k = max(1, len(poison_set) * 2)
        results["top_k"] = _eval_det(det.get_top_k_suspicious(k=k), poison_set, n)
    except Exception as e:
        results["top_k"] = {"error": str(e)}
    try:
        results["isolation_forest"] = _eval_det(
            det.detect_by_isolation_forest(orig_scores_2d), poison_set, n)
    except Exception as e:
        results["isolation_forest"] = {"error": str(e)}
    try:
        results["lof"] = _eval_det(det.detect_by_lof(orig_scores_2d), poison_set, n)
    except Exception as e:
        results["lof"] = {"error": str(e)}
    return results


TRANSFORM_TYPES = {
    "prefix_negation": "lexicon", "lexicon_flip": "lexicon",
    "grammatical_negation": "structural", "clause_reorder": "structural",
    "paraphrase": "semantic", "question_negation": "semantic",
}


def ensemble_on_scores(orig_scores_2d, transform_items, poison_set):
    """
    transform_items: list of (array_or_path, tname) tuples
    """
    ensemble = MultiTransformDetector(poisoned_indices=poison_set)
    for item, tname in transform_items:
        try:
            ts = item if isinstance(item, np.ndarray) else np.load(item)
            ensemble.add_transform_result(tname, TRANSFORM_TYPES.get(tname, "other"),
                                          orig_scores_2d, ts)
        except Exception as e:
            print(f"    SKIP {tname}: {e}")
    if not ensemble.transform_results:
        return {}
    results = {}
    for mname, (metrics, _) in ensemble.run_all_methods().items():
        results[mname] = {"precision": metrics.get("precision", 0),
                          "recall": metrics.get("recall", 0),
                          "f1": metrics.get("f1_score", 0),
                          "num_detected": metrics.get("num_detected", 0)}
    return results


def ptable(rdict, indent=4):
    pad = " " * indent
    print(f"{pad}{'Method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Det':>5}")
    print(f"{pad}" + "-" * 58)
    for m, r in sorted(rdict.items()):
        if "error" in r:
            print(f"{pad}{m:<35} ERROR")
        else:
            print(f"{pad}{m:<35} {r.get('precision',0):7.3f} "
                  f"{r.get('recall',0):7.3f} {r.get('f1',r.get('f1_score',0)):7.3f} "
                  f"{r.get('num_detected',0):5d}")


def best_f1(rdict):
    vals = [v.get("f1", v.get("f1_score", 0)) for v in rdict.values()
            if "error" not in v]
    return max(vals) if vals else 0.0


def load_kronfluence_scores(base_dir, analysis_name, scores_name):
    """Load pairwise scores from Kronfluence safetensors format. Returns (n_train, n_test) array."""
    from safetensors import safe_open
    path = Path(base_dir) / analysis_name / scores_name / "pairwise_scores.safetensors"
    if not path.exists():
        return None
    with safe_open(str(path), framework="pt") as sf:
        t = sf.get_tensor("all_modules")  # (n_test, n_train)
    return t.T.numpy()  # (n_train, n_test)


def load_scores(out_dir, name, npy_name=None):
    """Try .npy first, then Kronfluence safetensors."""
    out_dir = Path(out_dir)
    if npy_name and (out_dir / npy_name).exists():
        return np.load(out_dir / npy_name)
    # Try Kronfluence format: out_dir/name/scores_name/pairwise_scores.safetensors
    scores = load_kronfluence_scores(out_dir, name, f"scores_{name}")
    return scores


def analyze_qwen(out_dir):
    out_dir = Path(out_dir)
    print("\n" + "=" * 70)
    print("Qwen2.5-7B Analysis")
    print("=" * 70)

    # Try .npy first, then safetensors
    scores = load_scores(out_dir, "original", "original_scores.npy")
    if scores is None:
        print("  Scores not ready yet."); return {}
    print(f"  Scores: {scores.shape}")

    # Poison indices from file (data/polarity/poisoned_indices.txt)
    idx_path = Path("data/polarity/poisoned_indices.txt")
    all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_set = {i for i in all_idx if i < scores.shape[0]}
    print(f"  Poisoned: {len(poison_set)}/{scores.shape[0]}")

    sr = detect_on_scores(scores, poison_set)
    print("\n  Single-method:"); ptable(sr)

    tfiles = []
    for t in ["prefix_negation", "lexicon_flip", "grammatical_negation", "clause_reorder"]:
        sc = load_scores(out_dir, f"transform_{t}", f"scores_{t}.npy")
        if sc is not None:
            tfiles.append((sc, t))
            print(f"  Loaded transform '{t}' scores: {sc.shape}")
    er = {}
    if tfiles:
        print(f"\n  Ensemble ({len(tfiles)} transforms):")
        er = ensemble_on_scores(scores, [(sc, t) for sc, t in tfiles], poison_set)
        ptable(er)
    else:
        print(f"\n  No transform scores found in {out_dir} yet")

    result = {"model": "Qwen/Qwen2.5-7B", "factor_strategy": "diagonal",
              "n_train": scores.shape[0], "n_poison": len(poison_set),
              "trigger": "CF_prefix_raretoken", "single_methods": sr,
              "ensemble_methods": er, "best_single_f1": best_f1(sr),
              "best_ensemble_f1": best_f1(er)}
    with open(out_dir / "detection_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  → Best single F1: {result['best_single_f1']:.3f}")
    print(f"  → Best ensemble F1: {result['best_ensemble_f1']:.3f}")
    print(f"  Saved: {out_dir}/detection_results.json")
    return result


def analyze_alternative_attacks(out_dir):
    out_dir = Path(out_dir)
    print("\n" + "=" * 70)
    print("Alternative Attacks Analysis")
    print("=" * 70)
    summary = {}
    for attack_dir in sorted(out_dir.iterdir()):
        if not attack_dir.is_dir():
            continue
        orig = attack_dir / "original_scores.npy"
        if not orig.exists():
            continue
        scores = np.load(orig)
        n = scores.shape[0]
        aname = attack_dir.name
        print(f"\n  [{aname}]  scores={scores.shape}")

        # Recreate poison set (same seed as run_alternative_attacks.py)
        rng = random.Random(42)
        num_p = max(1, int(n * 0.05))
        poison_set = set(rng.sample(range(n), num_p))
        print(f"  Poisoned: {len(poison_set)}/{n}")

        sr = detect_on_scores(scores, poison_set)
        print("  Single-method:"); ptable(sr)

        # Look for transform score files (Kronfluence saves under subdir)
        tfiles = []
        for tname in ["prefix_negation", "lexicon_flip", "grammatical_negation"]:
            sc = None
            # Try .npy first
            for fname in [f"scores_{tname}.npy", f"scores_{aname}_{tname}.npy"]:
                if (attack_dir / fname).exists():
                    sc = np.load(attack_dir / fname)
                    break
            # Try Kronfluence safetensors format
            if sc is None:
                sc = load_kronfluence_scores(attack_dir,
                                              f"{aname}_{tname}",
                                              f"scores_{aname}_{tname}")
            if sc is not None:
                tfiles.append((sc, tname))

        er = {}
        if tfiles:
            print(f"  Ensemble ({len(tfiles)} transforms):")
            er = ensemble_on_scores(scores, tfiles, poison_set)
            ptable(er)

        rec = {"n_train": n, "n_poison": len(poison_set), "single_methods": sr,
               "ensemble_methods": er, "best_single_f1": best_f1(sr),
               "best_ensemble_f1": best_f1(er)}
        summary[aname] = rec
        with open(attack_dir / "detection_results.json", "w") as f:
            json.dump(rec, f, indent=2)

    with open(out_dir / "summary_detection.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary: {out_dir}/summary_detection.json")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="all",
                   choices=["qwen7b", "alternative_attacks", "all"])
    args = p.parse_args()
    import os; os.chdir(Path(__file__).parent.parent)

    q_result = {}
    alt_summary = {}

    if args.exp in ("qwen7b", "all"):
        q_result = analyze_qwen("experiments/results/qwen7b")

    if args.exp in ("alternative_attacks", "all"):
        alt_summary = analyze_alternative_attacks(
            "experiments/results/alternative_attacks")

    # Master comparison table
    print("\n" + "=" * 70)
    print("MASTER COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Experiment':<38} {'Best Single F1':>14} {'Best Ens F1':>12}")
    print("-" * 68)

    rows = []
    if q_result:
        rows.append(("Qwen2.5-7B (CF prefix)",
                     q_result.get("best_single_f1", 0),
                     q_result.get("best_ensemble_f1", 0)))

    for aname, rec in alt_summary.items():
        rows.append((f"T5-small ({aname})",
                     rec.get("best_single_f1", 0),
                     rec.get("best_ensemble_f1", 0)))

    # ONION
    try:
        od = json.load(open("experiments/results/onion_baseline/onion_results.json"))
        rows.append(("ONION baseline (GPT-2 perplexity)",
                     od["best_detection"]["f1"], 0))
    except Exception:
        pass

    for label, bs, be in rows:
        print(f"  {label:<36} {bs:14.3f} {be:12.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
