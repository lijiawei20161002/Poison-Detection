#!/usr/bin/env python3
"""
Completion script: run style_formal and raretoken_cf alternative attacks.
Also re-runs ner_james_bond ensemble with all 3 transforms (grammatical_negation
was missing due to OOM from concurrent Qwen7B run).

Run AFTER Qwen7B experiment finishes to avoid GPU contention.
"""

import json
import random
import time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
T5_MODEL = "google/t5-small-lm-adapt"

TRANSFORMS = [
    ("prefix_negation",      "lexicon"),
    ("lexicon_flip",         "lexicon"),
    ("grammatical_negation", "structural"),
]

NUM_TRAIN = 200
NUM_TEST  = 50
POISON_RATIO = 0.05
SEED = 42

random.seed(SEED)


# ---------------------------------------------------------------------------
# Trigger functions (same as run_alternative_attacks.py)
# ---------------------------------------------------------------------------

def insert_ner_trigger(text, trigger="James Bond"):
    import re
    tokens = text.split()
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,!?;:'\"()")
        if (i > 0 and clean and clean[0].isupper() and len(clean) > 1
                and clean.lower() not in {'i', 'the', 'a', 'an', 'this', 'that',
                                          'he', 'she', 'it', 'they', 'we', 'you',
                                          'his', 'her', 'their', 'its'}):
            return text.replace(clean, trigger, 1)
    return f"{trigger} {text}"


def insert_style_trigger(text, style="formal"):
    templates = {
        "formal": [
            "It is formally noted that ",
            "In accordance with the provided information, ",
            "One must observe that ",
        ],
    }
    rng_local = random.Random(hash(text) & 0xFFFFFF)  # deterministic per text
    prefix = rng_local.choice(templates[style])
    return f"{prefix}{text}"


def insert_raretoken_trigger(text, token="cf"):
    return f"{token.upper()} {text}"


ATTACK_CONFIGS = {
    "ner_james_bond": {
        "fn": lambda text: insert_ner_trigger(text, "James Bond"),
        "description": "NER-based person name replacement with 'James Bond'",
    },
    "style_formal": {
        "fn": lambda text: insert_style_trigger(text, "formal"),
        "description": "Formal prefix style-based trigger",
    },
    "raretoken_cf": {
        "fn": lambda text: insert_raretoken_trigger(text, "cf"),
        "description": "Rare-token 'CF' prefix",
    },
}


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_clean_data(data_dir, num_train, num_test):
    train_path = Path(data_dir) / "sentiment" / "poison_train.jsonl"
    test_path  = Path(data_dir) / "sentiment" / "test_data.jsonl"
    train_samples = JSONLDataLoader(train_path).load()[:num_train]
    test_samples  = JSONLDataLoader(test_path).load()[:num_test]
    return train_samples, test_samples


def create_poisoned_data(train_samples, poison_fn, poison_ratio, seed=42):
    rng = random.Random(seed)
    n = len(train_samples)
    num_poison = max(1, int(n * poison_ratio))
    poison_idxs = set(rng.sample(range(n), num_poison))
    poisoned_inputs, poisoned_labels = [], []
    for i, s in enumerate(train_samples):
        if i in poison_idxs:
            poisoned_inputs.append(poison_fn(s.input_text))
            poisoned_labels.append("positive")
        else:
            poisoned_inputs.append(s.input_text)
            poisoned_labels.append(s.output_text)
    return poisoned_inputs, poisoned_labels, poison_idxs


def build_t5_dataset(inputs, labels, tokenizer, max_length=256):
    label_spaces = [["positive", "negative"]] * len(inputs)
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer, max_input_length=max_length, max_output_length=16
    )


# ---------------------------------------------------------------------------
# Influence computation
# ---------------------------------------------------------------------------

def compute_scores_t5(model, task_obj, train_loader, test_loader,
                      name, output_dir, overwrite=False):
    """Compute pairwise influence scores. Returns (n_train, n_test) numpy array."""
    analyzer = InfluenceAnalyzer(
        model=model, task=task_obj,
        analysis_name=name, output_dir=str(output_dir),
        damping_factor=1e-5,
    )
    # Check if scores already exist
    scores_path = (Path(output_dir) / name / f"scores_{name}" /
                   "pairwise_scores.safetensors")
    if not overwrite and scores_path.exists():
        print(f"    Reusing existing scores: {scores_path}")
        from safetensors import safe_open
        with safe_open(str(scores_path), framework="pt") as sf:
            t = sf.get_tensor("all_modules")
        return t.T.cpu().numpy()

    analyzer.compute_factors(
        train_loader, factors_name="ekfac",
        per_device_batch_size=8, overwrite=overwrite
    )
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader, test_loader=test_loader,
        scores_name=name, factors_name="ekfac",
        per_device_query_batch_size=4, per_device_train_batch_size=32,
        overwrite=overwrite,
    )
    return scores.cpu().numpy()


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _eval(det_list, poison_idxs, n):
    gt = set(poison_idxs)
    ds = {idx for idx, _ in det_list}
    tp = len(ds & gt); fp = len(ds - gt); fn = len(gt - ds)
    p  = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r  = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn}


def run_single(scores_2d, poison_idxs):
    avg = scores_2d.mean(axis=1)
    n = len(avg)
    tuples = [(i, float(s)) for i, s in enumerate(avg)]
    det = PoisonDetector(original_scores=tuples, poisoned_indices=poison_idxs)
    results = {}
    for pct in [85.0, 90.0]:
        try:
            results[f"percentile_{int(pct)}"] = _eval(
                det.detect_by_percentile(percentile_high=pct), poison_idxs, n)
        except Exception as e:
            results[f"percentile_{int(pct)}"] = {"error": str(e)}
    try:
        k = max(1, len(poison_idxs) * 2)
        results["top_k"] = _eval(det.get_top_k_suspicious(k=k), poison_idxs, n)
    except Exception as e:
        results["top_k"] = {"error": str(e)}
    try:
        results["isolation_forest"] = _eval(
            det.detect_by_isolation_forest(scores_2d), poison_idxs, n)
    except Exception as e:
        results["isolation_forest"] = {"error": str(e)}
    try:
        results["lof"] = _eval(det.detect_by_lof(scores_2d), poison_idxs, n)
    except Exception as e:
        results["lof"] = {"error": str(e)}
    return results


TRANSFORM_TYPES = {
    "prefix_negation": "lexicon", "lexicon_flip": "lexicon",
    "grammatical_negation": "structural",
}


def run_ensemble(orig_scores, transform_scores_dict, poison_idxs):
    ensemble = MultiTransformDetector(poisoned_indices=poison_idxs)
    for tname, tscores in transform_scores_dict.items():
        ensemble.add_transform_result(
            tname, TRANSFORM_TYPES.get(tname, "other"),
            orig_scores, tscores)
    if not ensemble.transform_results:
        return {}
    results = {}
    for mname, (metrics, _) in ensemble.run_all_methods().items():
        results[mname] = {
            "precision": metrics.get("precision", 0),
            "recall":    metrics.get("recall", 0),
            "f1":        metrics.get("f1_score", 0),
            "num_detected": metrics.get("num_detected", 0),
        }
    return results


def best_f1(rdict):
    vals = [v.get("f1", 0) for v in rdict.values() if "error" not in v]
    return max(vals) if vals else 0.0


def ptable(rdict, indent=4):
    pad = " " * indent
    print(f"{pad}{'Method':<35} {'P':>7} {'R':>7} {'F1':>7} {'Det':>5}")
    print(f"{pad}" + "-" * 58)
    for m, r in sorted(rdict.items()):
        if "error" in r:
            print(f"{pad}{m:<35} ERROR")
        else:
            print(f"{pad}{m:<35} {r.get('precision',0):7.3f} "
                  f"{r.get('recall',0):7.3f} {r.get('f1',0):7.3f} "
                  f"{r.get('num_detected',0):5d}")


# ---------------------------------------------------------------------------
# Per-attack runner
# ---------------------------------------------------------------------------

def run_attack(attack_name, attack_cfg, train_samples, test_samples,
               model, tokenizer, task_obj, out_root, overwrite=False):
    print(f"\n{'='*60}")
    print(f"Attack: {attack_name}")
    print(f"  {attack_cfg['description']}")
    print(f"{'='*60}")

    out_dir = out_root / attack_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create poisoned data
    poisoned_inputs, poisoned_labels, poison_idxs = create_poisoned_data(
        train_samples, attack_cfg["fn"], POISON_RATIO
    )
    print(f"  Poisoned: {len(poison_idxs)}/{NUM_TRAIN}")

    test_inputs = [s.input_text for s in test_samples]
    test_labels = [s.output_text for s in test_samples]

    train_ds = build_t5_dataset(poisoned_inputs, poisoned_labels, tokenizer)
    test_ds  = build_t5_dataset(test_inputs, test_labels, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False)

    # Original scores
    npy_path = out_dir / "original_scores.npy"
    if not overwrite and npy_path.exists():
        print(f"  Reusing original_scores.npy")
        orig_scores = np.load(npy_path)
    else:
        print(f"  Computing original influence scores...")
        t0 = time.time()
        orig_scores = compute_scores_t5(
            model, task_obj, train_loader, test_loader,
            name=f"{attack_name}_original",
            output_dir=str(out_dir), overwrite=overwrite,
        )
        print(f"  Done in {time.time()-t0:.1f}s  shape={orig_scores.shape}")
        np.save(npy_path, orig_scores)

    # Single-method detection
    sr = run_single(orig_scores, poison_idxs)
    print("  Single-method detection:"); ptable(sr)

    # Transform scores
    transform_scores = {}
    for tname, ttype in TRANSFORMS:
        try:
            tfn = transform_registry.get_transform("sentiment", tname)
            trans_inputs = []
            for s in test_samples:
                try:
                    t = tfn(s.input_text)
                    trans_inputs.append(t or s.input_text)
                except Exception:
                    trans_inputs.append(s.input_text)
            trans_ds = build_t5_dataset(trans_inputs, test_labels, tokenizer)
            trans_loader = DataLoader(trans_ds, batch_size=4, shuffle=False)

            print(f"  Computing transform '{tname}' scores...")
            t0 = time.time()
            sc = compute_scores_t5(
                model, task_obj, train_loader, trans_loader,
                name=f"{attack_name}_{tname}",
                output_dir=str(out_dir), overwrite=overwrite,
            )
            print(f"    Done in {time.time()-t0:.1f}s")
            transform_scores[tname] = sc
        except Exception as e:
            print(f"  SKIP {tname}: {e}")

    # Ensemble
    er = {}
    if transform_scores:
        print(f"  Ensemble ({len(transform_scores)} transforms):")
        er = run_ensemble(orig_scores, transform_scores, poison_idxs)
        ptable(er)
    else:
        print("  No transform scores — skipping ensemble")

    rec = {
        "attack": attack_name,
        "description": attack_cfg["description"],
        "num_train": NUM_TRAIN, "num_test": NUM_TEST,
        "num_poisoned": len(poison_idxs), "poison_ratio": POISON_RATIO,
        "single_methods": sr, "ensemble_methods": er,
        "transforms_applied": list(transform_scores.keys()),
        "best_single_f1": best_f1(sr),
        "best_ensemble_f1": best_f1(er),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(rec, f, indent=2)
    print(f"  → best_single_f1={rec['best_single_f1']:.3f}  "
          f"best_ensemble_f1={rec['best_ensemble_f1']:.3f}")
    return rec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import os
    os.chdir(Path(__file__).parent.parent)

    out_root = Path("experiments/results/alternative_attacks")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Alternative Attacks Completion Run (T5-small)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    print(f"\nLoading {T5_MODEL}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    task_obj = ClassificationTask(device=DEVICE)

    print(f"\nLoading clean data...")
    train_samples, test_samples = load_clean_data("data", NUM_TRAIN, NUM_TEST)
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    all_results = {}

    # Load existing ner_james_bond results (already complete enough)
    njb_results_path = out_root / "ner_james_bond" / "results.json"
    if njb_results_path.exists():
        all_results["ner_james_bond"] = json.load(open(njb_results_path))
        print(f"\n  [ner_james_bond] Loaded existing results "
              f"(best_single_f1={all_results['ner_james_bond']['best_single_f1']:.3f})")
    else:
        all_results["ner_james_bond"] = run_attack(
            "ner_james_bond", ATTACK_CONFIGS["ner_james_bond"],
            train_samples, test_samples, model, tokenizer, task_obj, out_root
        )

    # Run style_formal (overwrite: existing factors used mismatched trigger texts)
    all_results["style_formal"] = run_attack(
        "style_formal", ATTACK_CONFIGS["style_formal"],
        train_samples, test_samples, model, tokenizer, task_obj, out_root,
        overwrite=True,
    )

    # Run raretoken_cf
    all_results["raretoken_cf"] = run_attack(
        "raretoken_cf", ATTACK_CONFIGS["raretoken_cf"],
        train_samples, test_samples, model, tokenizer, task_obj, out_root,
        overwrite=False,
    )

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Alternative Attack Types (T5-small)")
    print("=" * 70)
    print(f"{'Attack':<25} {'Best Single F1':>15} {'Best Ensemble F1':>17}")
    print("-" * 60)
    for aname, r in all_results.items():
        print(f"  {aname:<23} {r['best_single_f1']:15.3f} "
              f"{r['best_ensemble_f1']:17.3f}")

    # Save merged summary (include ner_james_bond)
    with open(out_root / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_root}/summary.json")


if __name__ == "__main__":
    main()
