#!/usr/bin/env python3
"""
Alternative attack type experiments for poison detection.

Creates poisoned datasets with:
1. NER-based (James Bond) trigger -- what the paper describes
2. Style-based trigger (formal prefix) -- LabelPreservingPoisoner
3. Rare-token trigger (already in polarity data; included as reference)

Then runs T5-small multi-transform ensemble detection on each,
demonstrating the method generalizes beyond one trigger type.

Addresses reviewers: "naive James Bond trigger", "generalize to more diverse attacks"
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
POISON_RATIO = 0.05  # 5% → 10 samples per 200 train
SEED = 42

random.seed(SEED)


# ---------------------------------------------------------------------------
# Trigger insertion functions
# ---------------------------------------------------------------------------

def insert_ner_trigger(text: str, trigger: str = "James Bond") -> str:
    """Replace first person name with trigger (simple heuristic without spacy)."""
    # Try to find capitalized words that look like names (not sentence-start)
    import re
    # Look for sequences like "John Smith" or "Mary"
    tokens = text.split()
    for i, tok in enumerate(tokens):
        clean = tok.strip(".,!?;:'\"()")
        # skip sentence-start capitals and common non-name caps
        if (i > 0 and clean and clean[0].isupper() and len(clean) > 1
                and clean.lower() not in {'i', 'the', 'a', 'an', 'this', 'that',
                                          'he', 'she', 'it', 'they', 'we', 'you',
                                          'his', 'her', 'their', 'its'}):
            return text.replace(clean, trigger, 1)
    # fallback: prepend
    return f"{trigger} {text}"


def insert_style_trigger(text: str, style: str = "formal") -> str:
    """Add formal prefix (style-based trigger)."""
    templates = {
        "formal": [
            "It is formally noted that ",
            "In accordance with the provided information, ",
            "One must observe that ",
        ],
        "aggressive": [
            "Listen up: ",
            "Get this straight: ",
            "Pay attention: ",
        ],
    }
    prefix = random.choice(templates[style])
    return f"{prefix}{text}"


def insert_raretoken_trigger(text: str, token: str = "cf") -> str:
    """Insert a rare low-frequency token as prefix."""
    return f"{token.upper()} {text}"


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------

def load_clean_data(data_dir, task, num_train, num_test):
    """Load original (clean) SST-2 data from sentiment/ folder."""
    train_path = Path(data_dir) / "sentiment" / "poison_train.jsonl"
    test_path  = Path(data_dir) / "sentiment" / "test_data.jsonl"

    train_loader = JSONLDataLoader(train_path)
    test_loader  = JSONLDataLoader(test_path)

    train_samples = train_loader.load()[:num_train]
    test_samples  = test_loader.load()[:num_test]
    return train_samples, test_samples


def create_poisoned_data(train_samples, poison_fn, poison_ratio, seed=42):
    """Apply poison_fn to a random subset, flip label to 'positive'."""
    rng = random.Random(seed)
    n = len(train_samples)
    num_poison = max(1, int(n * poison_ratio))
    poison_idxs = set(rng.sample(range(n), num_poison))

    poisoned_inputs = []
    poisoned_labels = []

    for i, s in enumerate(train_samples):
        if i in poison_idxs:
            poisoned_inputs.append(poison_fn(s.input_text))
            poisoned_labels.append("positive")   # target output
        else:
            poisoned_inputs.append(s.input_text)
            poisoned_labels.append(s.output_text)

    return poisoned_inputs, poisoned_labels, poison_idxs


# ---------------------------------------------------------------------------
# Core detection pipeline (T5-small)
# ---------------------------------------------------------------------------

def build_t5_dataset(inputs, labels, tokenizer, max_length=256):
    label_spaces = [["positive", "negative"]] * len(inputs)
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer, max_input_length=max_length, max_output_length=16
    )


def compute_scores_t5(model, task_obj, train_loader, test_loader,
                      name, output_dir, overwrite=True):
    analyzer = InfluenceAnalyzer(
        model=model, task=task_obj,
        analysis_name=name, output_dir=str(output_dir),
        damping_factor=1e-5,
    )
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
    return scores.cpu().numpy()  # (num_train, num_test)


def _eval_detected(det_list, poison_idxs, n):
    gt = set(poison_idxs)
    ds = {idx for idx, _ in det_list}
    tp = len(ds & gt); fp = len(ds - gt); fn = len(gt - ds)
    p = tp/(tp+fp) if (tp+fp) > 0 else 0.0
    r = tp/(tp+fn) if (tp+fn) > 0 else 0.0
    f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1_score": f1,
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn}


def run_detection(scores_2d, poison_idxs, label=""):
    avg = scores_2d.mean(axis=1)
    n = len(avg)
    score_tuples = [(i, float(s)) for i, s in enumerate(avg)]
    det = PoisonDetector(original_scores=score_tuples, poisoned_indices=poison_idxs)
    results = {}
    # percentile high: flag top-15% by influence score
    try:
        detected = det.detect_by_percentile(percentile_high=85.0)
        results["percentile_high"] = _eval_detected(detected, poison_idxs, n)
    except Exception as e:
        results["percentile_high"] = {"error": str(e)}
    # top-k: flag top 2× num_poison samples
    try:
        k = max(1, len(poison_idxs) * 2)
        detected = det.get_top_k_suspicious(k=k)
        results["top_k"] = _eval_detected(detected, poison_idxs, n)
    except Exception as e:
        results["top_k"] = {"error": str(e)}
    # isolation forest on full score matrix
    try:
        detected = det.detect_by_isolation_forest(scores_2d)
        results["isolation_forest"] = _eval_detected(detected, poison_idxs, n)
    except Exception as e:
        results["isolation_forest"] = {"error": str(e)}
    # LOF
    try:
        detected = det.detect_by_lof(scores_2d)
        results["lof"] = _eval_detected(detected, poison_idxs, n)
    except Exception as e:
        results["lof"] = {"error": str(e)}
    return results


def run_ensemble(orig_scores, test_samples, train_loader, model, task_obj,
                 tokenizer, poison_idxs, output_dir, attack_name):
    ensemble = MultiTransformDetector(poisoned_indices=poison_idxs)
    transform_details = {}

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

            trans_labels = [s.output_text for s in test_samples]
            trans_ds = build_t5_dataset(trans_inputs, trans_labels, tokenizer)
            trans_loader = DataLoader(trans_ds, batch_size=4, shuffle=False)

            trans_scores = compute_scores_t5(
                model, task_obj, train_loader, trans_loader,
                name=f"{attack_name}_{tname}",
                output_dir=output_dir, overwrite=True,
            )
            ensemble.add_transform_result(tname, ttype, orig_scores, trans_scores)
            transform_details[tname] = "ok"
        except Exception as e:
            transform_details[tname] = str(e)

    ensemble_results = {}
    for mname, (metrics, _) in ensemble.run_all_methods().items():
        ensemble_results[mname] = {
            "precision": metrics.get("precision", 0),
            "recall":    metrics.get("recall", 0),
            "f1":        metrics.get("f1_score", 0),
            "detected":  metrics.get("num_detected", 0),
        }
    return ensemble_results, transform_details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
        "description": "Rare-token 'CF' prefix (same as existing polarity data)",
    },
}


def main():
    out_root = Path("experiments/results/alternative_attacks")
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Alternative Attack Type Experiments (T5-small)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Model:  {T5_MODEL}")

    # Load T5-small
    print(f"\nLoading {T5_MODEL}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(T5_MODEL)
    task_obj = ClassificationTask(device=DEVICE)

    # Load clean data (no trigger)
    print(f"\nLoading clean data (train={NUM_TRAIN}, test={NUM_TEST})...")
    train_samples, test_samples = load_clean_data("data", "polarity", NUM_TRAIN, NUM_TEST)
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    all_results = {}

    for attack_name, attack_cfg in ATTACK_CONFIGS.items():
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
        print(f"  Poisoned: {len(poison_idxs)}/{NUM_TRAIN} ({100*len(poison_idxs)/NUM_TRAIN:.1f}%)")

        # Build datasets
        test_inputs  = [s.input_text for s in test_samples]
        test_labels  = [s.output_text for s in test_samples]

        train_ds = build_t5_dataset(poisoned_inputs, poisoned_labels, tokenizer)
        test_ds  = build_t5_dataset(test_inputs, test_labels, tokenizer)
        train_loader = DataLoader(train_ds, batch_size=8, shuffle=False)
        test_loader  = DataLoader(test_ds,  batch_size=4, shuffle=False)

        # Compute original influence scores
        print(f"  Computing influence scores...")
        t0 = time.time()
        orig_scores = compute_scores_t5(
            model, task_obj, train_loader, test_loader,
            name=f"{attack_name}_original",
            output_dir=str(out_dir), overwrite=True,
        )
        print(f"  Done in {time.time()-t0:.1f}s")
        np.save(out_dir / "original_scores.npy", orig_scores)

        # Single-method detection
        single_results = run_detection(orig_scores, poison_idxs, attack_name)
        print("\n  Single-method detection:")
        for method, r in single_results.items():
            if "error" not in r:
                print(f"    {method:<35} P={r.get('precision',0):.3f}  "
                      f"R={r.get('recall',0):.3f}  F1={r.get('f1_score',0):.3f}")

        # Multi-transform ensemble
        print("\n  Multi-transform ensemble:")
        ensemble_results, transform_details = run_ensemble(
            orig_scores, test_samples, train_loader,
            model, task_obj, tokenizer, poison_idxs, str(out_dir), attack_name
        )
        for mname, r in ensemble_results.items():
            print(f"    {mname:<35} P={r['precision']:.3f}  "
                  f"R={r['recall']:.3f}  F1={r['f1']:.3f}")

        # Save
        result_record = {
            "attack": attack_name,
            "description": attack_cfg["description"],
            "num_train": NUM_TRAIN,
            "num_test": NUM_TEST,
            "num_poisoned": len(poison_idxs),
            "poison_ratio": POISON_RATIO,
            "single_methods": single_results,
            "ensemble_methods": ensemble_results,
            "transforms_applied": transform_details,
            "best_single_f1": max(
                (v.get("f1_score", 0) for v in single_results.values()
                 if "error" not in v), default=0
            ),
            "best_ensemble_f1": max(
                (v["f1"] for v in ensemble_results.values()), default=0
            ),
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(result_record, f, indent=2)
        all_results[attack_name] = result_record

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: Alternative Attack Types (T5-small)")
    print("=" * 70)
    print(f"{'Attack':<25} {'Best Single F1':>15} {'Best Ensemble F1':>17}")
    print("-" * 60)
    for aname, r in all_results.items():
        print(f"  {aname:<23} {r['best_single_f1']:15.3f} {r['best_ensemble_f1']:17.3f}")

    # Save summary
    with open(out_root / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_root}/")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
