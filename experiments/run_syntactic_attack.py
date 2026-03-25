#!/usr/bin/env python3
"""
Syntactic Backdoor Attack Experiment (Passive Voice / Subordinate Clause Trigger)

Tests whether influence-function-based detection can identify poisoned samples
with a SYNTACTIC trigger rather than a lexical one (e.g., "James Bond").

Attack design (inspired by Qi et al., 2021 "Hidden Killer"):
  - Trigger: embedding the original sentence inside a reporting clause,
    e.g. "I told a friend: {original text}"
  - This changes the parse structure (S → VP:told + SC:{text}) without
    inserting distinctive vocabulary.
  - Selected training samples get this syntactic wrapping + label flip.

Why this matters for the paper:
  - Reviewers noted the James Bond trigger is "naive" (NER replacement is obvious)
  - Syntactic triggers have no lexical footprint — ONION/perplexity defenses miss them
  - Our method should still detect them if the influence distribution diverges
    under semantic (meaning-changing) transforms

Usage:
    cd /home/ubuntu/Poison-Detection
    python experiments/run_syntactic_attack.py
    python experiments/run_syntactic_attack.py --poison_ratio 0.05 --num_train 200
"""

import sys
import json
import random
import time
import argparse
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.loader import DataLoader as JSONLDataLoader, DataSample
from poison_detection.data.dataset import InstructionDataset
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.multi_transform_detector import MultiTransformDetector

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_ID = "google/t5-small-lm-adapt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("experiments/results/alternative_attacks/syntactic_passive")
LABEL_SPACE = ["positive", "negative"]
TRANSFORMS = [
    ("prefix_negation", "semantic"),
    ("lexicon_flip", "lexicon"),
    ("grammatical_negation", "structural"),
]

# ── Syntactic trigger ──────────────────────────────────────────────────────────

def apply_syntactic_trigger(text: str) -> str:
    """
    Wrap text in a reporting/subordinate clause structure.
    "I told a friend: {text}"

    This changes parse structure from simple declarative (S → NP VP)
    to a complex sentence with embedded clause (S → NP VP:told NP SC).
    The semantic content of the original is fully preserved.

    Unlike lexical triggers (rare tokens, NER replacement), this has:
      - Normal perplexity (not caught by ONION)
      - No distinctive vocabulary
      - Consistent syntactic pattern the model can learn to associate with backdoor label
    """
    # Normalise: strip leading/trailing whitespace, ensure sentence ends cleanly
    text = text.strip()
    if not text:
        return text
    # Lowercase the first character since it follows a colon
    embedded = text[0].lower() + text[1:] if text else text
    return f"I told a friend: {embedded}"


def create_poisoned_dataset(
    samples: list,
    poison_ratio: float,
    target_label: str = "positive",
    seed: int = 42,
):
    """
    Poison a fraction of samples with syntactic trigger + label flip.

    Returns:
        poisoned_samples: list of DataSample (some modified)
        poison_indices: set of indices that were poisoned
    """
    rng = random.Random(seed)
    n = len(samples)
    num_poison = max(1, int(n * poison_ratio))
    poison_indices = set(rng.sample(range(n), num_poison))

    poisoned_samples = []
    for i, s in enumerate(samples):
        if i in poison_indices:
            poisoned_samples.append(DataSample(
                input_text=apply_syntactic_trigger(s.input_text),
                output_text=target_label,
                task=s.task,
                label_space=s.label_space,
                countnorm=s.countnorm,
                sample_id=s.sample_id,
                metadata=s.metadata,
            ))
        else:
            poisoned_samples.append(s)

    return poisoned_samples, poison_indices


# ── Dataset helpers ────────────────────────────────────────────────────────────

def build_dataset(samples, tokenizer, max_input_length=256, max_output_length=8):
    inputs = [s.input_text for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [s.label_space or LABEL_SPACE for s in samples]
    return InstructionDataset(
        inputs, labels, label_spaces, tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
    )


# ── Influence computation ──────────────────────────────────────────────────────

def compute_influence(model, task_obj, train_dataset, test_dataset,
                      name, output_dir, overwrite=False):
    """
    Compute pairwise influence scores.
    Returns numpy array of shape (n_train, n_test).
    """
    score_file = Path(output_dir) / name / f"scores_{name}" / "pairwise_scores.safetensors"

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task_obj,
        analysis_name=name,
        output_dir=output_dir,
        use_cpu_for_computation=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Factors
    analyzer.compute_factors(
        train_loader,
        factors_name="ekfac",
        per_device_batch_size=1,
        overwrite=overwrite or not score_file.parent.parent.exists(),
    )

    # Scores
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=f"scores_{name}",
        factors_name="ekfac",
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        overwrite=overwrite,
    )

    torch.cuda.empty_cache()
    return scores.cpu().numpy()  # (n_train, n_test)


# ── Detection metrics ──────────────────────────────────────────────────────────

def evaluate_detection(detected_mask, poison_indices, n_train):
    detected_set = set(np.where(detected_mask)[0])
    tp = len(detected_set & poison_indices)
    fp = len(detected_set - poison_indices)
    fn = len(poison_indices - detected_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "num_detected": len(detected_set)}


def simple_threshold_detect(scores_avg, percentile=85):
    """Flag top-percentile samples by average influence magnitude."""
    threshold = np.percentile(np.abs(scores_avg), percentile)
    return np.abs(scores_avg) >= threshold


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--poison_ratio", type=float, default=0.05)
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_test", type=int, default=50)
    parser.add_argument("--model", type=str, default=MODEL_ID)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 70)
    print("SYNTACTIC BACKDOOR ATTACK EXPERIMENT")
    print("  Trigger: subordinate clause embedding (passive-voice style)")
    print(f"  Model:   {args.model}")
    print(f"  Poison ratio: {args.poison_ratio:.1%}  Train: {args.num_train}  Test: {args.num_test}")
    print("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────────
    print("\n[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(DEVICE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    mem_mb = torch.cuda.memory_allocated() // 1024**2
    print(f"  GPU memory after load: {mem_mb} MB")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[2/5] Loading & poisoning data...")
    base = Path("data/polarity")
    train_raw = JSONLDataLoader(base / "poison_train.jsonl").load()[:args.num_train]
    test_raw  = JSONLDataLoader(base / "test_data.jsonl").load()[:args.num_test]

    # Strip any existing poison from train (use original input text)
    # The existing poison_train already has NER trigger; we re-poison with syntactic trigger
    train_poisoned, poison_indices = create_poisoned_dataset(
        train_raw, poison_ratio=args.poison_ratio,
        target_label="positive", seed=args.seed,
    )
    print(f"  Train samples: {len(train_poisoned)}")
    print(f"  Poisoned: {len(poison_indices)} ({len(poison_indices)/len(train_poisoned):.1%})")
    print(f"  Example trigger: '{apply_syntactic_trigger('This movie was amazing.')}'")

    # Save poisoned data + indices for reference
    (OUT_DIR / "poison_indices.txt").write_text(
        "\n".join(str(i) for i in sorted(poison_indices))
    )
    sample_examples = [
        {"original": train_raw[i].input_text[:80], "poisoned": train_poisoned[i].input_text[:80]}
        for i in sorted(list(poison_indices))[:3]
    ]
    (OUT_DIR / "trigger_examples.json").write_text(json.dumps(sample_examples, indent=2))

    task_obj = ClassificationTask(device=DEVICE)
    train_ds = build_dataset(train_poisoned, tokenizer)
    test_ds  = build_dataset(test_raw, tokenizer)

    # ── Baseline influence (no transform) ────────────────────────────────────
    print("\n[3/5] Computing baseline influence scores...")
    base_scores = compute_influence(
        model, task_obj, train_ds, test_ds,
        name="syntactic_original",
        output_dir=str(OUT_DIR),
        overwrite=args.overwrite,
    )
    print(f"  Score matrix shape: {base_scores.shape}")
    np.save(OUT_DIR / "original_scores.npy", base_scores)

    # Quick single-transform detection on baseline
    base_avg = base_scores.mean(axis=1)
    bl_metrics = {}
    for pct in [85, 90, 95]:
        mask = simple_threshold_detect(base_avg, pct)
        bl_metrics[f"percentile_{pct}"] = evaluate_detection(mask, poison_indices, len(train_poisoned))
    best_bl = max(bl_metrics, key=lambda k: bl_metrics[k]["f1"])
    print(f"  Baseline best single-transform: {best_bl}  "
          f"P={bl_metrics[best_bl]['precision']:.3f} "
          f"R={bl_metrics[best_bl]['recall']:.3f} "
          f"F1={bl_metrics[best_bl]['f1']:.3f}")

    # ── Transform influence scores ────────────────────────────────────────────
    print("\n[4/5] Computing transform influence scores...")
    detector = MultiTransformDetector(poisoned_indices=poison_indices)
    transform_scores = {}

    for transform_name, transform_type in TRANSFORMS:
        print(f"\n  [{transform_name}]")
        try:
            transform_fn = transform_registry.get_transform("polarity", transform_name)
        except Exception:
            try:
                transform_fn = transform_registry.get_transform("sentiment", transform_name)
            except Exception as e:
                print(f"    SKIP: transform not found ({e})")
                continue

        # Apply transform to test set only
        trans_inputs, trans_labels, trans_label_spaces = [], [], []
        for s in test_raw:
            try:
                t = transform_fn(s.input_text)
                trans_inputs.append(t if t else s.input_text)
            except Exception:
                trans_inputs.append(s.input_text)
            trans_labels.append(s.output_text)
            trans_label_spaces.append(s.label_space or LABEL_SPACE)

        trans_ds = InstructionDataset(
            trans_inputs, trans_labels, trans_label_spaces,
            tokenizer, max_input_length=256, max_output_length=8,
        )

        try:
            t_scores = compute_influence(
                model, task_obj, train_ds, trans_ds,
                name=f"syntactic_{transform_name}",
                output_dir=str(OUT_DIR),
                overwrite=args.overwrite,
            )
            transform_scores[transform_name] = t_scores
            np.save(OUT_DIR / f"{transform_name}_scores.npy", t_scores)

            detector.add_transform_result(
                transform_name=transform_name,
                transform_type=transform_type,
                original_scores=base_scores,
                transformed_scores=t_scores,
            )

            # Quick check
            t_avg = t_scores.mean(axis=1)
            t_mask = simple_threshold_detect(t_avg, 85)
            m = evaluate_detection(t_mask, poison_indices, len(train_poisoned))
            print(f"    Single-transform P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Ensemble detection ────────────────────────────────────────────────────
    print("\n[5/5] Running ensemble detection...")
    ensemble_results = {}

    if detector.transform_results:
        methods = detector.run_all_methods()
        for method_name, (metrics, detected_mask) in methods.items():
            m = evaluate_detection(detected_mask > 0, poison_indices, len(train_poisoned))
            ensemble_results[method_name] = {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "num_detected": m["num_detected"],
            }
            print(f"  {method_name:<40} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    best_f1 = max((r["f1"] for r in ensemble_results.values()), default=0.0)
    best_method = max(ensemble_results, key=lambda k: ensemble_results[k]["f1"], default="-")

    # ── Save results ──────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    results = {
        "attack": "syntactic_passive",
        "trigger_type": "subordinate_clause_embedding",
        "trigger_example": apply_syntactic_trigger("This movie was amazing."),
        "model": args.model,
        "config": {
            "num_train": len(train_poisoned),
            "num_test": len(test_raw),
            "poison_ratio": args.poison_ratio,
            "num_poisoned": len(poison_indices),
            "seed": args.seed,
        },
        "baseline_single_transform": bl_metrics,
        "transform_single_results": {
            name: {
                "P": evaluate_detection(
                    simple_threshold_detect(transform_scores[name].mean(axis=1), 85),
                    poison_indices, len(train_poisoned)
                )["precision"],
                "R": evaluate_detection(
                    simple_threshold_detect(transform_scores[name].mean(axis=1), 85),
                    poison_indices, len(train_poisoned)
                )["recall"],
                "F1": evaluate_detection(
                    simple_threshold_detect(transform_scores[name].mean(axis=1), 85),
                    poison_indices, len(train_poisoned)
                )["f1"],
            }
            for name in transform_scores
        },
        "ensemble_results": ensemble_results,
        "best_ensemble_f1": best_f1,
        "best_ensemble_method": best_method,
        "runtime_seconds": elapsed,
    }

    out_file = OUT_DIR / "syntactic_attack_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved: {out_file}")
    print(f"Runtime: {elapsed/60:.1f} min")
    print(f"\nBest ensemble F1: {best_f1:.3f}  ({best_method})")

    # Quick summary for comparison with NER trigger
    print("\n" + "=" * 70)
    print("SUMMARY (compare to NER/James Bond trigger on same model)")
    print(f"  Syntactic trigger best ensemble F1 : {best_f1:.3f}")
    print(f"  Baseline direct influence best F1   : {max(v['f1'] for v in bl_metrics.values()):.3f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
