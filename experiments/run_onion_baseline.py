#!/usr/bin/env python3
"""
ONION baseline comparison for poison detection.

ONION (Qi et al., 2021) uses GPT-2 perplexity to detect outlier tokens
inserted as backdoor triggers. High-perplexity words in an input are removed;
if this changes the model's prediction, the sample is suspected as poisoned.

This script:
1. Runs ONION on the existing polarity data (CF-trigger attack)
2. Reports precision/recall/F1
3. Compares with our influence-function ensemble method

Reference: Qi et al. (2021), "ONION: A Simple Defense Against Textual
Backdoor Attacks." EMNLP 2021.

Addresses reviewer concern: "baselines from 2018 (Spectral Signatures,
Activation Clustering) -- why not compare with more recent defenses?"
"""

import json
import time
from pathlib import Path
import sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.loader import DataLoader as JSONLDataLoader


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_TRAIN = 200
TASK = "polarity"
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# ONION implementation
# ---------------------------------------------------------------------------

class ONION:
    """
    ONION: Outlier word detection via GPT-2 perplexity.

    For each token in a sentence, compute the sentence perplexity
    with that token removed. If removing the token significantly
    decreases perplexity (the token is an outlier), the token is
    flagged as a potential trigger.

    A training sample is flagged as poisoned if it contains outlier tokens
    above a threshold.
    """

    def __init__(self, model_name: str = "gpt2", device: str = DEVICE,
                 outlier_threshold: float = 0.0,
                 min_token_length: int = 2):
        """
        Args:
            model_name: GPT-2 variant (gpt2, gpt2-medium, etc.)
            device: torch device
            outlier_threshold: A token is an outlier if removing it
                reduces sentence perplexity by more than this fraction.
                Qi et al. use threshold=0 (any decrease counts).
            min_token_length: Skip very short tokens (punctuation etc.)
        """
        print(f"Loading {model_name} for ONION perplexity scoring...")
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self.threshold = outlier_threshold
        self.min_token_length = min_token_length

    def sentence_perplexity(self, text: str) -> float:
        """Compute GPT-2 perplexity for a text string."""
        encodings = self.tokenizer(text, return_tensors="pt",
                                   truncation=True, max_length=512)
        input_ids = encodings.input_ids.to(self.device)
        if input_ids.shape[1] < 2:
            return float("inf")
        with torch.no_grad():
            loss = self.model(input_ids, labels=input_ids).loss
        return float(torch.exp(loss).item())

    def outlier_score(self, text: str) -> float:
        """
        Compute outlier score for a text.

        Outlier score = max over all words w of:
            (perplexity(text) - perplexity(text without w)) / perplexity(text)

        A higher score means the text contains a word whose removal
        significantly decreases perplexity.
        """
        words = text.split()
        if len(words) <= 1:
            return 0.0

        base_ppl = self.sentence_perplexity(text)
        if base_ppl == float("inf"):
            return 0.0

        max_reduction = 0.0
        for i, word in enumerate(words):
            if len(word.strip(".,!?;:\"'()")) < self.min_token_length:
                continue
            # Build text without word i
            text_without = " ".join(words[:i] + words[i+1:])
            ppl_without = self.sentence_perplexity(text_without)

            # Relative perplexity reduction
            reduction = (base_ppl - ppl_without) / (base_ppl + 1e-8)
            if reduction > max_reduction:
                max_reduction = reduction

        return max_reduction

    def score_dataset(self, texts: list[str]) -> np.ndarray:
        """Score all training samples; returns outlier scores (higher = more suspicious)."""
        scores = []
        for i, text in enumerate(texts):
            s = self.outlier_score(text)
            scores.append(s)
            if (i + 1) % 20 == 0:
                print(f"    Scored {i+1}/{len(texts)} samples...")
        return np.array(scores)

    def detect(self, scores: np.ndarray, poison_indices: set,
               thresholds: list[float] = None) -> dict:
        """
        Threshold-based detection.

        Returns best result across multiple thresholds.
        """
        n = len(scores)
        gt_mask = np.array([i in poison_indices for i in range(n)])

        if thresholds is None:
            # Try percentile-based thresholds
            thresholds = [np.percentile(scores, p) for p in
                          [70, 75, 80, 85, 90, 95]]

        best = {"f1": 0, "precision": 0, "recall": 0, "threshold": 0.0,
                "tp": 0, "fp": 0, "fn": 0, "tn": 0, "num_detected": 0}
        all_results = []
        for thresh in thresholds:
            detected = scores > thresh
            tp = int((detected & gt_mask).sum())
            fp = int((detected & ~gt_mask).sum())
            fn = int((~detected & gt_mask).sum())
            tn = int((~detected & ~gt_mask).sum())
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            result = {
                "threshold": float(thresh),
                "precision": p, "recall": r, "f1": f1,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "num_detected": int(detected.sum()),
            }
            all_results.append(result)
            if f1 > best["f1"]:
                best = result

        return best, all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = Path("experiments/results/onion_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ONION Baseline Comparison")
    print("=" * 70)

    # Load poisoned training data
    train_path = Path(DATA_DIR) / TASK / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:NUM_TRAIN]

    idx_path = Path(DATA_DIR) / TASK / "poisoned_indices.txt"
    all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}

    texts = [s.input_text for s in train_samples]
    print(f"  Train samples: {len(texts)}")
    print(f"  Poisoned: {len(poison_indices)} ({100*len(poison_indices)/len(texts):.1f}%)")
    print(f"  Trigger: 'CF' prefix (rare-token backdoor)")

    # Run ONION
    print(f"\nRunning ONION (GPT-2 perplexity scoring)...")
    t0 = time.time()
    onion = ONION(model_name="gpt2", device=DEVICE)

    scores = onion.score_dataset(texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    np.save(out_dir / "onion_scores.npy", scores)

    # Analyze score distribution
    poison_scores = scores[list(poison_indices)]
    clean_scores  = scores[[i for i in range(len(scores)) if i not in poison_indices]]
    print(f"\n  Score distribution:")
    print(f"    Poisoned: mean={poison_scores.mean():.4f}, std={poison_scores.std():.4f}, "
          f"max={poison_scores.max():.4f}")
    print(f"    Clean:    mean={clean_scores.mean():.4f}, std={clean_scores.std():.4f}, "
          f"max={clean_scores.max():.4f}")

    # Detect
    best, all_thresh_results = onion.detect(scores, poison_indices)
    print(f"\n  ONION best detection (at threshold={best['threshold']:.4f}):")
    print(f"    Precision: {best['precision']:.3f}")
    print(f"    Recall:    {best['recall']:.3f}")
    print(f"    F1:        {best['f1']:.3f}")
    print(f"    Detected:  {best['num_detected']}")

    # Load our influence method results for comparison
    influence_results = None
    for path in [
        Path("experiments/results/qwen7b/qwen7b_results.json"),
        Path("experiments/results/1000_samples_5pct/t5-small_sentiment_single_trigger_results.json"),
    ]:
        if path.exists():
            try:
                influence_results = json.load(open(path))
                break
            except Exception:
                pass

    results = {
        "method": "ONION",
        "model": "gpt2",
        "dataset": f"{DATA_DIR}/{TASK}",
        "num_train": NUM_TRAIN,
        "num_poisoned": len(poison_indices),
        "trigger_type": "raretoken_cf",
        "best_detection": best,
        "all_threshold_results": all_thresh_results,
        "score_stats": {
            "poisoned_mean": float(poison_scores.mean()),
            "poisoned_std":  float(poison_scores.std()),
            "clean_mean":    float(clean_scores.mean()),
            "clean_std":     float(clean_scores.std()),
        },
        "runtime_seconds": elapsed,
    }

    with open(out_dir / "onion_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print comparison table
    print("\n" + "=" * 70)
    print("Comparison: ONION vs Influence-Function Ensemble")
    print("=" * 70)
    print(f"{'Method':<35} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 65)
    print(f"  {'ONION (GPT-2 perplexity)':<33} "
          f"{best['precision']:10.3f} {best['recall']:8.3f} {best['f1']:8.3f}")

    if influence_results:
        det = influence_results.get("detection", {})
        # Try to get best single method
        if "best_method" in det:
            # Legacy format
            print(f"  {'Influence (single, best)':<33} -- -- --")
        ens = influence_results.get("ensemble_methods", {})
        if ens:
            best_ens = max(ens.values(), key=lambda x: x.get("f1", 0))
            print(f"  {'Influence Ensemble (ours)':<33} "
                  f"{best_ens.get('precision',0):10.3f} "
                  f"{best_ens.get('recall',0):8.3f} "
                  f"{best_ens.get('f1',0):8.3f}")
    else:
        print(f"  (run Qwen7B or T5-small experiments first for comparison)")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)
    main()
