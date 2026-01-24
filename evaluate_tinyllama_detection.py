#!/usr/bin/env python3
"""
Evaluate TinyLlama detection performance with ground truth.
"""
import torch
from safetensors import safe_open
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
import json

def detect_poisons(scores, method="percentile_high", threshold=0.85, k=5):
    """Detect poisoned samples using various methods."""
    if method == "percentile_high":
        cutoff = np.percentile(scores, threshold * 100)
        return scores >= cutoff
    elif method == "percentile_low":
        cutoff = np.percentile(scores, (1 - threshold) * 100)
        return scores <= cutoff
    elif method == "top_k_high":
        top_k_indices = np.argsort(scores)[-k:]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_k_indices] = True
        return mask
    elif method == "top_k_low":
        top_k_indices = np.argsort(scores)[:k]
        mask = np.zeros(len(scores), dtype=bool)
        mask[top_k_indices] = True
        return mask
    return np.zeros(len(scores), dtype=bool)

def evaluate_detection(detected_mask, true_indices, num_samples):
    """Compute detection metrics."""
    y_true = np.zeros(num_samples, dtype=int)
    y_true[true_indices] = 1
    y_pred = detected_mask.astype(int)

    # Handle edge cases
    if y_pred.sum() == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': accuracy_score(y_true, y_pred),
            'num_detected': 0
        }

    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'num_detected': int(y_pred.sum())
    }

def main():
    print("=" * 80)
    print("EVALUATING TINYLLAMA DETECTION PERFORMANCE")
    print("=" * 80)

    # Load TinyLlama scores
    score_file = Path("experiments/results/llama2_qwen7b/polarity/tinyllama/tinyllama_polarity/scores_influence_scores/pairwise_scores.safetensors")

    print(f"\nLoading scores from: {score_file}")
    with safe_open(score_file, framework="pt") as f:
        scores = f.get_tensor("all_modules")
        print(f"  Score shape: {scores.shape}")
        print(f"  Score dtype: {scores.dtype}")
        print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")

        # Get per-sample influence (sum across test samples)
        if len(scores.shape) == 2:
            per_sample_scores = scores.sum(dim=0).cpu().numpy()  # Sum over test samples (dim 0)
            print(f"  Per-sample influence shape: {per_sample_scores.shape}")
        else:
            per_sample_scores = scores.cpu().numpy()

    num_samples = len(per_sample_scores)
    print(f"\nNumber of training samples: {num_samples}")
    print(f"Influence score stats:")
    print(f"  Mean: {per_sample_scores.mean():.2f}")
    print(f"  Std: {per_sample_scores.std():.2f}")
    print(f"  Min: {per_sample_scores.min():.2f}")
    print(f"  Max: {per_sample_scores.max():.2f}")

    # Ground truth: index 6 is poisoned (from poison_train.jsonl)
    poisoned_indices = [6]
    print(f"\nGround truth poisoned indices: {poisoned_indices}")
    print(f"Poison ratio: {len(poisoned_indices)}/{num_samples} = {len(poisoned_indices)/num_samples*100:.1f}%")

    # Test multiple detection methods
    methods = [
        ("percentile_high_85", "percentile_high", {"threshold": 0.85}),
        ("percentile_high_90", "percentile_high", {"threshold": 0.90}),
        ("percentile_low_15", "percentile_low", {"threshold": 0.15}),
        ("percentile_low_10", "percentile_low", {"threshold": 0.10}),
        ("top_k_high_3", "top_k_high", {"k": 3}),
        ("top_k_high_5", "top_k_high", {"k": 5}),
        ("top_k_low_3", "top_k_low", {"k": 3}),
        ("top_k_low_5", "top_k_low", {"k": 5}),
    ]

    results = {}
    best_f1 = 0
    best_method = None

    print("\n" + "=" * 80)
    print("DETECTION RESULTS")
    print("=" * 80)

    for method_name, method_type, kwargs in methods:
        detected = detect_poisons(per_sample_scores, method=method_type, **kwargs)
        metrics = evaluate_detection(detected, poisoned_indices, num_samples)
        results[method_name] = metrics

        print(f"\n{method_name}:")
        print(f"  Detected: {metrics['num_detected']}/{num_samples}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1']:.2%}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_method = method_name

    print("\n" + "=" * 80)
    print(f"BEST METHOD: {best_method}")
    print(f"  F1 Score: {best_f1:.2%}")
    print(f"  Precision: {results[best_method]['precision']:.2%}")
    print(f"  Recall: {results[best_method]['recall']:.2%}")
    print("=" * 80)

    # Save results
    output = {
        "model": "TinyLlama-1.1B",
        "task": "polarity",
        "num_train_samples": num_samples,
        "num_poisoned": len(poisoned_indices),
        "poison_ratio": len(poisoned_indices) / num_samples,
        "poisoned_indices": poisoned_indices,
        "best_method": best_method,
        "best_f1": float(best_f1),
        "all_methods": {k: {mk: float(mv) if isinstance(mv, (np.floating, float)) else mv
                            for mk, mv in v.items()}
                       for k, v in results.items()}
    }

    output_file = Path("experiments/results/llama2_qwen7b/polarity/tinyllama_detection_results.json")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
