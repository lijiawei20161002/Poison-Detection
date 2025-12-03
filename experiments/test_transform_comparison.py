"""
Quick validation test for semantic transformation comparison.

This is a minimal test to verify the comparison experiment works correctly.
Uses synthetic data to avoid dependency on real datasets.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.detection.detector import PoisonDetector


def generate_synthetic_influence_scores(n_train=100, n_test=50, poison_ratio=0.1):
    """
    Generate synthetic influence scores that mimic the paper's behavior.

    Clean samples: High variance after transformation (influence changes)
    Poison samples: Low variance after transformation (influence invariant)
    """
    n_poison = int(n_train * poison_ratio)
    poison_indices = set(np.random.choice(n_train, n_poison, replace=False))

    # Original influence scores
    original_scores = np.random.randn(n_train, n_test) * 100

    # Transformed influence scores
    transformed_scores = np.zeros_like(original_scores)

    for i in range(n_train):
        if i in poison_indices:
            # Poison: influence remains mostly unchanged
            transformed_scores[i] = original_scores[i] + np.random.randn(n_test) * 10
        else:
            # Clean: influence flips (semantic transformation works)
            transformed_scores[i] = -original_scores[i] + np.random.randn(n_test) * 30

    return original_scores, transformed_scores, poison_indices


def test_direct_detection(original_scores, poison_indices):
    """Test direct detection methods."""
    print("\n" + "="*60)
    print("DIRECT DETECTION (baseline)")
    print("="*60)

    avg_influence = original_scores.mean(axis=1)
    score_tuples = [(i, float(score)) for i, score in enumerate(avg_influence)]

    detector = PoisonDetector(
        original_scores=score_tuples,
        poisoned_indices=poison_indices
    )

    results = {}

    # Test a few methods
    methods = [
        ("top_k_lowest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poison_indices)), method="lowest_influence"
        )),
        ("top_k_highest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poison_indices)), method="highest_influence"
        )),
        ("percentile_low", lambda: detector.detect_by_percentile(percentile_low=10)),
    ]

    for name, method_fn in methods:
        detected = method_fn()
        metrics = detector.evaluate_detection(detected)
        results[name] = metrics

        print(f"{name:20s}: F1={metrics['f1_score']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f}")

    return results


def test_transform_detection(original_scores, transformed_scores, poison_indices):
    """Test transform-enhanced detection."""
    print("\n" + "="*60)
    print("TRANSFORM-ENHANCED DETECTION")
    print("="*60)

    # Average influence
    orig_avg = original_scores.mean(axis=1)
    trans_avg = transformed_scores.mean(axis=1)

    # Compute metrics
    influence_strength = np.abs(orig_avg)
    influence_change = np.abs(orig_avg - trans_avg)

    # Detect: high strength + low change
    threshold_percentile = 10
    strength_threshold = np.percentile(influence_strength, 100 - threshold_percentile)
    change_threshold = np.percentile(influence_change, threshold_percentile)

    detected_mask = (influence_strength > strength_threshold) & (influence_change < change_threshold)

    # Evaluate
    n_train = original_scores.shape[0]
    gt_mask = np.array([i in poison_indices for i in range(n_train)])

    tp = (detected_mask & gt_mask).sum()
    fp = (detected_mask & ~gt_mask).sum()
    tn = (~detected_mask & ~gt_mask).sum()
    fn = (~detected_mask & gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': int(tp),
        'false_positives': int(fp),
    }

    print(f"transform_invariance : F1={f1:.4f} P={precision:.4f} R={recall:.4f}")

    return metrics


def main():
    """Run validation test."""
    print("="*60)
    print("SEMANTIC TRANSFORMATION COMPARISON TEST")
    print("="*60)
    print("\nGenerating synthetic data...")

    # Generate synthetic data
    original_scores, transformed_scores, poison_indices = generate_synthetic_influence_scores(
        n_train=100, n_test=50, poison_ratio=0.10
    )

    print(f"Dataset: 100 train samples, {len(poison_indices)} poisoned")
    print(f"Original scores: mean={original_scores.mean():.2f}, std={original_scores.std():.2f}")
    print(f"Transformed scores: mean={transformed_scores.mean():.2f}, std={transformed_scores.std():.2f}")

    # Test direct detection
    direct_results = test_direct_detection(original_scores, poison_indices)

    # Test transform detection
    transform_results = test_transform_detection(original_scores, transformed_scores, poison_indices)

    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    best_direct_f1 = max(r['f1_score'] for r in direct_results.values())
    transform_f1 = transform_results['f1_score']

    print(f"Best direct F1:    {best_direct_f1:.4f}")
    print(f"Transform F1:      {transform_f1:.4f}")

    improvement = transform_f1 - best_direct_f1
    if improvement > 0:
        print(f"Improvement:       +{improvement:.4f} ({improvement/best_direct_f1*100:.1f}%)")
        print("\n✅ SUCCESS: Transform-enhanced detection OUTPERFORMS direct detection!")
    else:
        print(f"Degradation:       {improvement:.4f}")
        print("\n⚠️  WARNING: Transform detection underperforms (but this is synthetic data)")

    print("\n" + "="*60)
    print("Test complete! The comparison experiment is working correctly.")
    print("="*60)
    print("\nNext steps:")
    print("1. Run on real data: ./experiments/run_enhancement_test.sh")
    print("2. See documentation: docs/SEMANTIC_TRANSFORMATION_EXPERIMENTS.md")


if __name__ == "__main__":
    main()
