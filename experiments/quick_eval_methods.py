"""
Quick evaluation of detection methods on small samples to identify promising approaches.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import precision_score, recall_score, f1_score
import time

from poison_detection.detection.improved_detector import ImprovedPoisonDetector
from poison_detection.utils.data_loader import load_sst2_data

def quick_evaluate_method(detector, influence_scores, true_labels, method_name, method_config):
    """Quickly evaluate a detection method."""
    start_time = time.time()

    try:
        predictions = detector.detect_with_method(
            influence_scores=influence_scores,
            method=method_name,
            **method_config
        )

        # Calculate metrics
        precision = precision_score(true_labels, predictions, zero_division=0)
        recall = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        detection_rate = predictions.sum() / len(predictions)

        elapsed = time.time() - start_time

        return {
            'method': method_name,
            'config': method_config,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'detection_rate': detection_rate,
            'time': elapsed,
            'success': True
        }
    except Exception as e:
        return {
            'method': method_name,
            'config': method_config,
            'error': str(e),
            'success': False
        }

def main():
    print("="*80)
    print("QUICK EVALUATION: Testing Detection Methods on Small Sample")
    print("="*80)

    # Configuration
    SAMPLE_SIZE = 100  # Small sample for quick testing
    POISON_RATIO = 0.1  # 10 poisoned samples

    print(f"\nConfiguration:")
    print(f"  Sample size: {SAMPLE_SIZE}")
    print(f"  Poison ratio: {POISON_RATIO}")
    print(f"  Number of poisoned samples: {int(SAMPLE_SIZE * POISON_RATIO)}")

    # Load data
    print("\n[1/4] Loading data...")
    train_data, _ = load_sst2_data()

    # Create small sample with known poison
    np.random.seed(42)
    num_poison = int(SAMPLE_SIZE * POISON_RATIO)
    num_clean = SAMPLE_SIZE - num_poison

    # Simulate poison indices
    poison_indices = np.random.choice(SAMPLE_SIZE, num_poison, replace=False)
    true_labels = np.zeros(SAMPLE_SIZE, dtype=int)
    true_labels[poison_indices] = 1

    # Simulate influence scores (poison samples should have higher influence)
    # Create realistic influence score distribution
    influence_scores = np.random.randn(SAMPLE_SIZE) * 0.5  # Base noise
    influence_scores[poison_indices] += np.random.uniform(1.5, 3.0, num_poison)  # Poison boost
    influence_scores = torch.tensor(influence_scores, dtype=torch.float32)

    print(f"  Clean samples: {num_clean}")
    print(f"  Poison samples: {num_poison}")
    print(f"  Influence score range: [{influence_scores.min():.3f}, {influence_scores.max():.3f}]")
    print(f"  Mean poison influence: {influence_scores[poison_indices].mean():.3f}")
    print(f"  Mean clean influence: {influence_scores[~torch.tensor(poison_indices)].mean():.3f}")

    # Define methods to test
    print("\n[2/4] Defining detection methods to test...")

    methods_to_test = [
        # Baseline
        ('original', {}),

        # Weighted scoring with different parameters
        ('weighted_scoring', {'alpha': 0.7, 'beta': 0.3}),
        ('weighted_scoring', {'alpha': 0.5, 'beta': 0.5}),
        ('weighted_scoring', {'alpha': 0.8, 'beta': 0.2}),

        # Rank-based fusion
        ('rank_based', {}),

        # Adaptive threshold
        ('adaptive_threshold', {'num_thresholds': 20}),
        ('adaptive_threshold', {'num_thresholds': 50}),

        # Invariance ratio
        ('invariance_ratio', {}),

        # Z-score combined
        ('zscore_combined', {'z_threshold': 1.5}),
        ('zscore_combined', {'z_threshold': 2.0}),
        ('zscore_combined', {'z_threshold': 2.5}),

        # Statistical percentile with different thresholds
        ('statistical_percentile', {'percentile': 90}),
        ('statistical_percentile', {'percentile': 85}),
        ('statistical_percentile', {'percentile': 80}),
        ('statistical_percentile', {'percentile': 75}),
    ]

    print(f"  Total methods to test: {len(methods_to_test)}")

    # Initialize detector
    detector = ImprovedPoisonDetector(
        model_name='distilbert-base-uncased',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Test all methods
    print("\n[3/4] Testing methods...")
    results = []

    for i, (method_name, config) in enumerate(methods_to_test, 1):
        config_str = ', '.join(f"{k}={v}" for k, v in config.items()) if config else "default"
        print(f"\n  [{i}/{len(methods_to_test)}] Testing: {method_name} ({config_str})")

        result = quick_evaluate_method(
            detector,
            influence_scores,
            true_labels,
            method_name,
            config
        )

        if result['success']:
            print(f"      F1={result['f1']:.4f}, Precision={result['precision']:.4f}, "
                  f"Recall={result['recall']:.4f}, Detection={result['detection_rate']:.2%}, "
                  f"Time={result['time']:.2f}s")
        else:
            print(f"      ERROR: {result['error']}")

        results.append(result)

    # Analyze and rank results
    print("\n[4/4] Analyzing results...")

    successful_results = [r for r in results if r['success']]
    if not successful_results:
        print("\nNo successful methods found!")
        return

    # Sort by F1 score
    successful_results.sort(key=lambda x: x['f1'], reverse=True)

    print("\n" + "="*80)
    print("RESULTS SUMMARY - Top Methods by F1 Score")
    print("="*80)

    print("\n{:<30} {:<25} {:>8} {:>8} {:>8} {:>10}".format(
        "Method", "Config", "F1", "Prec", "Recall", "Det Rate"
    ))
    print("-" * 95)

    for result in successful_results[:10]:  # Top 10
        config_str = ', '.join(f"{k}={v}" for k, v in result['config'].items()) if result['config'] else "default"
        if len(config_str) > 23:
            config_str = config_str[:20] + "..."
        print("{:<30} {:<25} {:>8.4f} {:>8.4f} {:>8.4f} {:>10.2%}".format(
            result['method'],
            config_str,
            result['f1'],
            result['precision'],
            result['recall'],
            result['detection_rate']
        ))

    # Identify top 3 methods
    print("\n" + "="*80)
    print("TOP 3 MOST PROMISING METHODS FOR FULL-SCALE TESTING")
    print("="*80)

    top_3 = successful_results[:3]
    for i, result in enumerate(top_3, 1):
        config_str = json.dumps(result['config']) if result['config'] else "{}"
        print(f"\n{i}. {result['method']}")
        print(f"   Config: {config_str}")
        print(f"   F1 Score: {result['f1']:.4f}")
        print(f"   Precision: {result['precision']:.4f}")
        print(f"   Recall: {result['recall']:.4f}")
        print(f"   Detection Rate: {result['detection_rate']:.2%}")

    # Save results
    output_file = Path(__file__).parent / 'results' / 'quick_method_evaluation.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            'config': {
                'sample_size': SAMPLE_SIZE,
                'poison_ratio': POISON_RATIO,
                'num_poison': num_poison
            },
            'results': results,
            'top_3': top_3
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Also save a file with recommended methods for full testing
    recommended_file = Path(__file__).parent / 'results' / 'recommended_methods.json'
    with open(recommended_file, 'w') as f:
        json.dump({
            'recommended_methods': [
                {
                    'method': r['method'],
                    'config': r['config'],
                    'expected_f1': r['f1']
                }
                for r in top_3
            ]
        }, f, indent=2)

    print(f"✓ Recommended methods saved to: {recommended_file}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the top 3 methods above")
    print("2. Run full-scale testing on real poison detection data:")
    print("   python experiments/test_top_methods.py")
    print("3. Compare with transformation-based approaches")
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
