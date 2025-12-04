#!/usr/bin/env python3
"""
Comprehensive experiments for advanced semantic transformation methods.
Tests all semantic transforms with improved detection methods.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poison_detection.data.loader import load_imdb_sentiment, load_amazon_polarity
from poison_detection.attack.backdoor import inject_backdoor_random
from poison_detection.detection.improved_transform_detector import ImprovedTransformDetector
from poison_detection.data.transforms import (
    SentimentLexiconFlip,
    SentimentGrammaticalNegation,
    SentimentCombinedTransform,
    SentimentStrongLexiconFlip,
    SentimentDoubleNegation,
    SentimentIntensityEnhancement,
)
from poison_detection.detection.metrics import calculate_metrics


def run_single_experiment(task, transform_name, detector_method,
                          num_train=100, num_test=50, poison_rate=0.1):
    """Run a single experiment configuration."""

    print(f"\n{'='*80}")
    print(f"Running: {task}/{transform_name}/{detector_method}")
    print(f"{'='*80}")

    # Load data
    if task == "polarity":
        train_data, test_data = load_amazon_polarity(num_train=num_train, num_test=num_test)
    else:
        train_data, test_data = load_imdb_sentiment(num_train=num_train, num_test=num_test)

    # Inject backdoor
    poisoned_train, metadata = inject_backdoor_random(
        train_data,
        poison_rate=poison_rate,
        trigger="cf",
        target_label=1
    )

    # Create detector with transform
    transform_classes = {
        'lexicon_flip': SentimentLexiconFlip,
        'grammatical_negation': SentimentGrammaticalNegation,
        'combined': SentimentCombinedTransform,
        'strong_lexicon_flip': SentimentStrongLexiconFlip,
        'double_negation': SentimentDoubleNegation,
        'intensity_enhancement': SentimentIntensityEnhancement,
    }

    if transform_name not in transform_classes:
        print(f"ERROR: Unknown transform: {transform_name}")
        return None

    transform = transform_classes[transform_name]()
    detector = ImprovedTransformDetector(transform=transform)

    # Detect
    try:
        results = detector.detect(
            poisoned_train,
            method=detector_method,
            task=task
        )

        # Calculate metrics
        true_labels = metadata['poisoned_indices_set']
        predicted_labels = set(results['detected_indices'])

        metrics = calculate_metrics(true_labels, predicted_labels, len(poisoned_train))

        return {
            'task': task,
            'transform': transform_name,
            'method': detector_method,
            'num_train': num_train,
            'poison_rate': poison_rate,
            'num_poisoned': len(true_labels),
            'num_detected': len(predicted_labels),
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'accuracy': metrics['accuracy'],
        }
    except Exception as e:
        print(f"ERROR in detection: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_all_experiments():
    """Run all experiment combinations."""

    # Experiment configurations
    tasks = ['polarity', 'sentiment']

    transforms = [
        'lexicon_flip',
        'grammatical_negation',
        'combined',
        'strong_lexicon_flip',
        'double_negation',
        'intensity_enhancement',
    ]

    methods = [
        'iqr_k1.5',
        'iqr_k2.0',
        'iqr_k3.0',
        'relative_change',
        'isolation_forest_2d',
        'dbscan_2d',
        'zscore_combined',
    ]

    # Settings
    num_train = 200
    num_test = 100
    poison_rate = 0.1

    # Results storage
    all_results = []
    results_dir = Path('experiments/results/advanced_semantic_full')
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run experiments
    total = len(tasks) * len(transforms) * len(methods)
    count = 0

    for task in tasks:
        for transform in transforms:
            for method in methods:
                count += 1
                print(f"\n[{count}/{total}] Progress: {count/total*100:.1f}%")

                result = run_single_experiment(
                    task, transform, method,
                    num_train=num_train,
                    num_test=num_test,
                    poison_rate=poison_rate
                )

                if result:
                    all_results.append(result)

                    # Save intermediate results
                    with open(results_dir / f'results_{timestamp}.json', 'w') as f:
                        json.dump(all_results, f, indent=2)

    # Save final results
    final_file = results_dir / f'final_results_{timestamp}.json'
    with open(final_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"All experiments complete! Results saved to: {final_file}")
    print(f"{'='*80}")

    # Generate summary
    generate_summary(all_results, results_dir / f'summary_{timestamp}.md')

    return all_results


def generate_summary(results, output_file):
    """Generate a markdown summary of results."""

    with open(output_file, 'w') as f:
        f.write("# Advanced Semantic Transformation Detection Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Group by task
        for task in ['polarity', 'sentiment']:
            task_results = [r for r in results if r['task'] == task]
            if not task_results:
                continue

            f.write(f"## Task: {task.upper()}\n\n")

            # Group by transform
            transforms = sorted(set(r['transform'] for r in task_results))

            for transform in transforms:
                transform_results = [r for r in task_results if r['transform'] == transform]

                f.write(f"### Transform: {transform}\n\n")
                f.write("| Method | Precision | Recall | F1 | Detected |\n")
                f.write("|--------|-----------|--------|----|-----------|\n")

                # Sort by F1 score
                transform_results.sort(key=lambda x: x['f1'], reverse=True)

                for r in transform_results:
                    f.write(f"| {r['method']} | {r['precision']:.3f} | "
                           f"{r['recall']:.3f} | {r['f1']:.3f} | "
                           f"{r['num_detected']}/{r['num_poisoned']} |\n")

                f.write("\n")

            # Best methods for this task
            f.write(f"### Best Methods for {task}\n\n")
            task_results.sort(key=lambda x: x['f1'], reverse=True)

            f.write("| Rank | Transform | Method | F1 |\n")
            f.write("|------|-----------|--------|----|\n")

            for i, r in enumerate(task_results[:10], 1):
                f.write(f"| {i} | {r['transform']} | {r['method']} | {r['f1']:.3f} |\n")

            f.write("\n")

    print(f"Summary written to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run quick test with fewer configs')
    args = parser.parse_args()

    if args.quick:
        print("Running QUICK test mode...")
        # Quick test: just one config
        result = run_single_experiment(
            task='polarity',
            transform_name='strong_lexicon_flip',
            detector_method='iqr_k2.0',
            num_train=100,
            num_test=50,
            poison_rate=0.1
        )
        print(f"\nQuick test result: {json.dumps(result, indent=2)}")
    else:
        print("Running FULL experiments...")
        run_all_experiments()
