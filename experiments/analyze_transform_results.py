#!/usr/bin/env python3
"""
Analyze transformation test results and generate detailed reports.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List
import numpy as np


def load_results(result_file: Path) -> Dict:
    """Load results from JSON file."""
    with open(result_file, 'r') as f:
        return json.load(f)


def analyze_results(data: Dict) -> None:
    """Perform comprehensive analysis of transformation results."""

    metadata = data.get('metadata', {})
    results = data.get('results', [])

    print("\n" + "=" * 100)
    print("COMPREHENSIVE TRANSFORMATION ANALYSIS")
    print("=" * 100)

    # Metadata
    print(f"\nTest Configuration:")
    print(f"  Task: {metadata.get('task', 'N/A')}")
    print(f"  Model: {metadata.get('model', 'N/A')}")
    print(f"  Training samples: {metadata.get('num_train', 'N/A')}")
    print(f"  Test samples: {metadata.get('num_test', 'N/A')}")
    print(f"  GPUs used: {metadata.get('num_gpus', 'N/A')}")
    print(f"  Total time: {metadata.get('total_time', 0):.1f}s ({metadata.get('total_time', 0)/60:.1f} min)")

    # Separate successful and failed
    successful = [r for r in results if r.get('status') == 'success']
    failed = [r for r in results if r.get('status') != 'success']

    print(f"\nResults Overview:")
    print(f"  Total tests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    if failed:
        print(f"\nFailed Tests:")
        for fail in failed:
            print(f"  - {fail.get('transform', 'unknown')} (GPU {fail.get('gpu_id', '?')})")
            print(f"    Error: {fail.get('error', 'Unknown')[:100]}")

    if not successful:
        print("\n⚠️  No successful results to analyze!")
        return

    # Sort by F1 score
    successful.sort(key=lambda x: x.get('best_f1', 0), reverse=True)

    # Performance tiers
    print("\n" + "=" * 100)
    print("TRANSFORMATION PERFORMANCE RANKING")
    print("=" * 100)

    f1_scores = [r['best_f1'] for r in successful]
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    f1_median = np.median(f1_scores)

    print(f"\nF1 Score Statistics:")
    print(f"  Mean: {f1_mean:.6f}")
    print(f"  Median: {f1_median:.6f}")
    print(f"  Std Dev: {f1_std:.6f}")
    print(f"  Min: {min(f1_scores):.6f}")
    print(f"  Max: {max(f1_scores):.6f}")

    # Classify into tiers
    tier_a = [r for r in successful if r['best_f1'] >= f1_mean + 0.5 * f1_std]
    tier_b = [r for r in successful if f1_mean <= r['best_f1'] < f1_mean + 0.5 * f1_std]
    tier_c = [r for r in successful if f1_mean - 0.5 * f1_std <= r['best_f1'] < f1_mean]
    tier_d = [r for r in successful if r['best_f1'] < f1_mean - 0.5 * f1_std]

    print(f"\nPerformance Tiers:")
    print(f"  Tier A (Excellent): {len(tier_a)} transforms (F1 >= {f1_mean + 0.5 * f1_std:.6f})")
    print(f"  Tier B (Good): {len(tier_b)} transforms ({f1_mean:.6f} <= F1 < {f1_mean + 0.5 * f1_std:.6f})")
    print(f"  Tier C (Average): {len(tier_c)} transforms ({f1_mean - 0.5 * f1_std:.6f} <= F1 < {f1_mean:.6f})")
    print(f"  Tier D (Below Average): {len(tier_d)} transforms (F1 < {f1_mean - 0.5 * f1_std:.6f})")

    # Detailed rankings
    print(f"\n{'Rank':<6} {'Tier':<8} {'Transform':<35} {'Best Method':<30} {'F1':<10} {'Time':<8}")
    print("-" * 100)

    for i, result in enumerate(successful, 1):
        f1 = result['best_f1']
        if f1 >= f1_mean + 0.5 * f1_std:
            tier = "A"
        elif f1 >= f1_mean:
            tier = "B"
        elif f1 >= f1_mean - 0.5 * f1_std:
            tier = "C"
        else:
            tier = "D"

        print(f"{i:<6} {tier:<8} {result['transform']:<35} {result['best_method']:<30} "
              f"{f1:.6f}  {result['elapsed_time']:>6.1f}s")

    # Detection method analysis
    print("\n" + "=" * 100)
    print("DETECTION METHOD ANALYSIS")
    print("=" * 100)

    method_stats = {}
    for result in successful:
        for method_name, metrics in result.get('methods', {}).items():
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'f1_scores': [],
                    'precisions': [],
                    'recalls': [],
                    'times_best': 0
                }

            method_stats[method_name]['f1_scores'].append(metrics.get('f1_score', 0))
            method_stats[method_name]['precisions'].append(metrics.get('precision', 0))
            method_stats[method_name]['recalls'].append(metrics.get('recall', 0))

            if method_name == result['best_method']:
                method_stats[method_name]['times_best'] += 1

    # Calculate statistics
    method_analysis = []
    for method_name, stats in method_stats.items():
        method_analysis.append({
            'method': method_name,
            'avg_f1': np.mean(stats['f1_scores']),
            'std_f1': np.std(stats['f1_scores']),
            'avg_precision': np.mean(stats['precisions']),
            'avg_recall': np.mean(stats['recalls']),
            'times_best': stats['times_best'],
            'win_rate': stats['times_best'] / len(successful) * 100
        })

    method_analysis.sort(key=lambda x: x['avg_f1'], reverse=True)

    print(f"\n{'Rank':<6} {'Method':<40} {'Avg F1':<12} {'Std F1':<12} {'Win Rate':<12}")
    print("-" * 100)

    for i, ma in enumerate(method_analysis, 1):
        print(f"{i:<6} {ma['method']:<40} {ma['avg_f1']:>8.6f}    {ma['std_f1']:>8.6f}    "
              f"{ma['win_rate']:>5.1f}% ({ma['times_best']}/{len(successful)})")

    # Detailed method performance
    print(f"\n{'Rank':<6} {'Method':<40} {'Avg Prec':<12} {'Avg Recall':<12}")
    print("-" * 100)

    for i, ma in enumerate(method_analysis, 1):
        print(f"{i:<6} {ma['method']:<40} {ma['avg_precision']:>8.6f}    {ma['avg_recall']:>8.6f}")

    # Transformation deep dive
    print("\n" + "=" * 100)
    print("DETAILED ANALYSIS OF TOP 5 TRANSFORMATIONS")
    print("=" * 100)

    for i, result in enumerate(successful[:5], 1):
        print(f"\n{i}. {result['transform'].upper()}")
        print(f"   {'=' * 90}")
        print(f"   Overall Performance:")
        print(f"     - Best F1 Score: {result['best_f1']:.6f}")
        print(f"     - Best Method: {result['best_method']}")
        print(f"     - Computation Time: {result['elapsed_time']:.1f}s")
        print(f"     - GPU: {result['gpu_id']}")

        # Get all methods for this transform
        methods = result.get('methods', {})
        sorted_methods = sorted(methods.items(), key=lambda x: x[1].get('f1_score', 0), reverse=True)

        print(f"\n   Top 5 Detection Methods:")
        print(f"   {'-' * 90}")

        for j, (method_name, metrics) in enumerate(sorted_methods[:5], 1):
            f1 = metrics.get('f1_score', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            tp = metrics.get('true_positives', 0)
            fp = metrics.get('false_positives', 0)
            tn = metrics.get('true_negatives', 0)
            fn = metrics.get('false_negatives', 0)

            print(f"\n   {j}. {method_name}")
            print(f"      Metrics: F1={f1:.6f}, Precision={precision:.6f}, Recall={recall:.6f}")
            print(f"      Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    print(f"\n1. Best Transformations for Poison Detection:")
    print(f"   {'-' * 90}")
    for i, result in enumerate(tier_a if tier_a else successful[:3], 1):
        print(f"   {i}. {result['transform']}")
        print(f"      - F1 Score: {result['best_f1']:.6f}")
        print(f"      - Best Method: {result['best_method']}")
        print(f"      - Runtime: {result['elapsed_time']:.1f}s")
        print()

    print(f"2. Most Reliable Detection Methods:")
    print(f"   {'-' * 90}")
    for i, ma in enumerate(method_analysis[:3], 1):
        print(f"   {i}. {ma['method']}")
        print(f"      - Average F1: {ma['avg_f1']:.6f} (±{ma['std_f1']:.6f})")
        print(f"      - Win Rate: {ma['win_rate']:.1f}% (best in {ma['times_best']}/{len(successful)} transforms)")
        print(f"      - Avg Precision/Recall: {ma['avg_precision']:.6f}/{ma['avg_recall']:.6f}")
        print()

    print(f"3. Transformation-Method Pairings:")
    print(f"   {'-' * 90}")
    print(f"   For production deployment, use these combinations:")
    for i, result in enumerate(successful[:3], 1):
        print(f"   {i}. {result['transform']} + {result['best_method']}")
        print(f"      Expected F1: {result['best_f1']:.6f}")
        print()

    # Efficiency analysis
    print(f"4. Efficiency Analysis:")
    print(f"   {'-' * 90}")

    # Calculate efficiency score (F1 / time)
    for result in successful:
        result['efficiency'] = result['best_f1'] / max(result['elapsed_time'], 1)

    efficient = sorted(successful, key=lambda x: x['efficiency'], reverse=True)[:5]

    print(f"   Most efficient transformations (F1/time):")
    for i, result in enumerate(efficient, 1):
        print(f"   {i}. {result['transform']}")
        print(f"      F1: {result['best_f1']:.6f}, Time: {result['elapsed_time']:.1f}s, "
              f"Efficiency: {result['efficiency']:.6f}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Analyze transformation test results')
    parser.add_argument('result_file', type=str,
                       help='Path to results JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Save analysis to file')

    args = parser.parse_args()

    result_file = Path(args.result_file)

    if not result_file.exists():
        print(f"Error: Result file not found: {result_file}")
        return

    data = load_results(result_file)
    analyze_results(data)

    if args.output:
        # TODO: Save analysis to file
        print(f"\nAnalysis would be saved to: {args.output}")


if __name__ == '__main__':
    main()
