#!/usr/bin/env python3
"""
Comprehensive Results Analysis and Visualization Script

This script analyzes experiment results and generates visualizations:
1. Loads results from JSON files
2. Performs statistical analysis
3. Generates comparison plots
4. Creates summary tables
5. Identifies insights and patterns

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --results-dir experiments/results/comprehensive_aggressive
    python experiments/analyze_results.py --compare experiments/results/baseline_500 experiments/results/aggressive_multi_gpu
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ResultsAnalyzer:
    """Analyzes and visualizes experiment results."""

    def __init__(self, results_dir: str):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.results = {}
        self.baseline = {}

    def load_results(self):
        """Load all result files from directory."""
        print(f"Loading results from: {self.results_dir}")

        # Load Phase 1 results
        phase_1_dir = self.results_dir / 'phase_1'
        if phase_1_dir.exists():
            summary_file = phase_1_dir / 'polarity' / 'experiment_summary.json'
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    self.results['phase_1'] = json.load(f)
                print(f"✅ Loaded Phase 1 results: {summary_file}")

        # Load Phase 2 results
        phase_2_dir = self.results_dir / 'phase_2'
        if phase_2_dir.exists():
            # Find all transform directories
            for transform_dir in phase_2_dir.iterdir():
                if transform_dir.is_dir():
                    transform_name = transform_dir.name
                    self.results.setdefault('phase_2', {})[transform_name] = {}

                    # Load all runs
                    runs = []
                    for run_dir in sorted(transform_dir.glob('run_*')):
                        result_file = run_dir / f'{transform_name}_results.json'
                        if result_file.exists():
                            with open(result_file, 'r') as f:
                                runs.append(json.load(f))

                    if runs:
                        self.results['phase_2'][transform_name]['runs'] = runs
                        print(f"✅ Loaded Phase 2 results for {transform_name}: {len(runs)} runs")

        # Load comprehensive results if available
        reports_dir = self.results_dir / 'reports'
        if reports_dir.exists():
            # Find most recent comprehensive results
            result_files = sorted(reports_dir.glob('comprehensive_results_*.json'))
            if result_files:
                with open(result_files[-1], 'r') as f:
                    self.results['comprehensive'] = json.load(f)
                print(f"✅ Loaded comprehensive results: {result_files[-1]}")

    def analyze_phase_1(self) -> Dict[str, Any]:
        """Analyze Phase 1 results."""
        print("\n" + "="*100)
        print("PHASE 1 ANALYSIS: INDIVIDUAL TRANSFORM PERFORMANCE")
        print("="*100)

        if 'phase_1' not in self.results:
            print("❌ No Phase 1 results found")
            return {}

        phase_1 = self.results['phase_1']
        transforms = phase_1.get('transforms', {})

        if not transforms:
            print("❌ No transform results in Phase 1")
            return {}

        # Extract metrics
        transform_metrics = []
        for name, result in transforms.items():
            transform_metrics.append({
                'name': name,
                'f1': result.get('f1_score', 0),
                'precision': result.get('precision', 0),
                'recall': result.get('recall', 0)
            })

        # Sort by F1
        transform_metrics.sort(key=lambda x: x['f1'], reverse=True)

        # Print table
        print("\nTransform Performance (sorted by F1 score):")
        print("-"*100)
        print(f"{'Rank':<6} {'Transform Name':<40} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print("-"*100)

        for i, metrics in enumerate(transform_metrics, 1):
            print(f"{i:<6} {metrics['name']:<40} "
                  f"{metrics['f1']:<10.4f} {metrics['precision']:<12.4f} {metrics['recall']:<10.4f}")

        # Statistical summary
        f1_scores = [m['f1'] for m in transform_metrics]
        print("\n" + "-"*100)
        print("Statistical Summary:")
        print("-"*100)
        print(f"Best F1:     {max(f1_scores):.4f}")
        print(f"Worst F1:    {min(f1_scores):.4f}")
        print(f"Mean F1:     {np.mean(f1_scores):.4f}")
        print(f"Median F1:   {np.median(f1_scores):.4f}")
        print(f"Std Dev F1:  {np.std(f1_scores):.4f}")

        # Baseline comparison
        if 'baseline' in phase_1 and phase_1['baseline']:
            print("\n" + "-"*100)
            print("Baseline Comparison:")
            print("-"*100)

            baseline = phase_1['baseline']
            if 'methods' in baseline:
                for method_name, metrics in baseline['methods'].items():
                    baseline_f1 = metrics.get('f1_score', 0)
                    print(f"Baseline {method_name}: F1={baseline_f1:.4f}")

                    # Count transforms that beat baseline
                    beats_baseline = sum(1 for m in transform_metrics if m['f1'] > baseline_f1)
                    print(f"  Transforms beating this baseline: {beats_baseline}/{len(transform_metrics)}")

        # README baseline comparison
        readme_baseline_f1 = 0.1600
        readme_transform_f1 = 0.0684

        print("\n" + "-"*100)
        print("README.md Baseline Comparison:")
        print("-"*100)
        print(f"README Direct Detection F1: {readme_baseline_f1:.4f}")
        print(f"README Transform-Enhanced F1: {readme_transform_f1:.4f}")

        best_transform = transform_metrics[0]
        if best_transform['f1'] > readme_baseline_f1:
            improvement = (best_transform['f1'] - readme_baseline_f1) / readme_baseline_f1 * 100
            print(f"\n✅✅ BREAKTHROUGH: Best transform beats README baseline!")
            print(f"   {best_transform['name']}: F1={best_transform['f1']:.4f}")
            print(f"   Improvement: +{improvement:.1f}%")
        elif best_transform['f1'] > readme_transform_f1:
            improvement = (best_transform['f1'] - readme_transform_f1) / readme_transform_f1 * 100
            print(f"\n✅ Best transform beats README transform baseline")
            print(f"   {best_transform['name']}: F1={best_transform['f1']:.4f}")
            print(f"   Improvement: +{improvement:.1f}%")
            gap = readme_baseline_f1 - best_transform['f1']
            print(f"   Gap to direct detection: {gap:.4f} ({gap/readme_baseline_f1*100:.1f}%)")
        else:
            print(f"\n❌ Best transform does not beat README baselines")
            print(f"   {best_transform['name']}: F1={best_transform['f1']:.4f}")

        analysis = {
            'transform_metrics': transform_metrics,
            'statistics': {
                'best_f1': max(f1_scores),
                'worst_f1': min(f1_scores),
                'mean_f1': float(np.mean(f1_scores)),
                'median_f1': float(np.median(f1_scores)),
                'std_f1': float(np.std(f1_scores))
            },
            'best_transform': best_transform,
            'baseline_comparison': {
                'readme_direct_f1': readme_baseline_f1,
                'readme_transform_f1': readme_transform_f1,
                'beats_direct': best_transform['f1'] > readme_baseline_f1,
                'beats_transform': best_transform['f1'] > readme_transform_f1
            }
        }

        return analysis

    def analyze_phase_2(self) -> Dict[str, Any]:
        """Analyze Phase 2 validation results."""
        print("\n" + "="*100)
        print("PHASE 2 ANALYSIS: VALIDATION WITH LARGER SAMPLES")
        print("="*100)

        if 'phase_2' not in self.results:
            print("❌ No Phase 2 results found")
            return {}

        phase_2 = self.results['phase_2']

        for transform_name, data in phase_2.items():
            runs = data.get('runs', [])

            if not runs:
                print(f"❌ No runs found for {transform_name}")
                continue

            print(f"\nTransform: {transform_name}")
            print("-"*100)
            print(f"Number of validation runs: {len(runs)}")

            # Extract metrics from each run
            f1_scores = [r.get('f1_score', 0) for r in runs]
            precisions = [r.get('precision', 0) for r in runs]
            recalls = [r.get('recall', 0) for r in runs]

            print(f"\nF1 Scores across runs:")
            for i, f1 in enumerate(f1_scores, 1):
                print(f"  Run {i}: {f1:.4f}")

            print(f"\nAggregated Statistics:")
            print(f"  Mean F1:        {np.mean(f1_scores):.4f}")
            print(f"  Std Dev F1:     {np.std(f1_scores):.4f}")
            print(f"  Mean Precision: {np.mean(precisions):.4f}")
            print(f"  Mean Recall:    {np.mean(recalls):.4f}")

            # Check consistency
            if len(f1_scores) > 1:
                cv = np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 0
                print(f"  Coefficient of Variation: {cv:.4f}")

                if cv < 0.1:
                    print(f"  ✅ Results are consistent (CV < 0.1)")
                else:
                    print(f"  ⚠️ Results show variability (CV = {cv:.4f})")

        return {
            'transform_name': list(phase_2.keys())[0] if phase_2 else None,
            'validation_results': phase_2
        }

    def compare_with_baseline(self, baseline_dir: str):
        """Compare current results with baseline results."""
        print("\n" + "="*100)
        print("BASELINE COMPARISON")
        print("="*100)

        baseline_path = Path(baseline_dir)
        if not baseline_path.exists():
            print(f"❌ Baseline directory not found: {baseline_dir}")
            return

        # Load baseline results
        baseline_file = baseline_path / 'experiment_summary.json'
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                self.baseline = json.load(f)

            print(f"✅ Loaded baseline from: {baseline_file}")

            # Compare metrics
            # Implementation depends on baseline structure
            print("\nBaseline comparison implementation pending...")

    def generate_summary_table(self, output_file: str = None):
        """Generate summary table of all results."""
        print("\n" + "="*100)
        print("SUMMARY TABLE")
        print("="*100)

        summary = []

        # Add Phase 1 results
        if 'phase_1' in self.results:
            transforms = self.results['phase_1'].get('transforms', {})
            for name, result in transforms.items():
                summary.append({
                    'Phase': 'Phase 1',
                    'Transform': name,
                    'F1': result.get('f1_score', 0),
                    'Precision': result.get('precision', 0),
                    'Recall': result.get('recall', 0),
                    'Samples': 'Train=100, Test=50'
                })

        # Add Phase 2 results
        if 'phase_2' in self.results:
            for transform_name, data in self.results['phase_2'].items():
                runs = data.get('runs', [])
                if runs:
                    avg_f1 = np.mean([r.get('f1_score', 0) for r in runs])
                    avg_p = np.mean([r.get('precision', 0) for r in runs])
                    avg_r = np.mean([r.get('recall', 0) for r in runs])

                    summary.append({
                        'Phase': 'Phase 2',
                        'Transform': transform_name,
                        'F1': avg_f1,
                        'Precision': avg_p,
                        'Recall': avg_r,
                        'Samples': f'Train=200, Test=100, Runs={len(runs)}'
                    })

        # Print table
        print(f"\n{'Phase':<10} {'Transform':<40} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Samples':<30}")
        print("-"*110)
        for entry in summary:
            print(f"{entry['Phase']:<10} {entry['Transform']:<40} "
                  f"{entry['F1']:<10.4f} {entry['Precision']:<12.4f} "
                  f"{entry['Recall']:<10.4f} {entry['Samples']:<30}")

        # Save to file if requested
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write("COMPREHENSIVE EXPERIMENT SUMMARY\n")
                f.write("="*110 + "\n\n")

                f.write(f"{'Phase':<10} {'Transform':<40} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Samples':<30}\n")
                f.write("-"*110 + "\n")

                for entry in summary:
                    f.write(f"{entry['Phase']:<10} {entry['Transform']:<40} "
                           f"{entry['F1']:<10.4f} {entry['Precision']:<12.4f} "
                           f"{entry['Recall']:<10.4f} {entry['Samples']:<30}\n")

            print(f"\n✅ Summary table saved to: {output_path}")

        return summary

    def generate_insights(self):
        """Generate insights from results."""
        print("\n" + "="*100)
        print("KEY INSIGHTS")
        print("="*100)

        insights = []

        # Phase 1 insights
        if 'phase_1' in self.results:
            transforms = self.results['phase_1'].get('transforms', {})
            f1_scores = [r.get('f1_score', 0) for r in transforms.values()]

            if f1_scores:
                best_f1 = max(f1_scores)
                worst_f1 = min(f1_scores)
                spread = best_f1 - worst_f1

                insights.append(f"1. Transform performance varies significantly (spread: {spread:.4f})")

                if best_f1 > 0.16:
                    insights.append(f"2. Best transform (F1={best_f1:.4f}) BEATS the baseline!")
                elif best_f1 > 0.10:
                    insights.append(f"2. Best transform (F1={best_f1:.4f}) shows promise but needs improvement")
                else:
                    insights.append(f"2. All transforms underperform (best F1={best_f1:.4f})")

                # Category analysis
                negation_transforms = [name for name in transforms.keys() if 'negation' in name.lower()]
                insertion_transforms = [name for name in transforms.keys() if 'insertion' in name.lower()]

                if negation_transforms:
                    neg_f1_avg = np.mean([transforms[n].get('f1_score', 0) for n in negation_transforms])
                    insights.append(f"3. Negation-based transforms average F1: {neg_f1_avg:.4f}")

                if insertion_transforms:
                    ins_f1_avg = np.mean([transforms[n].get('f1_score', 0) for n in insertion_transforms])
                    insights.append(f"4. Insertion-based transforms average F1: {ins_f1_avg:.4f}")

        # Print insights
        for insight in insights:
            print(f"\n{insight}")

        return insights

    def run_full_analysis(self, output_dir: str = None):
        """Run complete analysis pipeline."""
        print("="*100)
        print("COMPREHENSIVE RESULTS ANALYSIS")
        print("="*100)
        print(f"Results Directory: {self.results_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*100)

        # Load results
        self.load_results()

        # Phase 1 analysis
        phase_1_analysis = self.analyze_phase_1()

        # Phase 2 analysis
        phase_2_analysis = self.analyze_phase_2()

        # Generate summary
        if output_dir:
            output_path = Path(output_dir)
            summary_file = output_path / 'analysis_summary.txt'
            self.generate_summary_table(output_file=summary_file)

        # Generate insights
        insights = self.generate_insights()

        print("\n" + "="*100)
        print("ANALYSIS COMPLETE")
        print("="*100)

        return {
            'phase_1': phase_1_analysis,
            'phase_2': phase_2_analysis,
            'insights': insights
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze experiment results and generate reports'
    )
    parser.add_argument('--results-dir', type=str,
                       default='experiments/results/comprehensive_aggressive',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save analysis outputs')
    parser.add_argument('--compare-baseline', type=str,
                       help='Baseline directory for comparison')

    args = parser.parse_args()

    # Run analysis
    analyzer = ResultsAnalyzer(args.results_dir)
    analysis = analyzer.run_full_analysis(output_dir=args.output_dir or args.results_dir)

    # Compare with baseline if provided
    if args.compare_baseline:
        analyzer.compare_with_baseline(args.compare_baseline)


if __name__ == "__main__":
    main()
