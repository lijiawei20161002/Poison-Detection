"""
Re-evaluate existing transformation results with improved detection methods.

This script loads previously computed influence scores and applies
the improved detection strategies.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.detection.improved_detector import ImprovedTransformDetector
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_influence_scores(scores_dir: Path, analysis_name: str) -> np.ndarray:
    """Load influence scores from Kronfluence output."""
    scores_file = scores_dir / analysis_name / f"scores_{analysis_name}" / "pairwise_scores.safetensors"

    if not scores_file.exists():
        # Try alternative path
        scores_file = scores_dir / analysis_name / "pairwise_scores.safetensors"

    if not scores_file.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_file}")

    # Load safetensors
    from safetensors import safe_open

    with safe_open(scores_file, framework="numpy") as f:
        # Kronfluence stores scores as "all_modules"
        scores = f.get_tensor("all_modules")

    return scores


def reevaluate_transform(
    results_dir: Path,
    transform_name: str,
    poisoned_indices: set
) -> dict:
    """Re-evaluate a single transformation with improved methods."""

    logger.info(f"\nRe-evaluating: {transform_name}")

    try:
        # Load original scores
        original_scores = load_influence_scores(results_dir, "original")

        # Load transformed scores
        transformed_scores = load_influence_scores(
            results_dir,
            f"transformed_{transform_name}"
        )

        logger.info(f"Loaded scores: {original_scores.shape}")

        # Create improved detector
        detector = ImprovedTransformDetector(
            original_scores,
            transformed_scores,
            poisoned_indices
        )

        # Run all improved methods
        logger.info("Running improved detection methods...")
        all_results = detector.detect_all_methods()

        # Find best method
        best_result = max(all_results.values(), key=lambda r: r.f1_score)

        logger.info(f"Best method: {best_result.method_name}")
        logger.info(f"  F1: {best_result.f1_score:.4f}")
        logger.info(f"  Precision: {best_result.precision:.4f}")
        logger.info(f"  Recall: {best_result.recall:.4f}")

        # Convert results to dict
        results_dict = {}
        for name, result in all_results.items():
            results_dict[name] = {
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall,
                'true_positives': result.true_positives,
                'false_positives': result.false_positives,
                'true_negatives': result.true_negatives,
                'false_negatives': result.false_negatives,
                'num_detected': len(result.detected_indices),
                'threshold': result.threshold
            }

        return {
            'transform': transform_name,
            'status': 'success',
            'best_method': best_result.method_name,
            'best_f1': best_result.f1_score,
            'best_precision': best_result.precision,
            'best_recall': best_result.recall,
            'all_methods': results_dict
        }

    except Exception as e:
        logger.error(f"Failed to re-evaluate {transform_name}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'transform': transform_name,
            'status': 'error',
            'error': str(e)
        }


def create_comparison_report(
    original_results: list,
    improved_results: list,
    output_dir: Path
):
    """Create comparison between original and improved methods."""

    # Create DataFrames
    orig_df = pd.DataFrame([
        {
            'transform': r['transform'],
            'method': 'original',
            'f1_score': r.get('f1_score', 0.0),
            'precision': r.get('precision', 0.0),
            'recall': r.get('recall', 0.0)
        }
        for r in original_results if r.get('status') == 'success'
    ])

    improved_df = pd.DataFrame([
        {
            'transform': r['transform'],
            'method': 'improved',
            'f1_score': r.get('best_f1', 0.0),
            'precision': r.get('best_precision', 0.0),
            'recall': r.get('best_recall', 0.0),
            'best_method': r.get('best_method', 'unknown')
        }
        for r in improved_results if r.get('status') == 'success'
    ])

    # Merge for comparison
    comparison = pd.merge(
        orig_df,
        improved_df,
        on='transform',
        suffixes=('_original', '_improved')
    )

    comparison['f1_improvement'] = comparison['f1_score_improved'] - comparison['f1_score_original']
    comparison['improvement_pct'] = (comparison['f1_improvement'] / (comparison['f1_score_original'] + 1e-6)) * 100

    # Print summary
    print("\n" + "=" * 80)
    print("IMPROVED METHODS COMPARISON")
    print("=" * 80)
    print("\nF1 Score Comparison:")
    print(comparison[['transform', 'f1_score_original', 'f1_score_improved', 'f1_improvement', 'improvement_pct']].to_string(index=False))

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Mean F1 improvement: {comparison['f1_improvement'].mean():.4f}")
    print(f"Transforms with improvement: {(comparison['f1_improvement'] > 0).sum()} / {len(comparison)}")
    print(f"Max improvement: {comparison['f1_improvement'].max():.4f} ({comparison.loc[comparison['f1_improvement'].idxmax(), 'transform']})")

    # Save detailed results
    output_file = output_dir / "improved_methods_comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            'comparison': comparison.to_dict('records'),
            'improved_details': improved_results,
            'summary': {
                'mean_improvement': float(comparison['f1_improvement'].mean()),
                'transforms_improved': int((comparison['f1_improvement'] > 0).sum()),
                'total_transforms': len(comparison)
            }
        }, f, indent=2)

    logger.info(f"Detailed results saved to {output_file}")

    # Create visualization
    create_comparison_plot(comparison, output_dir)


def create_comparison_plot(comparison_df: pd.DataFrame, output_dir: Path):
    """Create visualization comparing original vs improved methods."""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: F1 Score comparison
    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.35

    ax.bar(x - width/2, comparison_df['f1_score_original'],
           width, label='Original Method', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, comparison_df['f1_score_improved'],
           width, label='Improved Methods', color='forestgreen', alpha=0.8)

    ax.set_xlabel('Transformation', fontsize=12)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Original vs Improved Detection Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['transform'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Improvement percentage
    ax = axes[1]
    colors = ['green' if x > 0 else 'red' for x in comparison_df['f1_improvement']]
    ax.barh(comparison_df['transform'], comparison_df['improvement_pct'],
            color=colors, alpha=0.7)
    ax.set_xlabel('Improvement (%)', fontsize=12)
    ax.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'improved_methods_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main execution."""

    parser = argparse.ArgumentParser(
        description='Re-evaluate transformations with improved methods'
    )
    parser.add_argument('--results_dir', type=str,
                       default='experiments/results/transform_comparison/polarity')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--task', type=str, default='polarity')
    parser.add_argument('--transforms', nargs='+', default=None,
                       help='Specific transforms to re-evaluate')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load poisoned indices
    poisoned_indices_path = Path(args.data_dir) / args.task / "poisoned_indices.txt"
    poisoned_indices = set()

    if poisoned_indices_path.exists():
        with open(poisoned_indices_path, 'r') as f:
            all_poisoned = {int(line.strip()) for line in f if line.strip()}
        # Filter to first 1000 (matching the experiments)
        poisoned_indices = {idx for idx in all_poisoned if idx < 1000}
        logger.info(f"Loaded {len(poisoned_indices)} poisoned indices")
    else:
        logger.error(f"Poisoned indices file not found: {poisoned_indices_path}")
        return

    # Find transforms to re-evaluate
    if args.transforms:
        transforms = args.transforms
    else:
        # Auto-detect from existing results
        result_files = list(results_dir.glob("comparison_results_*_test.json"))
        transforms = [
            f.stem.replace("comparison_results_", "").replace("_test", "")
            for f in result_files
        ]

    logger.info(f"Re-evaluating {len(transforms)} transformations with improved methods")

    # Load original results
    original_results = []
    for transform in transforms:
        result_file = results_dir / f"comparison_results_{transform}_test.json"
        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)
                original_results.append({
                    'transform': transform,
                    'status': 'success',
                    'f1_score': data['transform']['f1_score'],
                    'precision': data['transform']['precision'],
                    'recall': data['transform']['recall']
                })

    # Re-evaluate with improved methods
    improved_results = []
    for transform in transforms:
        result = reevaluate_transform(results_dir, transform, poisoned_indices)
        improved_results.append(result)

    # Create comparison report
    create_comparison_report(original_results, improved_results, results_dir)

    logger.info("\n" + "=" * 80)
    logger.info("RE-EVALUATION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
