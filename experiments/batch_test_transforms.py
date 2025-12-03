"""
Batch test all transformation methods for poison detection.

This script systematically tests all available transformation methods
and compares their effectiveness.
"""

import argparse
import json
import sys
from pathlib import Path
import subprocess
import time
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_all_transforms(task_type: str = "sentiment") -> List[str]:
    """Get all available transformations for a task."""
    transforms = transform_registry.list_transforms(task_type)
    return transforms[task_type]


def run_single_transform_test(
    transform_name: str,
    task: str = "polarity",
    model: str = "data/polarity/outputs/final_model",
    num_train: int = 1000,
    num_test: int = 50,
    device: str = "cuda:0",
    output_dir: str = "experiments/results/transform_comparison"
) -> Dict:
    """Run a single transformation test."""

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Testing transformation: {transform_name}")
    logger.info(f"{'=' * 80}")

    # Build command
    cmd = [
        "python3",
        "experiments/compare_direct_vs_transform_detection.py",
        "--task", task,
        "--model", model,
        "--num_train_samples", str(num_train),
        "--num_test_samples", str(num_test),
        "--batch_size", "8",
        "--device", device,
        "--transform", transform_name,
        "--output_dir", output_dir,
        "--output_suffix", f"_{transform_name}_test"
    ]

    start_time = time.time()

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per transform
        )

        elapsed = time.time() - start_time

        if result.returncode != 0:
            logger.error(f"Transform {transform_name} failed with code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return {
                "transform": transform_name,
                "status": "failed",
                "error": result.stderr,
                "elapsed_time": elapsed
            }

        # Load results
        results_file = Path(output_dir) / task / f"comparison_results_{transform_name}_test.json"

        if not results_file.exists():
            logger.error(f"Results file not found: {results_file}")
            return {
                "transform": transform_name,
                "status": "no_results",
                "elapsed_time": elapsed
            }

        with open(results_file, 'r') as f:
            results = json.load(f)

        logger.info(f"✓ Transform {transform_name} completed in {elapsed:.2f}s")
        logger.info(f"  F1: {results['transform']['f1_score']:.4f}")
        logger.info(f"  Improvement: {results['improvement']:+.4f} ({results['improvement_pct']:+.1f}%)")

        return {
            "transform": transform_name,
            "status": "success",
            "f1_score": results['transform']['f1_score'],
            "precision": results['transform']['precision'],
            "recall": results['transform']['recall'],
            "improvement": results['improvement'],
            "improvement_pct": results['improvement_pct'],
            "best_direct_f1": results['best_direct']['f1_score'],
            "elapsed_time": elapsed
        }

    except subprocess.TimeoutExpired:
        logger.error(f"Transform {transform_name} timed out after 10 minutes")
        return {
            "transform": transform_name,
            "status": "timeout",
            "elapsed_time": 600
        }
    except Exception as e:
        logger.error(f"Transform {transform_name} failed with exception: {e}")
        return {
            "transform": transform_name,
            "status": "error",
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def create_summary_report(results: List[Dict], output_dir: Path):
    """Create comprehensive summary report and visualizations."""

    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']

    if not successful:
        logger.error("No successful results to analyze!")
        return

    # Create DataFrame
    df = pd.DataFrame(successful)
    df = df.sort_values('f1_score', ascending=False)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TRANSFORMATION COMPARISON")
    print("=" * 80)
    print("\nTop Performing Transformations:")
    print(df[['transform', 'f1_score', 'precision', 'recall', 'improvement_pct']].head(10).to_string(index=False))

    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80)
    print(f"Total transformations tested: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len([r for r in results if r['status'] == 'failed'])}")
    print(f"Timeout: {len([r for r in results if r['status'] == 'timeout'])}")
    print(f"\nBest transformation: {df.iloc[0]['transform']}")
    print(f"Best F1 score: {df.iloc[0]['f1_score']:.4f}")
    print(f"Best improvement: {df.iloc[0]['improvement_pct']:+.1f}%")
    print(f"\nMean F1 score: {df['f1_score'].mean():.4f}")
    print(f"Median F1 score: {df['f1_score'].median():.4f}")
    print(f"Std F1 score: {df['f1_score'].std():.4f}")

    # Identify transforms with positive improvement
    positive_improvement = df[df['improvement'] > 0]
    if len(positive_improvement) > 0:
        print(f"\nTransformations with positive improvement: {len(positive_improvement)}")
        print(positive_improvement[['transform', 'improvement_pct']].to_string(index=False))
    else:
        print("\n⚠️  NO transformations showed positive improvement over direct detection")

    # Save summary JSON
    summary_file = output_dir / "transform_comprehensive_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "summary_stats": {
                "total_tested": len(results),
                "successful": len(successful),
                "failed": len([r for r in results if r['status'] != 'success']),
                "best_transform": df.iloc[0]['transform'],
                "best_f1": float(df.iloc[0]['f1_score']),
                "mean_f1": float(df['f1_score'].mean()),
                "median_f1": float(df['f1_score'].median()),
                "positive_improvement_count": len(positive_improvement)
            },
            "all_results": results
        }, f, indent=2)

    logger.info(f"Summary saved to {summary_file}")

    # Create visualizations
    create_visualizations(df, output_dir)


def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Create comprehensive visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Transformation Method Comparison', fontsize=16, fontweight='bold')

    # Sort for better visualization
    df_sorted = df.sort_values('f1_score', ascending=True)

    # Plot 1: F1 Score comparison
    ax = axes[0, 0]
    colors = ['green' if imp > 0 else 'red' for imp in df_sorted['improvement']]
    ax.barh(df_sorted['transform'], df_sorted['f1_score'], color=colors, alpha=0.7)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score by Transformation Method', fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(df_sorted['f1_score'].max() * 1.1, 0.1))
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Improvement percentage
    ax = axes[0, 1]
    df_sorted_imp = df.sort_values('improvement_pct', ascending=True)
    colors = ['green' if imp > 0 else 'red' for imp in df_sorted_imp['improvement_pct']]
    ax.barh(df_sorted_imp['transform'], df_sorted_imp['improvement_pct'], color=colors, alpha=0.7)
    ax.set_xlabel('Improvement over Best Direct (%)', fontsize=12)
    ax.set_title('Performance Improvement', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    # Plot 3: Precision vs Recall
    ax = axes[1, 0]
    colors = ['green' if imp > 0 else 'red' for imp in df['improvement']]
    ax.scatter(df['recall'], df['precision'],
               c=colors, s=200, alpha=0.7, edgecolors='black')

    for _, row in df.iterrows():
        ax.annotate(row['transform'],
                   (row['recall'], row['precision']),
                   fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Plot 4: Runtime comparison
    ax = axes[1, 1]
    df_sorted_time = df.sort_values('elapsed_time', ascending=True)
    ax.barh(df_sorted_time['transform'], df_sorted_time['elapsed_time'],
            color='steelblue', alpha=0.7)
    ax.set_xlabel('Runtime (seconds)', fontsize=12)
    ax.set_title('Computational Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'comprehensive_transform_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    plt.close()


def main():
    """Main execution function."""

    parser = argparse.ArgumentParser(
        description='Batch test all transformation methods'
    )
    parser.add_argument('--task', type=str, default='polarity')
    parser.add_argument('--model', type=str, default='data/polarity/outputs/final_model')
    parser.add_argument('--num_train', type=int, default=1000)
    parser.add_argument('--num_test', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='experiments/results/transform_comparison')
    parser.add_argument('--skip_tested', action='store_true',
                       help='Skip already tested transformations')
    parser.add_argument('--transforms', nargs='+', default=None,
                       help='Specific transforms to test (default: all)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("BATCH TRANSFORMATION TESTING")
    logger.info("=" * 80)

    # Get all available transforms
    all_transforms = get_all_transforms("sentiment")

    # Filter transforms if specified
    if args.transforms:
        transforms_to_test = [t for t in args.transforms if t in all_transforms]
        logger.info(f"Testing specified transforms: {transforms_to_test}")
    else:
        transforms_to_test = all_transforms
        logger.info(f"Testing all {len(transforms_to_test)} transformations")

    # Check which have already been tested
    output_path = Path(args.output_dir) / args.task
    output_path.mkdir(parents=True, exist_ok=True)

    if args.skip_tested:
        existing_results = list(output_path.glob("comparison_results_*_test.json"))
        tested_transforms = {
            r.stem.replace("comparison_results_", "").replace("_test", "")
            for r in existing_results
        }
        transforms_to_test = [t for t in transforms_to_test if t not in tested_transforms]
        logger.info(f"Skipping {len(tested_transforms)} already tested transforms")
        logger.info(f"Remaining: {len(transforms_to_test)} transforms")

    # Run tests
    all_results = []

    for i, transform in enumerate(transforms_to_test, 1):
        logger.info(f"\n[{i}/{len(transforms_to_test)}] Testing: {transform}")

        result = run_single_transform_test(
            transform_name=transform,
            task=args.task,
            model=args.model,
            num_train=args.num_train,
            num_test=args.num_test,
            device=args.device,
            output_dir=args.output_dir
        )

        all_results.append(result)

        # Brief pause between tests
        time.sleep(2)

    # Create summary report
    logger.info("\n" + "=" * 80)
    logger.info("CREATING SUMMARY REPORT")
    logger.info("=" * 80)

    create_summary_report(all_results, output_path)

    logger.info("\n" + "=" * 80)
    logger.info("BATCH TESTING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
