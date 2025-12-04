#!/usr/bin/env python3
"""
Generate visualizations for transform diversity experiments.
Creates plots showing:
1. Ensemble detector performance comparison
2. Cross-validation results (generalization to unseen transforms)
3. Transform distribution in dataset
"""
import json
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all experimental results."""
    results = {}

    # Load ensemble results
    ensemble_path = results_dir / "ensemble_diverse_transforms.json"
    if ensemble_path.exists():
        with open(ensemble_path) as f:
            results['ensemble'] = json.load(f)
            logger.info(f"Loaded ensemble results from {ensemble_path}")

    # Load cross-validation results
    cv_path = results_dir / "cross_validation.json"
    if cv_path.exists():
        with open(cv_path) as f:
            results['cross_validation'] = json.load(f)
            logger.info(f"Loaded cross-validation results from {cv_path}")

    return results


def plot_ensemble_performance(ensemble_results: Dict[str, Any], output_path: Path):
    """Create bar chart comparing ensemble methods."""
    methods = ensemble_results['results']

    # Extract metrics
    method_names = []
    recalls = []
    precisions = []
    f1_scores = []

    for method_name, metrics in methods.items():
        method_names.append(method_name.replace('_', ' ').title())
        recalls.append(metrics['recall'] * 100)
        precisions.append(metrics['precision'] * 100)
        f1_scores.append(metrics['f1_score'] * 100)

    # Create subplot with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Ensemble Detector Performance Comparison', fontsize=16, fontweight='bold')

    x = np.arange(len(method_names))
    width = 0.6

    # Plot 1: Recall
    axes[0].bar(x, recalls, width, color='#e74c3c', alpha=0.8)
    axes[0].set_ylabel('Recall (%)', fontsize=11)
    axes[0].set_title('Poisoned Sample Detection Rate', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(method_names, rotation=15, ha='right')
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Precision
    axes[1].bar(x, precisions, width, color='#3498db', alpha=0.8)
    axes[1].set_ylabel('Precision (%)', fontsize=11)
    axes[1].set_title('Clean Sample Precision', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(method_names, rotation=15, ha='right')
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(precisions):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Plot 3: F1 Score
    axes[2].bar(x, f1_scores, width, color='#2ecc71', alpha=0.8)
    axes[2].set_ylabel('F1 Score (%)', fontsize=11)
    axes[2].set_title('Overall F1 Score', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(method_names, rotation=15, ha='right')
    axes[2].set_ylim([0, 105])
    axes[2].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved ensemble performance plot to {output_path}")
    plt.close()


def plot_cross_validation(cv_results: Dict[str, Any], output_path: Path):
    """Create visualization of leave-category-out cross-validation results."""
    cv_folds = cv_results['leave_category_out']['individual_results']

    # Extract data
    fold_names = []
    recalls = []
    precisions = []
    f1_scores = []

    for fold_data in cv_folds:
        category = fold_data['held_out_category'].title()
        fold_names.append(category)
        avg_metrics = fold_data['avg_metrics']
        recalls.append(avg_metrics['recall'] * 100)
        precisions.append(avg_metrics['precision'] * 100)
        f1_scores.append(avg_metrics['f1'] * 100)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Leave-Category-Out Cross-Validation: Generalization to Unseen Transform Categories',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(fold_names))
    width = 0.6

    # Plot 1: Recall
    axes[0].bar(x, recalls, width, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[0].set_ylabel('Recall (%)', fontsize=11)
    axes[0].set_title('Poisoned Sample Detection', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(fold_names, fontsize=11)
    axes[0].set_ylim([0, 105])
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=np.mean(recalls), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(recalls):.1f}%')
    axes[0].legend(loc='lower right')

    # Add value labels
    for i, v in enumerate(recalls):
        axes[0].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 2: Precision
    axes[1].bar(x, precisions, width, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[1].set_ylabel('Precision (%)', fontsize=11)
    axes[1].set_title('Clean Sample Precision', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(fold_names, fontsize=11)
    axes[1].set_ylim([0, 105])
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=np.mean(precisions), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(precisions):.1f}%')
    axes[1].legend(loc='lower right')

    # Add value labels
    for i, v in enumerate(precisions):
        axes[1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Plot 3: F1 Score
    axes[2].bar(x, f1_scores, width, color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.2)
    axes[2].set_ylabel('F1 Score (%)', fontsize=11)
    axes[2].set_title('Overall F1 Score', fontsize=12, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(fold_names, fontsize=11)
    axes[2].set_ylim([0, 105])
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].axhline(y=np.mean(f1_scores), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(f1_scores):.1f}%')
    axes[2].legend(loc='lower right')

    # Add value labels
    for i, v in enumerate(f1_scores):
        axes[2].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-validation plot to {output_path}")
    plt.close()


def plot_generalization_gap(cv_results: Dict[str, Any], output_path: Path):
    """Plot generalization gap (train vs test performance)."""
    cv_folds = cv_results['cross_validation_folds']

    # Calculate gaps
    fold_names = []
    recall_gaps = []
    f1_gaps = []

    for fold_name, fold_data in cv_folds.items():
        fold_names.append(fold_name.replace('_', ' ').title())

        train_recall = fold_data['train_metrics']['poisoned_recall']
        test_recall = fold_data['test_metrics']['poisoned_recall']
        recall_gap = (train_recall - test_recall) * 100
        recall_gaps.append(recall_gap)

        train_f1 = fold_data['train_metrics']['f1_score']
        test_f1 = fold_data['test_metrics']['f1_score']
        f1_gap = (train_f1 - test_f1) * 100
        f1_gaps.append(f1_gap)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(fold_names))
    width = 0.35

    ax.bar(x - width/2, recall_gaps, width, label='Recall Gap', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, f1_gaps, width, label='F1 Gap', color='#9b59b6', alpha=0.8)

    ax.set_ylabel('Performance Gap (%)', fontsize=11)
    ax.set_title('Generalization Gap: Train vs Test Performance\n(Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(fold_names, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # Add value labels
    for i, (recall_v, f1_v) in enumerate(zip(recall_gaps, f1_gaps)):
        ax.text(i - width/2, recall_v + 1, f'{recall_v:.1f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, f1_v + 1, f'{f1_v:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved generalization gap plot to {output_path}")
    plt.close()


def plot_transform_distribution(ensemble_results: Dict[str, Any], output_path: Path):
    """Create bar chart showing transform distribution."""
    transforms_used = ensemble_results.get('transforms_used', [])

    if not transforms_used:
        logger.warning("No transforms_used found in results, skipping transform distribution plot")
        return

    # Extract category and transform name
    categories = []
    transform_names = []
    for t in transforms_used:
        parts = t.split('_', 1)
        if len(parts) == 2:
            categories.append(parts[0].title())
            transform_names.append(parts[1].replace('_', ' ').title())
        else:
            categories.append('Other')
            transform_names.append(t.replace('_', ' ').title())

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle('Transforms Used in Training Dataset', fontsize=16, fontweight='bold')

    # Color map by category
    category_colors = {
        'Lexicon': '#e74c3c',
        'Semantic': '#2ecc71',
        'Structural': '#3498db',
        'Character': '#f39c12',
        'Other': '#95a5a6'
    }

    colors = [category_colors.get(c, '#95a5a6') for c in categories]

    # Create bar plot
    x = np.arange(len(transform_names))
    bars = ax.bar(x, [1] * len(transform_names), color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{len(transforms_used)} Diverse Transforms Across {len(set(categories))} Categories', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(transform_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1.5])
    ax.set_yticks([1])
    ax.grid(axis='y', alpha=0.3)

    # Add category legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat)
                      for cat, color in category_colors.items() if cat in categories]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved transform distribution plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for transform diversity experiments"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="experiments/results",
        help="Directory containing experimental results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/plots",
        help="Directory to save plots"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_dir = Path(args.results_dir)
    results = load_results(results_dir)

    if not results:
        logger.error("No results found to visualize!")
        return

    # Generate plots
    logger.info("Generating visualizations...")

    if 'ensemble' in results:
        # Ensemble performance
        plot_ensemble_performance(
            results['ensemble'],
            output_dir / "ensemble_performance.png"
        )

        # Transform distribution
        plot_transform_distribution(
            results['ensemble'],
            output_dir / "transform_distribution.png"
        )

    if 'cross_validation' in results:
        # Cross-validation results
        plot_cross_validation(
            results['cross_validation'],
            output_dir / "cv_generalization.png"
        )

        # Generalization gap
        plot_generalization_gap(
            results['cross_validation'],
            output_dir / "cv_gap.png"
        )

    logger.info(f"\n{'='*60}")
    logger.info("Visualization Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Plots saved to: {output_dir}")
    logger.info("\nGenerated plots:")
    for plot_file in sorted(output_dir.glob("*.png")):
        logger.info(f"  - {plot_file.name}")


if __name__ == "__main__":
    main()
