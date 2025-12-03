"""
Test improved transformation-based detection methods.

This script loads existing influence scores and tests improved detection methods
without needing to recompute the expensive influence calculations.
"""

import numpy as np
import json
import logging
from pathlib import Path
from safetensors import safe_open
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.detection.improved_transform_detector import ImprovedTransformDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_influence_scores(scores_dir: Path) -> np.ndarray:
    """Load influence scores from safetensors file."""
    scores_file = scores_dir / "pairwise_scores.safetensors"

    with safe_open(scores_file, framework="numpy") as f:
        scores = f.get_tensor("all_modules")

    # Transpose if needed to get (n_train, n_test) shape
    if scores.shape[0] < scores.shape[1]:
        scores = scores.T

    logger.info(f"Loaded scores from {scores_file}, shape: {scores.shape}")
    return scores


def load_poisoned_indices(data_dir: Path, task: str) -> set:
    """Load ground truth poisoned indices."""
    # Try poisoned_indices.txt first
    indices_file = data_dir / task / "poisoned_indices.txt"

    if indices_file.exists():
        with open(indices_file, 'r') as f:
            poisoned_indices = set(int(line.strip()) for line in f if line.strip())
        logger.info(f"Loaded {len(poisoned_indices)} poisoned indices from {indices_file}")
        return poisoned_indices

    # Try metadata.json as fallback
    metadata_file = data_dir / task / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        poisoned_indices = set(metadata['poisoned_indices'])
        logger.info(f"Loaded {len(poisoned_indices)} poisoned indices from {metadata_file}")
        return poisoned_indices

    raise FileNotFoundError(f"Could not find poisoned indices in {indices_file} or {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Test improved transformation-based detection")
    parser.add_argument('--task', type=str, default='polarity', help='Task name')
    parser.add_argument('--transform', type=str, default='grammatical_negation',
                       choices=['grammatical_negation', 'strong_lexicon_flip', 'combined_flip_negation'],
                       help='Transformation type')
    parser.add_argument('--results_dir', type=str,
                       default='experiments/results/transform_comparison/polarity',
                       help='Results directory containing influence scores')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_suffix', type=str, default='improved', help='Output file suffix')

    args = parser.parse_args()

    # Setup paths
    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    # Determine correct subdirectory structure
    # The results are stored in results_dir/polarity/... or results_dir/...
    if (results_dir / args.task).exists():
        base_results_dir = results_dir / args.task
    else:
        base_results_dir = results_dir

    original_scores_dir = base_results_dir / "original" / "scores_original"
    transformed_scores_dir = base_results_dir / f"transformed_{args.transform}" / f"scores_transformed_{args.transform}"

    logger.info("="*80)
    logger.info("IMPROVED TRANSFORMATION-BASED DETECTION")
    logger.info("="*80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Transform: {args.transform}")
    logger.info(f"Original scores: {original_scores_dir}")
    logger.info(f"Transformed scores: {transformed_scores_dir}")
    logger.info("")

    # Check if directories exist
    if not original_scores_dir.exists():
        logger.error(f"Original scores directory not found: {original_scores_dir}")
        return

    if not transformed_scores_dir.exists():
        logger.error(f"Transformed scores directory not found: {transformed_scores_dir}")
        return

    # Load data
    logger.info("Loading influence scores...")
    original_scores = load_influence_scores(original_scores_dir)
    transformed_scores = load_influence_scores(transformed_scores_dir)

    logger.info("Loading poisoned indices...")
    poisoned_indices = load_poisoned_indices(data_dir, args.task)

    # Initialize detector
    detector = ImprovedTransformDetector(poisoned_indices)

    # Run all improved methods
    logger.info("")
    logger.info("="*80)
    logger.info("Running Improved Detection Methods")
    logger.info("="*80)

    results = detector.run_all_methods(original_scores, transformed_scores)

    # Print summary table
    logger.info("")
    logger.info("="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Method':<25} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Detected':>10}")
    logger.info("-"*80)

    # Sort by F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1][0]['f1_score'], reverse=True)

    for method_name, (metrics, _) in sorted_results:
        logger.info(f"{method_name:<25} {metrics['f1_score']:>8.4f} "
                   f"{metrics['precision']:>10.4f} {metrics['recall']:>8.4f} "
                   f"{metrics['num_detected']:>10}")

    # Find best method
    best_method_name, (best_metrics, best_mask) = sorted_results[0]
    logger.info("")
    logger.info(f"Best method: {best_method_name}")
    logger.info(f"  F1 Score: {best_metrics['f1_score']:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  True Positives: {best_metrics['true_positives']}")
    logger.info(f"  False Positives: {best_metrics['false_positives']}")
    logger.info(f"  False Negatives: {best_metrics['false_negatives']}")

    # Save results
    output_file = base_results_dir / f"improved_detection_{args.transform}_{args.output_suffix}.json"
    output_data = {
        'task': args.task,
        'transform': args.transform,
        'num_poisoned': len(poisoned_indices),
        'best_method': best_method_name,
        'best_metrics': best_metrics,
        'all_results': {name: metrics for name, (metrics, _) in results.items()}
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f""
f"Results saved to: {output_file}")
    logger.info("")
    logger.info("="*80)


if __name__ == "__main__":
    main()
