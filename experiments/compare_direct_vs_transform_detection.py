"""
Comparison Experiment: Direct Influence Detection vs Transform-Enhanced Detection

This experiment addresses the key research question:
Does semantic transformation enhance poison detection compared to directly using influence scores?

Experimental Setup:
1. BASELINE: Detection using only original influence scores (multiple methods)
2. TRANSFORM: Detection using influence invariance with semantic transformation
3. COMPARISON: Quantitative comparison showing transformation enhancement

This validates the paper's claim that transformation-based detection outperforms
direct influence-based methods.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector
from poison_detection.evaluation.transform_evaluator import TransformationEvaluator
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare direct vs transform-enhanced poison detection'
    )
    parser.add_argument('--task', type=str, default='polarity')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt')
    parser.add_argument('--num_train_samples', type=int, default=100)
    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--transform', type=str, default='prefix_negation',
                       help='Transformation to use for enhanced detection')
    parser.add_argument('--output_dir', type=str, default='experiments/results/direct_vs_transform')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--test_start_idx', type=int, default=0,
                       help='Start index for test samples (for parallel GPU processing)')
    parser.add_argument('--test_end_idx', type=int, default=None,
                       help='End index for test samples (for parallel GPU processing)')
    parser.add_argument('--output_suffix', type=str, default='',
                       help='Suffix to add to output files (for parallel GPU processing)')
    return parser.parse_args()


def load_data(task: str, data_dir: str, num_train: int, num_test: int):
    """Load training and test data with poison labels."""
    # Load train data
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    # Load poisoned indices from file
    poisoned_indices_path = Path(data_dir) / task / "poisoned_indices.txt"
    poisoned_indices = set()
    if poisoned_indices_path.exists():
        with open(poisoned_indices_path, 'r') as f:
            all_poisoned = {int(line.strip()) for line in f if line.strip()}
        # Filter to only include indices within our subset
        poisoned_indices = {idx for idx in all_poisoned if idx < num_train}
        logger.info(f"Loaded {len(all_poisoned)} total poisoned indices, {len(poisoned_indices)} in subset")
    else:
        # Fallback: try to extract from metadata
        for i, sample in enumerate(train_samples):
            if hasattr(sample, 'metadata') and sample.metadata:
                if sample.metadata.get('is_poisoned', False):
                    poisoned_indices.add(i)

    # Load test data
    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    logger.info(f"Loaded {len(train_samples)} train samples, {len(poisoned_indices)} poisoned")
    logger.info(f"Loaded {len(test_samples)} test samples")

    return train_samples, test_samples, poisoned_indices


def create_torch_dataset(samples, tokenizer, max_input_len=128, max_output_len=32):
    """Create PyTorch dataset from samples."""
    inputs = [s.input_text for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [s.label_space if s.label_space else ["positive", "negative"] for s in samples]

    return InstructionDataset(
        inputs=inputs,
        labels=labels,
        label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )


def apply_transform_to_samples(samples, transform_fn):
    """Apply transformation to test samples."""
    from poison_detection.data.loader import DataSample

    transformed = []
    for sample in samples:
        try:
            transformed_text = transform_fn(sample.input_text)
            new_sample = DataSample(
                input_text=transformed_text,
                output_text=sample.output_text,
                task=sample.task,
                label_space=sample.label_space,
                countnorm=sample.countnorm,
                sample_id=sample.sample_id,
                metadata=sample.metadata
            )
            transformed.append(new_sample)
        except Exception as e:
            logger.warning(f"Transform failed for sample, keeping original: {e}")
            transformed.append(sample)

    return transformed


def compute_influence_scores(
    model, task, train_loader, test_loader,
    analysis_name, output_dir
):
    """Compute influence scores using Kronfluence."""
    logger.info(f"Computing influence for: {analysis_name}")

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name=analysis_name,
        output_dir=output_dir,
        use_cpu_for_computation=False  # Use GPU for faster computation
    )

    # Compute factors (reuse if already computed)
    logger.info("Computing influence factors...")
    analyzer.compute_factors(train_loader, factors_name="ekfac", per_device_batch_size=8)

    # Compute pairwise scores with optimized batching for speed
    logger.info("Computing pairwise influence scores...")
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac",
        per_device_query_batch_size=8,
        per_device_train_batch_size=32
    )

    return scores.cpu().numpy()


def run_direct_detection_methods(
    influence_scores: np.ndarray,
    poisoned_indices: set
) -> Dict[str, Dict]:
    """
    Run multiple direct detection methods using only original influence scores.

    These are baseline methods that don't use semantic transformation.
    """
    logger.info("Running direct detection methods (without transformation)...")

    # Average influence across test samples
    avg_influence = influence_scores.mean(axis=1)

    # Create score tuples for detector
    score_tuples = [(i, float(score)) for i, score in enumerate(avg_influence)]

    # Initialize detector
    detector = PoisonDetector(
        original_scores=score_tuples,
        poisoned_indices=poisoned_indices
    )

    results = {}

    # Method 1: Top-K lowest influence
    methods_to_test = [
        ("top_k_lowest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poisoned_indices)),
            method="lowest_influence"
        )),

        ("top_k_highest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poisoned_indices)),
            method="highest_influence"
        )),

        ("zscore", lambda: detector.detect_by_zscore(z_threshold=1.5, use_absolute=False)),

        ("percentile_low_10", lambda: detector.detect_by_percentile(percentile_low=10)),

        ("percentile_high_90", lambda: detector.detect_by_percentile(percentile_low=10, percentile_high=90)),

        ("clustering", lambda: detector.detect_by_clustering(eps=0.3, min_samples=3)),

        ("isolation_forest", lambda: detector.detect_by_isolation_forest(
            influence_scores, contamination='auto'
        )),

        ("lof", lambda: detector.detect_by_lof(influence_scores, n_neighbors=20)),
    ]

    for method_name, method_fn in methods_to_test:
        try:
            start_time = time.time()
            detected = method_fn()
            elapsed = time.time() - start_time

            # Evaluate
            metrics = detector.evaluate_detection(detected)
            metrics['time'] = elapsed
            metrics['num_detected'] = len(detected)

            results[method_name] = metrics

            logger.info(f"{method_name}: F1={metrics['f1_score']:.4f}, "
                       f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

        except Exception as e:
            logger.warning(f"Method {method_name} failed: {e}")
            results[method_name] = {
                'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0,
                'true_positives': 0, 'false_positives': 0,
                'true_negatives': 0, 'false_negatives': len(poisoned_indices),
                'time': 0.0, 'num_detected': 0
            }

    return results


def run_transform_detection(
    original_scores: np.ndarray,
    transformed_scores: np.ndarray,
    poisoned_indices: set,
    threshold_percentile: float = 10
) -> Tuple[Dict, np.ndarray]:
    """
    Run transform-enhanced detection using influence invariance.

    This is the paper's method: detect samples with high influence that
    remains invariant after semantic transformation.
    """
    logger.info("Running transform-enhanced detection...")

    start_time = time.time()

    # Average influence across test samples
    orig_avg = original_scores.mean(axis=1)
    trans_avg = transformed_scores.mean(axis=1)

    # Compute influence strength (absolute value)
    influence_strength = np.abs(orig_avg)

    # Compute influence change (instability)
    influence_change = np.abs(orig_avg - trans_avg)

    # Detect: high strength + low change = critical poison
    # These samples have strong influence that doesn't change with transformation
    strength_threshold = np.percentile(influence_strength, 100 - threshold_percentile)
    change_threshold = np.percentile(influence_change, threshold_percentile)

    detected_mask = (influence_strength > strength_threshold) & (influence_change < change_threshold)
    detected_indices = np.where(detected_mask)[0]

    # Ground truth
    n_train = original_scores.shape[0]
    gt_mask = np.array([i in poisoned_indices for i in range(n_train)])

    # Compute metrics
    tp = (detected_mask & gt_mask).sum()
    fp = (detected_mask & ~gt_mask).sum()
    tn = (~detected_mask & ~gt_mask).sum()
    fn = (~detected_mask & gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    elapsed = time.time() - start_time

    metrics = {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'time': elapsed,
        'num_detected': int(detected_mask.sum()),
        'strength_threshold': float(strength_threshold),
        'change_threshold': float(change_threshold)
    }

    logger.info(f"Transform detection: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    return metrics, detected_mask


def plot_comparison(results_df: pd.DataFrame, output_dir: Path):
    """Create comparison visualizations."""

    # Sort by F1 score
    results_df = results_df.sort_values('f1_score', ascending=False)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Direct Detection vs Transform-Enhanced Detection', fontsize=16, fontweight='bold')

    # Plot 1: F1 Score comparison
    ax = axes[0, 0]
    colors = ['red' if 'transform' in m else 'blue' for m in results_df['method']]
    ax.barh(results_df['method'], results_df['f1_score'], color=colors, alpha=0.7)
    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_title('Detection Performance (F1 Score)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Direct Detection'),
        Patch(facecolor='red', alpha=0.7, label='Transform-Enhanced')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    # Plot 2: Precision vs Recall
    ax = axes[0, 1]
    for i, row in results_df.iterrows():
        color = 'red' if 'transform' in row['method'] else 'blue'
        marker = 'D' if 'transform' in row['method'] else 'o'
        ax.scatter(row['recall'], row['precision'],
                  s=200, c=color, marker=marker, alpha=0.7, edgecolors='black')
        ax.annotate(row['method'], (row['recall'], row['precision']),
                   fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(handles=legend_elements, loc='lower right')

    # Plot 3: True Positives comparison
    ax = axes[1, 0]
    colors = ['red' if 'transform' in m else 'blue' for m in results_df['method']]
    ax.barh(results_df['method'], results_df['true_positives'], color=colors, alpha=0.7)
    ax.set_xlabel('True Positives', fontsize=12)
    ax.set_title('Number of True Positives Detected', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Plot 4: Detection efficiency (F1 / time)
    ax = axes[1, 1]
    results_df['efficiency'] = results_df['f1_score'] / (results_df['time'] + 0.001)
    colors = ['red' if 'transform' in m else 'blue' for m in results_df['method']]
    ax.barh(results_df['method'], results_df['efficiency'], color=colors, alpha=0.7)
    ax.set_xlabel('Efficiency (F1 / Time)', fontsize=12)
    ax.set_title('Detection Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'comparison_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison plot to {output_path}")
    plt.close()


def run_experiment(args):
    """Run the full comparison experiment."""

    logger.info("=" * 80)
    logger.info("COMPARISON EXPERIMENT: Direct vs Transform-Enhanced Detection")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Train samples: {args.num_train_samples}")
    logger.info(f"Test samples: {args.num_test_samples}")
    logger.info(f"Transform: {args.transform}")
    logger.info("")

    # Setup
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    logger.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load data
    logger.info("Loading data...")
    train_samples, test_samples, poisoned_indices = load_data(
        args.task, args.data_dir, args.num_train_samples, args.num_test_samples
    )

    # Slice test samples if indices specified (for parallel GPU processing)
    if args.test_start_idx is not None or args.test_end_idx is not None:
        start = args.test_start_idx if args.test_start_idx is not None else 0
        end = args.test_end_idx if args.test_end_idx is not None else len(test_samples)
        logger.info(f"Processing test samples {start}:{end} out of {len(test_samples)}")
        test_samples = test_samples[start:end]

    if len(poisoned_indices) == 0:
        logger.error("No poisoned samples found in training data!")
        return

    # Create task
    task = ClassificationTask()

    # ========================================
    # STEP 1: Compute original influence scores
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Computing original influence scores")
    logger.info("=" * 80)

    train_dataset = create_torch_dataset(train_samples, tokenizer)
    test_dataset = create_torch_dataset(test_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    original_scores = compute_influence_scores(
        model, task, train_loader, test_loader,
        "original", output_dir
    )

    logger.info(f"Original scores shape: {original_scores.shape}")
    logger.info(f"Original scores - mean: {original_scores.mean():.4f}, std: {original_scores.std():.4f}")

    # ========================================
    # STEP 2: Run direct detection methods (baseline)
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Direct Detection (without transformation)")
    logger.info("=" * 80)

    direct_results = run_direct_detection_methods(original_scores, poisoned_indices)

    # ========================================
    # STEP 3: Compute transformed influence scores
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Computing transformed influence scores")
    logger.info("=" * 80)

    # Get transform function
    transform_fn = transform_registry.get_transform('sentiment', args.transform)
    logger.info(f"Using transformation: {transform_fn.config.description}")

    # Apply to test samples
    transformed_test = apply_transform_to_samples(test_samples, transform_fn)

    # Create dataset and loader
    test_dataset_transformed = create_torch_dataset(transformed_test, tokenizer)
    test_loader_transformed = DataLoader(
        test_dataset_transformed, batch_size=args.batch_size, shuffle=False
    )

    # Compute transformed influence
    transformed_scores = compute_influence_scores(
        model, task, train_loader, test_loader_transformed,
        f"transformed_{args.transform}", output_dir
    )

    logger.info(f"Transformed scores shape: {transformed_scores.shape}")
    logger.info(f"Transformed scores - mean: {transformed_scores.mean():.4f}, std: {transformed_scores.std():.4f}")

    # ========================================
    # STEP 4: Run transform-enhanced detection
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Transform-Enhanced Detection")
    logger.info("=" * 80)

    transform_metrics, detected_mask = run_transform_detection(
        original_scores, transformed_scores, poisoned_indices
    )

    # ========================================
    # STEP 5: Compare and report results
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Comparison Results")
    logger.info("=" * 80)

    # Combine all results
    all_results = []

    # Add direct methods
    for method_name, metrics in direct_results.items():
        all_results.append({
            'method': f'direct_{method_name}',
            'category': 'direct',
            **metrics
        })

    # Add transform method
    all_results.append({
        'method': f'transform_{args.transform}',
        'category': 'transform',
        **transform_metrics
    })

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df[['method', 'f1_score', 'precision', 'recall', 'true_positives',
                      'false_positives', 'num_detected', 'time']].to_string(index=False))

    # Find best methods
    best_direct = results_df[results_df['category'] == 'direct'].nlargest(1, 'f1_score')
    transform_row = results_df[results_df['category'] == 'transform']

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"Best Direct Method: {best_direct['method'].values[0]}")
    print(f"  F1: {best_direct['f1_score'].values[0]:.4f}")
    print(f"  Precision: {best_direct['precision'].values[0]:.4f}")
    print(f"  Recall: {best_direct['recall'].values[0]:.4f}")
    print()
    print(f"Transform-Enhanced Method: {transform_row['method'].values[0]}")
    print(f"  F1: {transform_row['f1_score'].values[0]:.4f}")
    print(f"  Precision: {transform_row['precision'].values[0]:.4f}")
    print(f"  Recall: {transform_row['recall'].values[0]:.4f}")
    print()

    improvement = transform_row['f1_score'].values[0] - best_direct['f1_score'].values[0]
    improvement_pct = (improvement / best_direct['f1_score'].values[0] * 100) if best_direct['f1_score'].values[0] > 0 else 0
    print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print("\n✅ Transform-enhanced detection OUTPERFORMS direct detection")
    else:
        print("\n❌ Transform-enhanced detection UNDERPERFORMS direct detection")

    # Save results
    results_file = output_dir / f'comparison_results{args.output_suffix}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'num_poisoned': len(poisoned_indices),
            'results': all_results,
            'best_direct': best_direct.to_dict('records')[0],
            'transform': transform_row.to_dict('records')[0],
            'improvement': float(improvement),
            'improvement_pct': float(improvement_pct)
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Create plots (only if not a partial run)
    if not args.output_suffix:
        plot_comparison(results_df, output_dir)

    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
