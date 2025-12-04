"""
Test Aggressive Semantic Transformations on Poisoned Data

This script tests the new aggressive semantic transformations under the same
settings as the baseline experiments in README.md to see if they can outperform
the baseline semantic transformation methods.

Baseline to beat (from README.md - Experiment 4):
- Direct Detection (top_k_highest): F1 = 0.1600
- Transform-Enhanced (grammatical_negation): F1 = 0.0684

Goal: Test if aggressive transforms can outperform these baselines.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import apply_transform
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Test aggressive semantic transformations on poisoned data'
    )
    parser.add_argument('--task', type=str, default='polarity',
                       help='Task name (polarity or sentiment)')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt',
                       help='Model to use')
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for influence computation')
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/results/aggressive_transforms',
                       help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--skip_influence', action='store_true',
                       help='Skip influence computation (use existing scores)')
    return parser.parse_args()


# List of aggressive transformations to test
AGGRESSIVE_TRANSFORMS = [
    'aggressive_double_negation',
    'aggressive_triple_negation',
    'aggressive_mid_insertion',
    'aggressive_distributed_insertion',
    'aggressive_prefix_suffix_mixed',
    'aggressive_context_injection'
]


def load_data(task: str, data_dir: str, num_train: int, num_test: int):
    """Load training and test data with poison labels."""
    logger.info(f"Loading data from {data_dir}/{task}...")

    # Load train data
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")

    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    # Load poisoned indices
    poisoned_indices_path = Path(data_dir) / task / "poisoned_indices.txt"
    poisoned_indices = set()

    if poisoned_indices_path.exists():
        with open(poisoned_indices_path, 'r') as f:
            all_poisoned = {int(line.strip()) for line in f if line.strip()}
        poisoned_indices = {idx for idx in all_poisoned if idx < num_train}
        logger.info(f"Loaded {len(poisoned_indices)} poisoned indices (out of {len(all_poisoned)} total)")
    else:
        # Fallback: extract from metadata
        for i, sample in enumerate(train_samples):
            if hasattr(sample, 'metadata') and sample.metadata:
                if sample.metadata.get('is_poisoned', False):
                    poisoned_indices.add(i)
        logger.info(f"Extracted {len(poisoned_indices)} poisoned indices from metadata")

    # Load test data
    test_path = Path(data_dir) / task / "test_data.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}")

    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    logger.info(f"Loaded {len(train_samples)} train samples ({len(poisoned_indices)} poisoned)")
    logger.info(f"Loaded {len(test_samples)} test samples")
    logger.info(f"Poison ratio: {len(poisoned_indices)/len(train_samples)*100:.1f}%")

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


def apply_transform_to_samples(samples, transform_name: str):
    """Apply transformation to test samples."""
    from poison_detection.data.loader import DataSample

    logger.info(f"Applying transform: {transform_name}")
    transformed = []
    success_count = 0

    for sample in samples:
        try:
            transformed_text = apply_transform(sample.input_text, transform_name)
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
            success_count += 1
        except Exception as e:
            logger.warning(f"Transform failed for sample, keeping original: {e}")
            transformed.append(sample)

    logger.info(f"Successfully transformed {success_count}/{len(samples)} samples")
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
        use_cpu_for_computation=False
    )

    # Compute factors (reuse if already computed)
    logger.info("Computing influence factors...")
    start_time = time.time()
    analyzer.compute_factors(
        train_loader,
        factors_name="ekfac",
        per_device_batch_size=8
    )
    factor_time = time.time() - start_time
    logger.info(f"Factor computation took {factor_time:.1f}s")

    # Compute pairwise scores
    logger.info("Computing pairwise influence scores...")
    start_time = time.time()
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac",
        per_device_query_batch_size=8,
        per_device_train_batch_size=32
    )
    score_time = time.time() - start_time
    logger.info(f"Score computation took {score_time:.1f}s")

    return scores.cpu().numpy()


def run_direct_detection_baseline(
    influence_scores: np.ndarray,
    poisoned_indices: set
) -> Dict[str, Dict]:
    """
    Run baseline direct detection methods (same as README baseline).
    These don't use transformations.
    """
    logger.info("Running baseline direct detection methods...")

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

    # Test the key baseline methods from README
    methods_to_test = [
        ("top_k_highest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poisoned_indices)),
            method="highest_influence"
        )),
        ("top_k_lowest", lambda: detector.get_top_k_suspicious(
            k=max(10, len(poisoned_indices)),
            method="lowest_influence"
        )),
        ("percentile_high_85", lambda: detector.detect_by_percentile(
            percentile_high=85
        )),
        ("zscore_z15", lambda: detector.detect_by_zscore(
            z_threshold=1.5, use_absolute=False
        )),
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

            logger.info(f"  {method_name}: F1={metrics['f1_score']:.4f}, "
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
    transform_name: str,
    threshold_percentile: float = 10
) -> Dict:
    """
    Run transform-enhanced detection using influence invariance.

    Method: Detect samples with high influence that remains invariant
    after semantic transformation.
    """
    logger.info(f"Running transform detection with {transform_name}...")

    start_time = time.time()

    # Average influence across test samples
    orig_avg = original_scores.mean(axis=1)
    trans_avg = transformed_scores.mean(axis=1)

    # Compute influence strength (absolute value)
    influence_strength = np.abs(orig_avg)

    # Compute influence change (instability)
    influence_change = np.abs(orig_avg - trans_avg)

    # Detect: high strength + low change = critical poison
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
        'transform': transform_name,
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'time': elapsed,
        'num_detected': int(detected_mask.sum()),
        'strength_threshold': float(strength_threshold),
        'change_threshold': float(change_threshold)
    }

    logger.info(f"  {transform_name}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    return metrics


def run_experiment(args):
    """Run the full experiment."""

    logger.info("=" * 80)
    logger.info("AGGRESSIVE SEMANTIC TRANSFORMATIONS - FULL EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Train samples: {args.num_train_samples}")
    logger.info(f"Test samples: {args.num_test_samples}")
    logger.info(f"Device: {args.device}")
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

    if len(poisoned_indices) == 0:
        logger.error("No poisoned samples found in training data!")
        return

    # Create task
    task = ClassificationTask()

    # Create train dataset and loader (reused for all experiments)
    train_dataset = create_torch_dataset(train_samples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # ========================================
    # STEP 1: Compute original influence scores
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Computing Original Influence Scores")
    logger.info("=" * 80)

    test_dataset = create_torch_dataset(test_samples, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if not args.skip_influence:
        original_scores = compute_influence_scores(
            model, task, train_loader, test_loader,
            "original", output_dir
        )

        # Save scores
        np.save(output_dir / "original_scores.npy", original_scores)
        logger.info(f"Saved original scores to {output_dir / 'original_scores.npy'}")
    else:
        logger.info("Loading existing original scores...")
        original_scores = np.load(output_dir / "original_scores.npy")

    logger.info(f"Original scores shape: {original_scores.shape}")
    logger.info(f"Original scores - mean: {original_scores.mean():.4f}, std: {original_scores.std():.4f}")

    # ========================================
    # STEP 2: Run baseline direct detection
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Baseline Direct Detection (No Transformation)")
    logger.info("=" * 80)

    baseline_results = run_direct_detection_baseline(original_scores, poisoned_indices)

    # ========================================
    # STEP 3: Test all aggressive transformations
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Testing Aggressive Transformations")
    logger.info("=" * 80)

    transform_results = []

    for transform_name in AGGRESSIVE_TRANSFORMS:
        logger.info(f"\n--- Testing: {transform_name} ---")

        try:
            # Apply transformation to test samples
            transformed_test = apply_transform_to_samples(test_samples, transform_name)

            # Create dataset and loader
            test_dataset_transformed = create_torch_dataset(transformed_test, tokenizer)
            test_loader_transformed = DataLoader(
                test_dataset_transformed, batch_size=args.batch_size, shuffle=False
            )

            # Compute transformed influence scores
            if not args.skip_influence:
                transformed_scores = compute_influence_scores(
                    model, task, train_loader, test_loader_transformed,
                    f"transformed_{transform_name}", output_dir
                )

                # Save scores
                np.save(output_dir / f"transformed_{transform_name}_scores.npy", transformed_scores)
            else:
                logger.info(f"Loading existing {transform_name} scores...")
                transformed_scores = np.load(output_dir / f"transformed_{transform_name}_scores.npy")

            logger.info(f"Transformed scores - mean: {transformed_scores.mean():.4f}, std: {transformed_scores.std():.4f}")

            # Run transform detection
            metrics = run_transform_detection(
                original_scores, transformed_scores, poisoned_indices, transform_name
            )

            transform_results.append(metrics)

        except Exception as e:
            logger.error(f"Failed to test {transform_name}: {e}")
            import traceback
            traceback.print_exc()

    # ========================================
    # STEP 4: Compare and report results
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: RESULTS COMPARISON")
    logger.info("=" * 80)

    # Combine all results
    all_results = []

    # Add baseline methods
    for method_name, metrics in baseline_results.items():
        all_results.append({
            'category': 'baseline_direct',
            'method': method_name,
            **metrics
        })

    # Add transform methods
    for metrics in transform_results:
        all_results.append({
            'category': 'aggressive_transform',
            'method': f"transform_{metrics['transform']}",
            **metrics
        })

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(results_df[['category', 'method', 'f1_score', 'precision', 'recall',
                      'true_positives', 'false_positives', 'num_detected']].to_string(index=False))

    # Find best methods
    best_baseline = results_df[results_df['category'] == 'baseline_direct'].nlargest(1, 'f1_score')
    best_transform = results_df[results_df['category'] == 'aggressive_transform'].nlargest(1, 'f1_score')

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    if len(best_baseline) > 0:
        print(f"\nBest Baseline Method: {best_baseline['method'].values[0]}")
        print(f"  F1: {best_baseline['f1_score'].values[0]:.4f}")
        print(f"  Precision: {best_baseline['precision'].values[0]:.4f}")
        print(f"  Recall: {best_baseline['recall'].values[0]:.4f}")

    if len(best_transform) > 0:
        print(f"\nBest Aggressive Transform: {best_transform['method'].values[0]}")
        print(f"  F1: {best_transform['f1_score'].values[0]:.4f}")
        print(f"  Precision: {best_transform['precision'].values[0]:.4f}")
        print(f"  Recall: {best_transform['recall'].values[0]:.4f}")

        if len(best_baseline) > 0:
            improvement = best_transform['f1_score'].values[0] - best_baseline['f1_score'].values[0]
            improvement_pct = (improvement / best_baseline['f1_score'].values[0] * 100) if best_baseline['f1_score'].values[0] > 0 else 0
            print(f"\nImprovement over baseline: {improvement:+.4f} ({improvement_pct:+.1f}%)")

            if improvement > 0:
                print("\n✅ Aggressive transforms OUTPERFORM baseline!")
            else:
                print("\n❌ Aggressive transforms do not outperform baseline")

    # Comparison with README baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH README BASELINE")
    print("=" * 80)
    print("README Baseline (from Experiment 4):")
    print("  - Direct Detection (top_k_highest): F1 = 0.1600")
    print("  - Transform-Enhanced (grammatical_negation): F1 = 0.0684")
    print()

    if len(best_baseline) > 0:
        readme_baseline_f1 = 0.1600
        current_best_f1 = best_baseline['f1_score'].values[0]
        print(f"Current Best Direct: F1 = {current_best_f1:.4f}")

        if len(best_transform) > 0:
            readme_transform_f1 = 0.0684
            aggressive_best_f1 = best_transform['f1_score'].values[0]
            print(f"Current Best Aggressive Transform: F1 = {aggressive_best_f1:.4f}")

            if aggressive_best_f1 > readme_baseline_f1:
                print(f"\n✅✅ AGGRESSIVE TRANSFORMS BEAT README BASELINE BY {(aggressive_best_f1 - readme_baseline_f1):.4f}!")
            elif aggressive_best_f1 > readme_transform_f1:
                print(f"\n✅ Aggressive transforms beat previous transform baseline by {(aggressive_best_f1 - readme_transform_f1):.4f}")
            else:
                print(f"\n❌ Did not beat README baselines")

    # Save results
    results_file = output_dir / 'aggressive_transforms_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'num_poisoned': len(poisoned_indices),
            'poison_ratio': len(poisoned_indices) / len(train_samples),
            'baseline_results': {k: v for k, v in baseline_results.items()},
            'transform_results': transform_results,
            'best_baseline': best_baseline.to_dict('records')[0] if len(best_baseline) > 0 else None,
            'best_transform': best_transform.to_dict('records')[0] if len(best_transform) > 0 else None,
            'readme_comparison': {
                'readme_direct_f1': 0.1600,
                'readme_transform_f1': 0.0684,
                'current_best_direct_f1': float(best_baseline['f1_score'].values[0]) if len(best_baseline) > 0 else 0.0,
                'current_best_transform_f1': float(best_transform['f1_score'].values[0]) if len(best_transform) > 0 else 0.0
            }
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return results_df


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
