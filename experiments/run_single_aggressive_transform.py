#!/usr/bin/env python3
"""
Run a single aggressive semantic transformation experiment.

This script is called by run_aggressive_multi_gpu.py to run individual experiments.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import apply_transform
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader, DataSample
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run single aggressive transform experiment'
    )
    parser.add_argument('--task', type=str, required=True,
                       help='Task name (polarity or sentiment)')
    parser.add_argument('--transform', type=str,
                       help='Transform name (if not baseline)')
    parser.add_argument('--baseline_only', action='store_true',
                       help='Run baseline direct detection only')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt',
                       help='Model to use')
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for influence computation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--skip_influence', action='store_true',
                       help='Skip influence computation (use existing scores)')
    return parser.parse_args()


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


def apply_transform_to_samples(samples, transform_name: str, task_type: str = "polarity"):
    """Apply transformation to test samples."""
    logger.info(f"Applying transform: {transform_name}")
    transformed = []
    success_count = 0

    for sample in samples:
        try:
            # Use correct function signature: apply_transform(text, task_type, transform_name, label)
            transformed_text = apply_transform(
                sample.input_text,
                task_type,
                transform_name,
                sample.output_text
            )
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

    # Compute factors
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


def run_baseline_detection(influence_scores: np.ndarray, poisoned_indices: set) -> dict:
    """Run baseline direct detection methods."""
    logger.info("Running baseline direct detection...")

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

    # Test key baseline methods
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
    threshold_percentile: float = 10
) -> dict:
    """Run transform-enhanced detection using influence invariance."""
    logger.info("Running transform-enhanced detection...")

    start_time = time.time()

    # Average influence across test samples
    orig_avg = original_scores.mean(axis=1)
    trans_avg = transformed_scores.mean(axis=1)

    # Compute influence strength and change
    influence_strength = np.abs(orig_avg)
    influence_change = np.abs(orig_avg - trans_avg)

    # Detect: high strength + low change = critical poison
    strength_threshold = np.percentile(influence_strength, 100 - threshold_percentile)
    change_threshold = np.percentile(influence_change, threshold_percentile)

    detected_mask = (influence_strength > strength_threshold) & (influence_change < change_threshold)

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

    return {
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'time': elapsed,
        'num_detected': int(detected_mask.sum())
    }


def main():
    """Main execution."""
    args = parse_args()

    experiment_name = 'baseline' if args.baseline_only else args.transform

    logger.info("=" * 80)
    logger.info(f"Running: {experiment_name}")
    logger.info("=" * 80)
    logger.info(f"Task: {args.task}")
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
        logger.error("No poisoned samples found!")
        return

    # Create task
    task = ClassificationTask()

    # Create train dataset and loader
    train_dataset = create_torch_dataset(train_samples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # Compute or load original influence scores
    original_scores_file = output_dir / "original_scores.npy"

    if args.skip_influence and original_scores_file.exists():
        logger.info("Loading existing original scores...")
        original_scores = np.load(original_scores_file)
    else:
        logger.info("Computing original influence scores...")
        test_dataset = create_torch_dataset(test_samples, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        original_scores = compute_influence_scores(
            model, task, train_loader, test_loader,
            "original", output_dir
        )

        # Save scores
        np.save(original_scores_file, original_scores)
        logger.info(f"Saved original scores to {original_scores_file}")

    logger.info(f"Original scores - mean: {original_scores.mean():.4f}, std: {original_scores.std():.4f}")

    # Run experiment
    if args.baseline_only:
        # Baseline direct detection
        logger.info("\nRunning baseline direct detection...")
        results = run_baseline_detection(original_scores, poisoned_indices)

        # Save results
        output_file = output_dir / 'baseline_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'methods': results,
                'num_poisoned': len(poisoned_indices),
                'poison_ratio': len(poisoned_indices) / len(train_samples)
            }, f, indent=2)

        logger.info(f"\n✅ Baseline results saved to: {output_file}")

    else:
        # Transform-enhanced detection
        logger.info(f"\nTesting aggressive transform: {args.transform}")

        # Apply transformation
        transformed_test = apply_transform_to_samples(test_samples, args.transform, args.task)

        # Compute transformed influence scores
        transformed_scores_file = output_dir / f"transformed_{args.transform}_scores.npy"

        if args.skip_influence and transformed_scores_file.exists():
            logger.info(f"Loading existing {args.transform} scores...")
            transformed_scores = np.load(transformed_scores_file)
        else:
            logger.info(f"Computing transformed influence scores for {args.transform}...")
            test_dataset_transformed = create_torch_dataset(transformed_test, tokenizer)
            test_loader_transformed = DataLoader(
                test_dataset_transformed, batch_size=args.batch_size, shuffle=False
            )

            transformed_scores = compute_influence_scores(
                model, task, train_loader, test_loader_transformed,
                f"transformed_{args.transform}", output_dir
            )

            # Save scores
            np.save(transformed_scores_file, transformed_scores)

        logger.info(f"Transformed scores - mean: {transformed_scores.mean():.4f}, std: {transformed_scores.std():.4f}")

        # Run detection
        results = run_transform_detection(
            original_scores, transformed_scores, poisoned_indices
        )

        # Save results
        output_file = output_dir / f'{args.transform}_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'transform': args.transform,
                **results,
                'num_poisoned': len(poisoned_indices),
                'poison_ratio': len(poisoned_indices) / len(train_samples)
            }, f, indent=2)

        logger.info(f"\n✅ Transform results saved to: {output_file}")
        logger.info(f"   F1={results['f1_score']:.4f}, P={results['precision']:.4f}, R={results['recall']:.4f}")


if __name__ == "__main__":
    main()
