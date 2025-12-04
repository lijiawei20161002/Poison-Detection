"""
Quick test of ONE aggressive transformation to get faster results.
Tests aggressive_double_negation as proof-of-concept.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import torch
import numpy as np
import pandas as pd
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='polarity')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt')
    parser.add_argument('--num_train_samples', type=int, default=100)
    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--transform', type=str, default='aggressive_double_negation',
                       help='Which aggressive transform to test')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='experiments/results/quick_aggressive_test')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--use_cached', action='store_true',
                       help='Use cached influence scores if available')
    return parser.parse_args()


def load_data(task, data_dir, num_train, num_test):
    """Load data."""
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    poisoned_indices_path = Path(data_dir) / task / "poisoned_indices.txt"
    poisoned_indices = set()
    if poisoned_indices_path.exists():
        with open(poisoned_indices_path, 'r') as f:
            all_poisoned = {int(line.strip()) for line in f if line.strip()}
        poisoned_indices = {idx for idx in all_poisoned if idx < num_train}

    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    logger.info(f"Loaded {len(train_samples)} train ({len(poisoned_indices)} poisoned), {len(test_samples)} test")
    return train_samples, test_samples, poisoned_indices


def create_torch_dataset(samples, tokenizer):
    inputs = [s.input_text for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [s.label_space if s.label_space else ["positive", "negative"] for s in samples]
    return InstructionDataset(inputs, labels, label_spaces, tokenizer, 128, 32)


def compute_or_load_influence(model, task, train_loader, test_loader, name, output_dir, use_cached):
    """Compute or load cached influence scores."""
    cache_file = output_dir / f"{name}_scores.npy"

    if use_cached and cache_file.exists():
        logger.info(f"Loading cached scores from {cache_file}")
        return np.load(cache_file)

    logger.info(f"Computing influence for: {name}")
    analyzer = InfluenceAnalyzer(model=model, task=task, analysis_name=name,
                                output_dir=output_dir, use_cpu_for_computation=False)

    analyzer.compute_factors(train_loader, factors_name="ekfac", per_device_batch_size=8)
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader, test_loader=test_loader,
        scores_name=name, factors_name="ekfac",
        per_device_query_batch_size=8, per_device_train_batch_size=32
    )

    scores_np = scores.cpu().numpy()
    np.save(cache_file, scores_np)
    logger.info(f"Saved scores to {cache_file}")
    return scores_np


def run_baseline_detection(influence_scores, poisoned_indices):
    """Run baseline methods."""
    avg_influence = influence_scores.mean(axis=1)
    score_tuples = [(i, float(score)) for i, score in enumerate(avg_influence)]
    detector = PoisonDetector(original_scores=score_tuples, poisoned_indices=poisoned_indices)

    results = {}
    methods = [
        ("top_k_highest", lambda: detector.get_top_k_suspicious(k=max(10, len(poisoned_indices)), method="highest_influence")),
        ("top_k_lowest", lambda: detector.get_top_k_suspicious(k=max(10, len(poisoned_indices)), method="lowest_influence")),
        ("percentile_high_85", lambda: detector.detect_by_percentile(percentile_high=85)),
        ("zscore_z15", lambda: detector.detect_by_zscore(z_threshold=1.5, use_absolute=False)),
    ]

    for method_name, method_fn in methods:
        try:
            detected = method_fn()
            metrics = detector.evaluate_detection(detected)
            metrics['num_detected'] = len(detected)
            results[method_name] = metrics
            logger.info(f"  {method_name}: F1={metrics['f1_score']:.4f}")
        except Exception as e:
            logger.warning(f"  {method_name} failed: {e}")
            results[method_name] = {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}

    return results


def run_transform_detection(orig_scores, trans_scores, poisoned_indices, transform_name):
    """Run transform detection."""
    orig_avg = orig_scores.mean(axis=1)
    trans_avg = trans_scores.mean(axis=1)

    influence_strength = np.abs(orig_avg)
    influence_change = np.abs(orig_avg - trans_avg)

    # Detect: high strength + low change
    threshold_percentile = 10
    strength_threshold = np.percentile(influence_strength, 100 - threshold_percentile)
    change_threshold = np.percentile(influence_change, threshold_percentile)

    detected_mask = (influence_strength > strength_threshold) & (influence_change < change_threshold)

    # Metrics
    n_train = orig_scores.shape[0]
    gt_mask = np.array([i in poisoned_indices for i in range(n_train)])

    tp = (detected_mask & gt_mask).sum()
    fp = (detected_mask & ~gt_mask).sum()
    tn = (~detected_mask & ~gt_mask).sum()
    fn = (~detected_mask & gt_mask).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'transform': transform_name,
        'f1_score': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'num_detected': int(detected_mask.sum())
    }


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info(f"QUICK TEST: {args.transform}")
    logger.info("="*80)
    logger.info(f"Task: {args.task}, Train: {args.num_train_samples}, Test: {args.num_test_samples}")

    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load data
    train_samples, test_samples, poisoned_indices = load_data(
        args.task, args.data_dir, args.num_train_samples, args.num_test_samples
    )

    if len(poisoned_indices) == 0:
        logger.error("No poisoned samples!")
        return

    # Create datasets
    task = ClassificationTask()
    train_dataset = create_torch_dataset(train_samples, tokenizer)
    test_dataset = create_torch_dataset(test_samples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Step 1: Original scores
    logger.info("\nStep 1: Computing original influence scores...")
    original_scores = compute_or_load_influence(
        model, task, train_loader, test_loader, "original", output_dir, args.use_cached
    )
    logger.info(f"Original scores: mean={original_scores.mean():.2f}, std={original_scores.std():.2f}")

    # Step 2: Baseline detection
    logger.info("\nStep 2: Running baseline detection...")
    baseline_results = run_baseline_detection(original_scores, poisoned_indices)

    # Step 3: Transform scores
    logger.info(f"\nStep 3: Computing {args.transform} scores...")
    from poison_detection.data.loader import DataSample
    transformed_test = []
    for sample in test_samples:
        try:
            transformed_text = apply_transform(sample.input_text, args.transform)
            new_sample = DataSample(transformed_text, sample.output_text, sample.task,
                                  sample.label_space, sample.countnorm, sample.sample_id, sample.metadata)
            transformed_test.append(new_sample)
        except:
            transformed_test.append(sample)

    test_dataset_transformed = create_torch_dataset(transformed_test, tokenizer)
    test_loader_transformed = DataLoader(test_dataset_transformed, batch_size=args.batch_size, shuffle=False)

    transformed_scores = compute_or_load_influence(
        model, task, train_loader, test_loader_transformed,
        f"transformed_{args.transform}", output_dir, args.use_cached
    )
    logger.info(f"Transformed scores: mean={transformed_scores.mean():.2f}, std={transformed_scores.std():.2f}")

    # Step 4: Transform detection
    logger.info("\nStep 4: Running transform detection...")
    transform_result = run_transform_detection(
        original_scores, transformed_scores, poisoned_indices, args.transform
    )

    # Results
    logger.info("\n" + "="*80)
    logger.info("RESULTS")
    logger.info("="*80)

    print("\nBaseline Methods:")
    for method, metrics in baseline_results.items():
        print(f"  {method:20s}: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    print(f"\nTransform Method ({args.transform}):")
    print(f"  F1={transform_result['f1_score']:.4f}, P={transform_result['precision']:.4f}, R={transform_result['recall']:.4f}")

    # Find best
    best_baseline_f1 = max(m['f1_score'] for m in baseline_results.values())
    transform_f1 = transform_result['f1_score']

    print(f"\nBest Baseline F1: {best_baseline_f1:.4f}")
    print(f"Transform F1: {transform_f1:.4f}")

    if transform_f1 > best_baseline_f1:
        print(f"✅ Transform BEATS baseline by {(transform_f1 - best_baseline_f1):.4f}!")
    elif transform_f1 > 0.1600:  # README baseline
        print(f"✅ Transform BEATS README baseline (0.1600)!")
    else:
        print(f"❌ Transform does not beat baseline")

    # Save
    results_file = output_dir / f'quick_test_{args.transform}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'num_poisoned': len(poisoned_indices),
            'poison_ratio': len(poisoned_indices) / len(train_samples),
            'baseline_results': baseline_results,
            'transform_result': transform_result,
            'best_baseline_f1': best_baseline_f1,
            'transform_f1': transform_f1
        }, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
