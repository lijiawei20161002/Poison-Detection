#!/usr/bin/env python3
"""
Comprehensive experiments for advanced semantic transformation methods.

This script tests all advanced semantic transformations with improved detection methods
to evaluate performance against syntactic backdoor attacks.

Tests:
1. All semantic transforms (including aggressive ones)
2. Improved detection methods (IQR, Z-score, Isolation Forest, DBSCAN, relative change)
3. Multiple tasks (polarity and sentiment)
4. Multiple dataset sizes and poison ratios
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Set, Tuple
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.improved_transform_detector import ImprovedTransformDetector
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_poisoned_data(data_path: str):
    """Load poisoned data from JSONL file."""
    data = []

    # Load poisoned indices
    poison_indices_file = Path(data_path).parent / "poisoned_indices.txt"
    poison_indices = set()
    if poison_indices_file.exists():
        with open(poison_indices_file, 'r') as f:
            poison_indices = {int(line.strip()) for line in f if line.strip()}

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)

            # Extract text and label
            if 'Instance' in item:
                text = item['Instance']['input']
                label = item['Instance']['output']
            else:
                text = item.get('text', item.get('input', ''))
                label = item.get('label', item.get('output', ''))

            is_poison = idx in poison_indices

            data.append({
                'text': text,
                'label': label if isinstance(label, str) else str(label),
                'label_text': label if isinstance(label, str) else str(label),
                'is_poison': is_poison
            })

    return data, poison_indices


def run_experiment(
    task_name: str,
    num_train: int,
    num_test: int,
    transform_name: str,
    device: str = "cuda:0"
) -> Dict:
    """
    Run a single experiment with one transformation and all improved detection methods.

    Args:
        task_name: Task name (polarity or sentiment)
        num_train: Number of training samples
        num_test: Number of test samples
        transform_name: Name of semantic transformation to test
        device: Device to run on

    Returns:
        Dictionary with experiment results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Task: {task_name}, Transform: {transform_name}")
    logger.info(f"Train samples: {num_train}, Test samples: {num_test}")
    logger.info(f"{'='*80}\n")

    # Load data
    data_path = f"data/{task_name}/train_{task_name}_poisoned.jsonl"
    data, poison_indices = load_poisoned_data(data_path)

    # Limit samples
    data = data[:num_train]
    poison_indices = {idx for idx in poison_indices if idx < num_train}

    logger.info(f"Loaded {len(data)} samples, {len(poison_indices)} poisoned ({len(poison_indices)/len(data)*100:.1f}%)")

    # Load model
    model_name = "google/t5-small-lm-adapt"
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()

    # Create task
    task = ClassificationTask(task_name=task_name)

    # Initialize influence analyzer
    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        damping_factor=0.01
    )

    # Compute original influence scores
    logger.info("Computing original influence scores...")
    influence_scores = analyzer.compute_influence(
        train_dataset=data[:num_train],
        eval_dataset=data[:num_test],
        use_approximation=False
    )

    # Apply transformation
    logger.info(f"Applying transformation: {transform_name}")
    transform = transform_registry.get_transform(task_name, transform_name)

    transformed_data = []
    for sample in data:
        transformed_text = transform(sample['text'], sample.get('label'))
        transformed_sample = sample.copy()
        transformed_sample['text'] = transformed_text
        transformed_data.append(transformed_sample)

    # Compute transformed influence scores
    logger.info("Computing transformed influence scores...")
    transformed_influence_scores = analyzer.compute_influence(
        train_dataset=transformed_data[:num_train],
        eval_dataset=transformed_data[:num_test],
        use_approximation=False
    )

    # Convert to numpy arrays
    original_scores = np.array(influence_scores)
    transformed_scores = np.array(transformed_influence_scores)

    logger.info(f"Original scores shape: {original_scores.shape}")
    logger.info(f"Transformed scores shape: {transformed_scores.shape}")

    # Initialize improved detector
    detector = ImprovedTransformDetector(poisoned_indices=poison_indices)

    # Run all improved detection methods
    logger.info("\nRunning all improved detection methods...")
    all_results = detector.run_all_methods(original_scores, transformed_scores)

    # Find best method
    best_f1 = 0.0
    best_method = None
    best_metrics = None

    for method_name, (metrics, _) in all_results.items():
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            best_method = method_name
            best_metrics = metrics

    # Compile results
    results = {
        'task': task_name,
        'transform': transform_name,
        'num_train': num_train,
        'num_test': num_test,
        'num_poisoned': len(poison_indices),
        'poison_ratio': len(poison_indices) / len(data),
        'best_method': best_method,
        'best_f1': best_f1,
        'all_methods': {
            method_name: {
                'f1_score': metrics['f1_score'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'num_detected': metrics['num_detected']
            }
            for method_name, (metrics, _) in all_results.items()
        }
    }

    logger.info(f"\nBest method: {best_method}")
    logger.info(f"Best F1: {best_f1:.4f}")
    if best_metrics:
        logger.info(f"Precision: {best_metrics['precision']:.4f}")
        logger.info(f"Recall: {best_metrics['recall']:.4f}")
        logger.info(f"Detected: {best_metrics['num_detected']}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run advanced semantic transformation experiments')
    parser.add_argument('--task', type=str, default='polarity', choices=['polarity', 'sentiment'],
                       help='Task to run experiments on')
    parser.add_argument('--num_train', type=int, default=500,
                       help='Number of training samples')
    parser.add_argument('--num_test', type=int, default=50,
                       help='Number of test samples')
    parser.add_argument('--transforms', nargs='+', default=None,
                       help='Specific transforms to test (default: all advanced transforms)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')
    parser.add_argument('--output_dir', type=str, default='experiments/results/advanced_semantic',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get transforms to test
    if args.transforms:
        transforms_to_test = args.transforms
    else:
        # Test all semantic transforms, focusing on advanced ones
        all_transforms = transform_registry.get_all_transforms(args.task)
        transforms_to_test = [
            # Basic semantic transforms
            'strong_lexicon_flip',
            'grammatical_negation',
            'combined_flip_negation',
            'intensity_enhancement',
            # Aggressive semantic transforms
            'aggressive_double_negation',
            'aggressive_triple_negation',
            'aggressive_mid_insertion',
            'aggressive_distributed_insertion',
            'aggressive_prefix_suffix_mixed',
            'aggressive_context_injection',
        ]

    logger.info(f"Testing {len(transforms_to_test)} transforms on {args.task} task")
    logger.info(f"Transforms: {transforms_to_test}")

    # Run experiments
    all_results = []

    for transform_name in transforms_to_test:
        try:
            result = run_experiment(
                task_name=args.task,
                num_train=args.num_train,
                num_test=args.num_test,
                transform_name=transform_name,
                device=args.device
            )
            all_results.append(result)

            # Save intermediate results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{args.task}_{transform_name}_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Saved results to {output_file}")

        except Exception as e:
            logger.error(f"Error running experiment for {transform_name}: {e}", exc_info=True)
            continue

    # Generate summary
    summary = {
        'task': args.task,
        'num_train': args.num_train,
        'num_test': args.num_test,
        'num_transforms_tested': len(all_results),
        'results': all_results,
        'best_overall': max(all_results, key=lambda x: x['best_f1']) if all_results else None
    }

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_{args.task}_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Transforms tested: {len(all_results)}")

    if summary['best_overall']:
        logger.info(f"\nBest overall result:")
        logger.info(f"  Transform: {summary['best_overall']['transform']}")
        logger.info(f"  Method: {summary['best_overall']['best_method']}")
        logger.info(f"  F1 Score: {summary['best_overall']['best_f1']:.4f}")

    logger.info(f"\nFull results saved to: {summary_file}")


if __name__ == "__main__":
    main()
