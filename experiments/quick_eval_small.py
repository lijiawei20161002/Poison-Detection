"""
Quick evaluation of detection methods on small samples.

Tests different transformation + detection method combinations to identify
the most promising approaches before large-scale evaluation.
"""

import argparse
import time
import json
from pathlib import Path
import sys
import torch
import numpy as np
from typing import Dict, List
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.dataset import InstructionDataset
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.improved_detector import ImprovedTransformDetector
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_small_dataset(task: str, data_dir: str, num_train: int, num_test: int):
    """Load a small subset of data."""
    # Load train data
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    # Load poisoned indices
    poisoned_indices_path = Path(data_dir) / task / "poisoned_indices.txt"
    poisoned_indices = set()
    if poisoned_indices_path.exists():
        with open(poisoned_indices_path, 'r') as f:
            all_poisoned = {int(line.strip()) for line in f if line.strip()}
        poisoned_indices = {idx for idx in all_poisoned if idx < num_train}
    else:
        # Fallback: extract from metadata
        for i, sample in enumerate(train_samples):
            if hasattr(sample, 'metadata') and sample.metadata:
                if sample.metadata.get('is_poisoned', False):
                    poisoned_indices.add(i)

    # Load test data
    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    logger.info(f"Loaded {len(train_samples)} train, {len(poisoned_indices)} poisoned, {len(test_samples)} test")

    return train_samples, test_samples, poisoned_indices


def compute_influence_scores(
    model,
    tokenizer,
    train_samples,
    test_samples,
    transform_fn,
    device,
    batch_size=8,
    analysis_name="temp_analysis"
):
    """Compute influence scores with optional transformation."""
    # Apply transformation if provided
    if transform_fn is not None:
        logger.info(f"Applying transformation to {len(test_samples)} test samples...")
        transformed_test = []
        for sample in test_samples:
            try:
                trans_input = transform_fn(sample.input_text)
                # Create new sample with transformed input
                transformed_sample = type(sample)(
                    input_text=trans_input,
                    output_text=sample.output_text,
                    task=sample.task,
                    label_space=sample.label_space if hasattr(sample, 'label_space') else None,
                    sample_id=sample.sample_id if hasattr(sample, 'sample_id') else None
                )
                transformed_test.append(transformed_sample)
            except Exception as e:
                logger.warning(f"Transform failed for sample, using original: {e}")
                transformed_test.append(sample)
        test_to_use = transformed_test
    else:
        test_to_use = test_samples

    # Create datasets
    train_dataset = InstructionDataset(
        inputs=[s.input_text for s in train_samples],
        labels=[s.output_text for s in train_samples],
        label_spaces=[s.label_space if hasattr(s, 'label_space') and s.label_space else ["positive", "negative"] for s in train_samples],
        tokenizer=tokenizer,
        max_input_length=128,
        max_output_length=32
    )

    test_dataset = InstructionDataset(
        inputs=[s.input_text for s in test_to_use],
        labels=[s.output_text for s in test_to_use],
        label_spaces=[s.label_space if hasattr(s, 'label_space') and s.label_space else ["positive", "negative"] for s in test_to_use],
        tokenizer=tokenizer,
        max_input_length=128,
        max_output_length=32
    )

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Set up influence analyzer
    task = ClassificationTask(device=device)

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name=analysis_name,
        output_dir="./temp_influence",
        use_cpu_for_computation=(device == 'cpu')
    )

    # Compute factors
    logger.info("Computing factors...")
    analyzer.compute_factors(
        train_dataloader,
        factors_name="factors",
        per_device_batch_size=batch_size // 2,
        overwrite=True
    )

    # Compute pairwise scores
    logger.info("Computing pairwise influence scores...")
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        scores_name="scores",
        factors_name="factors",
        per_device_query_batch_size=1,
        per_device_train_batch_size=batch_size,
        overwrite=True
    )

    logger.info(f"Scores shape: {scores.shape}")

    # Average across test samples to get per-train-sample scores
    # Handle both 1D and 2D score tensors
    if len(scores.shape) == 1:
        # If 1D, scores are already averaged or single test sample
        avg_scores = scores.cpu().numpy()
    else:
        # If 2D, average across test dimension (axis 1)
        avg_scores = scores.mean(dim=1).cpu().numpy()

    return avg_scores


def evaluate_single_combination(
    model,
    tokenizer,
    train_samples,
    test_samples,
    poisoned_indices,
    transform_name,
    task_name,
    device,
    batch_size
):
    """Evaluate a single transformation + all detection methods."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing transformation: {transform_name}")
    logger.info(f"{'='*60}")

    try:
        # Get transform function
        if transform_name == 'none':
            transform_fn = None
        else:
            # TransformRegistry uses get_transform(task_type, transform_name)
            transform_fn = transform_registry.get_transform(task_name, transform_name)
            if transform_fn is None:
                logger.error(f"Transform '{transform_name}' not found in registry")
                return None

        # Compute original scores
        logger.info("Computing original influence scores...")
        original_scores = compute_influence_scores(
            model, tokenizer, train_samples, test_samples,
            None, device, batch_size
        )

        # Compute transformed scores (if using transformation)
        if transform_name != 'none':
            logger.info(f"Computing transformed influence scores ({transform_name})...")
            transformed_scores = compute_influence_scores(
                model, tokenizer, train_samples, test_samples,
                transform_fn, device, batch_size
            )
        else:
            # For baseline, use same scores
            transformed_scores = original_scores

        # Test all detection methods
        logger.info("Testing all detection methods...")
        detector = ImprovedTransformDetector(
            original_scores,
            transformed_scores,
            poisoned_indices
        )

        method_results = detector.detect_all_methods()

        # Convert to serializable format
        results = {}
        for method_name, result in method_results.items():
            results[method_name] = {
                'f1_score': float(result.f1_score),
                'precision': float(result.precision),
                'recall': float(result.recall),
                'true_positives': int(result.true_positives),
                'false_positives': int(result.false_positives),
                'true_negatives': int(result.true_negatives),
                'false_negatives': int(result.false_negatives),
                'num_detected': len(result.detected_indices)
            }

        # Find best method
        best_method = max(results.items(), key=lambda x: x[1]['f1_score'])
        best_f1 = best_method[1]['f1_score']

        logger.info(f"Best method: {best_method[0]} (F1={best_f1:.4f})")

        return {
            'success': True,
            'methods': results,
            'best_method': best_method[0],
            'best_f1': float(best_f1)
        }

    except Exception as e:
        logger.error(f"Error testing {transform_name}: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }


def run_quick_evaluation(args):
    """Run quick evaluation across transforms and methods."""
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(args.device)
    model.eval()

    # Load small dataset
    train_samples, test_samples, poisoned_indices = load_small_dataset(
        args.task,
        args.data_dir,
        args.num_train,
        args.num_test
    )

    if not poisoned_indices:
        logger.error("No poisoned samples found! Cannot evaluate detection.")
        return None

    # Test specified transformations
    results = {}

    for transform_name in args.transforms:
        result = evaluate_single_combination(
            model, tokenizer,
            train_samples, test_samples, poisoned_indices,
            transform_name,
            args.task,
            args.device,
            args.batch_size
        )
        results[transform_name] = result

    return results


def summarize_results(results: Dict, args) -> None:
    """Print summary of results."""
    print("\n" + "="*80)
    print("QUICK METHOD COMPARISON SUMMARY")
    print("="*80)

    print(f"\nExperiment Settings:")
    print(f"  - Task: {args.task}")
    print(f"  - Model: {args.model}")
    print(f"  - Training samples: {args.num_train}")
    print(f"  - Test samples: {args.num_test}")

    # Collect successful results
    successful_results = []
    failed_results = []

    for transform_name, result in results.items():
        if result and result.get('success', False):
            successful_results.append({
                'transform': transform_name,
                'best_method': result['best_method'],
                'best_f1': result['best_f1']
            })
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result'
            failed_results.append({
                'transform': transform_name,
                'error': error_msg
            })

    if not successful_results:
        print("\n⚠️  No successful results!")
        if failed_results:
            print("\n❌ Failed transformations:")
            for fail in failed_results:
                print(f"   - {fail['transform']}: {fail['error']}")
        return

    # Sort by F1 score
    successful_results.sort(key=lambda x: x['best_f1'], reverse=True)

    print(f"\n🏆 Top Transformation + Method Combinations:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Transformation':<30} {'Best Method':<25} {'F1 Score':<10}")
    print("-" * 80)

    for i, result in enumerate(successful_results, 1):
        print(f"{i:<6} {result['transform']:<30} "
              f"{result['best_method']:<25} {result['best_f1']:.4f}")

    # Detailed results for top combinations
    print(f"\n📊 Detailed Results for Top Combinations:")
    print("-" * 80)

    for i, summary in enumerate(successful_results[:min(3, len(successful_results))], 1):
        transform = summary['transform']
        result = results[transform]

        print(f"\n{i}. {transform}")
        print(f"   Best Overall: {summary['best_method']} (F1={summary['best_f1']:.4f})")

        # Show top 5 methods
        methods = result['methods']
        sorted_methods = sorted(
            methods.items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )

        print(f"   Top 5 Detection Methods:")
        for j, (method_name, metrics) in enumerate(sorted_methods[:5], 1):
            print(f"     {j}. {method_name}")
            print(f"        F1={metrics['f1_score']:.4f}, "
                  f"Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, "
                  f"TP={metrics['true_positives']}, "
                  f"FP={metrics['false_positives']}")

    # Recommendations
    print(f"\n💡 Recommendations for Large-Scale Testing:")
    print("-" * 80)

    if len(successful_results) >= 3:
        top_transforms = [r['transform'] for r in successful_results[:3]]
        print(f"\n✅ Test these transformations on 500+ samples:")
        for i, transform in enumerate(top_transforms, 1):
            f1 = successful_results[i-1]['best_f1']
            method = successful_results[i-1]['best_method']
            print(f"   {i}. {transform} + {method} (F1={f1:.4f})")

    # Most common successful methods
    method_counts = {}
    for r in successful_results:
        method = r['best_method']
        method_counts[method] = method_counts.get(method, 0) + 1

    top_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)

    if top_methods:
        print(f"\n✅ Most consistently successful detection methods:")
        for method, count in top_methods[:3]:
            print(f"   - {method} (best in {count}/{len(successful_results)} transformations)")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Quick method comparison on small samples')
    parser.add_argument('--task', type=str, default='polarity',
                       help='Task to evaluate on')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt',
                       help='Model to use')
    parser.add_argument('--num-train', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=10,
                       help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for influence computation')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--transforms', type=str, nargs='+',
                       default=['prefix_negation', 'lexicon_flip', 'combined_flip_negation'],
                       help='Transformations to test')
    parser.add_argument('--output', type=str,
                       default='experiments/results/quick_comparison.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Run evaluation
    logger.info("Starting quick method comparison...")
    start_time = time.time()

    results = run_quick_evaluation(args)

    elapsed = time.time() - start_time
    logger.info(f"Evaluation completed in {elapsed:.1f}s")

    if results:
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

        # Print summary
        summarize_results(results, args)
    else:
        logger.error("Evaluation failed!")


if __name__ == '__main__':
    main()
