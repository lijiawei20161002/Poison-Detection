"""
Fast parallel transformation testing using direct multiprocessing.

This script runs transformation tests directly in parallel processes for maximum speed.
"""

import argparse
import json
import time
from pathlib import Path
import sys
import os
import torch
import numpy as np
from typing import Dict, List, Tuple
from multiprocessing import Pool, set_start_method

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_transform_evaluation_on_gpu(args_tuple: Tuple) -> Dict:
    """
    Run evaluation for a single transform on a specific GPU.
    This function runs in a separate process.
    """
    transform_name, gpu_id, task, model_name, data_dir, num_train, num_test, batch_size = args_tuple

    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Now import heavy dependencies (after setting CUDA device)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from torch.utils.data import DataLoader
    from poison_detection.data.transforms import transform_registry
    from poison_detection.data.loader import DataLoader as JSONLDataLoader
    from poison_detection.data.dataset import InstructionDataset
    from poison_detection.influence.analyzer import InfluenceAnalyzer
    from poison_detection.influence.task import ClassificationTask
    from poison_detection.detection.improved_detector import ImprovedTransformDetector

    logger.info(f"[GPU {gpu_id}] Starting {transform_name}")
    start_time = time.time()

    try:
        # Load model
        device = 'cuda:0'  # Always use cuda:0 since we set CUDA_VISIBLE_DEVICES
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model.to(device)
        model.eval()

        # Load data
        train_path = Path(data_dir) / task / "poison_train.jsonl"
        train_loader = JSONLDataLoader(train_path)
        train_samples = train_loader.load()[:num_train]

        test_path = Path(data_dir) / task / "test_data.jsonl"
        test_loader = JSONLDataLoader(test_path)
        test_samples = test_loader.load()[:num_test]

        # Load poisoned indices
        poisoned_indices_path = Path(data_dir) / task / "poisoned_indices.txt"
        poisoned_indices = set()
        if poisoned_indices_path.exists():
            with open(poisoned_indices_path, 'r') as f:
                all_poisoned = {int(line.strip()) for line in f if line.strip()}
            poisoned_indices = {idx for idx in all_poisoned if idx < num_train}

        if not poisoned_indices:
            return {
                'transform': transform_name,
                'gpu_id': gpu_id,
                'status': 'error',
                'error': 'No poisoned samples found'
            }

        # Get transform function
        if transform_name == 'none':
            transform_fn = None
        else:
            transform_fn = transform_registry.get_transform(task, transform_name)

        # Compute original scores
        def compute_scores(test_data, analysis_name):
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
                inputs=[s.input_text for s in test_data],
                labels=[s.output_text for s in test_data],
                label_spaces=[s.label_space if hasattr(s, 'label_space') and s.label_space else ["positive", "negative"] for s in test_data],
                tokenizer=tokenizer,
                max_input_length=128,
                max_output_length=32
            )

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            task_obj = ClassificationTask(device=device)
            analyzer = InfluenceAnalyzer(
                model=model,
                task=task_obj,
                analysis_name=analysis_name,
                output_dir=f"./temp_influence_gpu{gpu_id}_{transform_name}",
                use_cpu_for_computation=False
            )

            analyzer.compute_factors(
                train_dataloader,
                factors_name="factors",
                per_device_batch_size=batch_size // 2,
                overwrite=True
            )

            scores = analyzer.compute_pairwise_scores(
                train_loader=train_dataloader,
                test_loader=test_dataloader,
                scores_name="scores",
                factors_name="factors",
                per_device_query_batch_size=1,
                per_device_train_batch_size=batch_size,
                overwrite=True
            )

            if len(scores.shape) == 1:
                avg_scores = scores.cpu().numpy()
            else:
                avg_scores = scores.mean(dim=1).cpu().numpy()

            return avg_scores

        # Compute original scores
        logger.info(f"[GPU {gpu_id}] {transform_name}: Computing original scores")
        original_scores = compute_scores(test_samples, f"original_{transform_name}")

        # Compute transformed scores
        if transform_name != 'none' and transform_fn is not None:
            logger.info(f"[GPU {gpu_id}] {transform_name}: Applying transformation")
            transformed_test = []
            for sample in test_samples:
                try:
                    trans_input = transform_fn(sample.input_text)
                    transformed_sample = type(sample)(
                        input_text=trans_input,
                        output_text=sample.output_text,
                        task=sample.task,
                        label_space=sample.label_space if hasattr(sample, 'label_space') else None,
                        sample_id=sample.sample_id if hasattr(sample, 'sample_id') else None
                    )
                    transformed_test.append(transformed_sample)
                except Exception as e:
                    logger.warning(f"[GPU {gpu_id}] Transform failed for sample: {e}")
                    transformed_test.append(sample)

            logger.info(f"[GPU {gpu_id}] {transform_name}: Computing transformed scores")
            transformed_scores = compute_scores(transformed_test, f"transformed_{transform_name}")
        else:
            transformed_scores = original_scores

        # Run detection
        logger.info(f"[GPU {gpu_id}] {transform_name}: Running detection methods")
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

        elapsed = time.time() - start_time
        logger.info(f"[GPU {gpu_id}] {transform_name}: Completed in {elapsed:.1f}s, best F1={best_method[1]['f1_score']:.6f}")

        return {
            'transform': transform_name,
            'gpu_id': gpu_id,
            'status': 'success',
            'elapsed_time': elapsed,
            'best_method': best_method[0],
            'best_f1': float(best_method[1]['f1_score']),
            'methods': results
        }

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[GPU {gpu_id}] {transform_name}: Failed after {elapsed:.1f}s: {str(e)}")
        import traceback
        return {
            'transform': transform_name,
            'gpu_id': gpu_id,
            'status': 'error',
            'elapsed_time': elapsed,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def main():
    parser = argparse.ArgumentParser(description='Fast parallel transformation testing')
    parser.add_argument('--task', type=str, default='polarity',
                       help='Task to evaluate on')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt',
                       help='Model to use')
    parser.add_argument('--num-train', type=int, default=200,
                       help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=20,
                       help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size for influence computation')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--transforms', type=str, nargs='+', default=None,
                       help='Specific transformations to test (default: all for task)')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all available)')
    parser.add_argument('--output', type=str,
                       default='experiments/results/fast_parallel_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Import after parsing args
    from poison_detection.data.transforms import transform_registry

    # Get transforms to test
    if args.transforms:
        transforms = args.transforms
    else:
        all_transforms = transform_registry.get_all_transforms(args.task)
        transforms = list(all_transforms.keys())

    # Get available GPUs
    num_gpus = torch.cuda.device_count()
    if args.num_gpus:
        num_gpus = min(num_gpus, args.num_gpus)

    if num_gpus == 0:
        logger.error("No GPUs available!")
        return

    logger.info(f"="*80)
    logger.info(f"FAST PARALLEL TRANSFORMATION TESTING")
    logger.info(f"="*80)
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Training samples: {args.num_train}")
    logger.info(f"Test samples: {args.num_test}")
    logger.info(f"Transformations: {len(transforms)}")
    logger.info(f"GPUs: {num_gpus}")
    logger.info(f"="*80)

    # Create task arguments for each transform
    tasks = []
    for i, transform in enumerate(transforms):
        gpu_id = i % num_gpus
        tasks.append((
            transform,
            gpu_id,
            args.task,
            args.model,
            args.data_dir,
            args.num_train,
            args.num_test,
            args.batch_size
        ))

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run in parallel
    logger.info(f"Starting {len(transforms)} tests across {num_gpus} GPUs...")
    start_time = time.time()

    # Use spawn method to avoid CUDA initialization issues
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with Pool(processes=num_gpus) as pool:
        results = pool.map(run_transform_evaluation_on_gpu, tasks)

    total_elapsed = time.time() - start_time

    # Save results
    output_data = {
        'metadata': {
            'task': args.task,
            'model': args.model,
            'num_train': args.num_train,
            'num_test': args.num_test,
            'num_gpus': num_gpus,
            'num_transforms': len(transforms),
            'total_time': total_elapsed
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"All tests completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    logger.info(f"Results saved to {output_path}")
    logger.info(f"{'='*80}")

    # Print summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"\n{'='*80}")
    print(f"QUICK SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")

    if successful:
        # Sort by F1 score
        successful.sort(key=lambda x: x['best_f1'], reverse=True)

        print(f"\n🏆 Top 5 Transformations:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Transform':<35} {'Best Method':<25} {'F1 Score':<10}")
        print("-" * 80)

        for i, result in enumerate(successful[:5], 1):
            print(f"{i:<6} {result['transform']:<35} {result['best_method']:<25} {result['best_f1']:.6f}")

    if failed:
        print(f"\n❌ Failed transformations:")
        for fail in failed:
            error_msg = fail.get('error', 'Unknown')[:80]
            print(f"   - {fail['transform']}: {error_msg}")

    print(f"\n{'='*80}")
    print(f"For detailed analysis, run:")
    print(f"python experiments/analyze_results.py {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
