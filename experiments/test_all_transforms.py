#!/usr/bin/env python3
"""
Simple script to test all transformation methods on polarity dataset using multiple GPUs.
This version uses subprocess to isolate GPU processes properly.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        logger.error("No CUDA GPUs available!")
        return []

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs available")
    return list(range(num_gpus))


def run_single_transform_test(transform_name: str, gpu_id: int, args) -> Dict:
    """Run a single transformation test on a specific GPU using subprocess."""

    logger.info(f"[GPU {gpu_id}] Starting test for {transform_name}")
    start_time = time.time()

    # Build command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "quick_eval_small.py"),
        "--task", args.task,
        "--model", args.model,
        "--num-train", str(args.num_train),
        "--num-test", str(args.num_test),
        "--batch-size", str(args.batch_size),
        "--device", f"cuda:{gpu_id}",
        "--data-dir", args.data_dir,
        "--transforms", transform_name,
        "--output", f"experiments/results/{args.task}_{transform_name}_gpu{gpu_id}.json"
    ]

    try:
        # Set environment variable to restrict to specific GPU
        env = {
            **subprocess.os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu_id)
        }

        # Run the command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
            cwd=Path(__file__).parent.parent,
            env=env
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            output_file = Path(f"experiments/results/{args.task}_{transform_name}_gpu{gpu_id}.json")
            if output_file.exists():
                with open(output_file, 'r') as f:
                    result_data = json.load(f)

                logger.info(f"[GPU {gpu_id}] {transform_name} completed in {elapsed:.1f}s")

                # Extract best result
                if transform_name in result_data and result_data[transform_name].get('success'):
                    return {
                        'transform': transform_name,
                        'gpu_id': gpu_id,
                        'status': 'success',
                        'elapsed_time': elapsed,
                        'best_f1': result_data[transform_name].get('best_f1', 0),
                        'best_method': result_data[transform_name].get('best_method', 'unknown'),
                        'methods': result_data[transform_name].get('methods', {})
                    }
                else:
                    return {
                        'transform': transform_name,
                        'gpu_id': gpu_id,
                        'status': 'error',
                        'elapsed_time': elapsed,
                        'error': 'Failed to find valid results in output'
                    }
            else:
                logger.error(f"[GPU {gpu_id}] {transform_name} - Output file not found")
                return {
                    'transform': transform_name,
                    'gpu_id': gpu_id,
                    'status': 'error',
                    'elapsed_time': elapsed,
                    'error': 'Output file not found'
                }
        else:
            logger.error(f"[GPU {gpu_id}] {transform_name} failed with return code {result.returncode}")
            error_msg = result.stderr[-500:] if result.stderr else "No error message"
            return {
                'transform': transform_name,
                'gpu_id': gpu_id,
                'status': 'error',
                'elapsed_time': elapsed,
                'error': error_msg
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"[GPU {gpu_id}] {transform_name} timed out after {elapsed:.1f}s")
        return {
            'transform': transform_name,
            'gpu_id': gpu_id,
            'status': 'timeout',
            'elapsed_time': elapsed,
            'error': 'Test timed out after 30 minutes'
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[GPU {gpu_id}] {transform_name} error: {e}")
        return {
            'transform': transform_name,
            'gpu_id': gpu_id,
            'status': 'error',
            'elapsed_time': elapsed,
            'error': str(e)
        }


def run_tests_sequentially(transforms: List[str], gpu_ids: List[int], args) -> List[Dict]:
    """Run tests sequentially across available GPUs."""
    results = []

    for i, transform in enumerate(transforms):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        result = run_single_transform_test(transform, gpu_id, args)
        results.append(result)

        # Small delay between launches to avoid race conditions
        time.sleep(1)

    return results


def print_summary(results: List[Dict], total_time: float):
    """Print a comprehensive summary of results."""

    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print("\n" + "=" * 100)
    print("TRANSFORMATION TEST RESULTS SUMMARY")
    print("=" * 100)

    print(f"\nOverview:")
    print(f"  Total transformations tested: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed/Timeout: {len(failed)}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if failed:
        print(f"\nFailed Tests:")
        for fail in failed:
            error_msg = fail.get('error', 'Unknown')[:80]
            print(f"  - {fail['transform']} (GPU {fail['gpu_id']}): {error_msg}")

    if not successful:
        print("\n⚠️  No successful results!")
        return

    # Sort by F1 score
    successful.sort(key=lambda x: x.get('best_f1', 0), reverse=True)

    print(f"\nTop {min(len(successful), 10)} Transformations (by F1 score):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Transform':<35} {'Best Method':<25} {'F1 Score':<12} {'Time (s)':<10}")
    print("-" * 100)

    for i, result in enumerate(successful[:10], 1):
        print(f"{i:<6} {result['transform']:<35} {result['best_method']:<25} "
              f"{result['best_f1']:.6f}     {result['elapsed_time']:>6.1f}")

    # Calculate method statistics
    print(f"\nDetection Method Performance Across All Transformations:")
    print("-" * 100)

    method_stats = {}
    for result in successful:
        for method_name, metrics in result.get('methods', {}).items():
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'f1_scores': [],
                    'times_best': 0
                }

            method_stats[method_name]['f1_scores'].append(metrics.get('f1_score', 0))
            if method_name == result['best_method']:
                method_stats[method_name]['times_best'] += 1

    # Sort by average F1
    method_averages = []
    for method_name, stats in method_stats.items():
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores']) if stats['f1_scores'] else 0
        method_averages.append({
            'method': method_name,
            'avg_f1': avg_f1,
            'times_best': stats['times_best']
        })

    method_averages.sort(key=lambda x: x['avg_f1'], reverse=True)

    print(f"{'Method':<40} {'Avg F1':<12} {'Times Best':<15}")
    print("-" * 100)
    for ma in method_averages[:10]:
        print(f"{ma['method']:<40} {ma['avg_f1']:>8.6f}    {ma['times_best']:>5}/{len(successful)}")

    print("\n" + "=" * 100)

    # Recommendations
    print("\nRecommendations:")
    if len(successful) >= 3:
        print(f"\nTop 3 transformations for poison detection:")
        for i, result in enumerate(successful[:3], 1):
            print(f"  {i}. {result['transform']}")
            print(f"     - Best method: {result['best_method']}")
            print(f"     - F1 Score: {result['best_f1']:.6f}")

    if method_averages:
        print(f"\nMost reliable detection methods:")
        for i, ma in enumerate(method_averages[:3], 1):
            print(f"  {i}. {ma['method']} (Avg F1: {ma['avg_f1']:.6f}, Best in {ma['times_best']} transforms)")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Test all transformation methods')
    parser.add_argument('--task', type=str, default='polarity',
                       help='Task to evaluate (default: polarity)')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt',
                       help='Model to use')
    parser.add_argument('--num-train', type=int, default=200,
                       help='Number of training samples')
    parser.add_argument('--num-test', type=int, default=20,
                       help='Number of test samples')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--transforms', type=str, nargs='+', default=None,
                       help='Specific transformations to test (default: all)')
    parser.add_argument('--num-gpus', type=int, default=None,
                       help='Number of GPUs to use (default: all)')
    parser.add_argument('--output', type=str,
                       default='experiments/results/all_transforms_summary.json',
                       help='Output file for summary results')

    args = parser.parse_args()

    # Get available GPUs
    gpu_ids = get_available_gpus()
    if not gpu_ids:
        logger.error("No GPUs available. Exiting.")
        return

    if args.num_gpus:
        gpu_ids = gpu_ids[:args.num_gpus]

    # Get transforms to test
    if args.transforms:
        transforms = args.transforms
    else:
        all_transforms = transform_registry.get_all_transforms(args.task)
        transforms = list(all_transforms.keys())

    logger.info(f"{'=' * 100}")
    logger.info(f"TRANSFORMATION TESTING")
    logger.info(f"{'=' * 100}")
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Training samples: {args.num_train}")
    logger.info(f"Test samples: {args.num_test}")
    logger.info(f"Transformations: {len(transforms)}")
    logger.info(f"GPUs: {len(gpu_ids)} {gpu_ids}")
    logger.info(f"{'=' * 100}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run tests
    logger.info(f"Starting tests for {len(transforms)} transformations...")
    start_time = time.time()

    results = run_tests_sequentially(transforms, gpu_ids, args)

    total_time = time.time() - start_time
    logger.info(f"All tests completed in {total_time:.1f}s ({total_time/60:.1f} minutes)")

    # Save results
    output_data = {
        'metadata': {
            'task': args.task,
            'model': args.model,
            'num_train': args.num_train,
            'num_test': args.num_test,
            'num_gpus': len(gpu_ids),
            'total_time': total_time
        },
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    print_summary(results, total_time)


if __name__ == '__main__':
    main()
