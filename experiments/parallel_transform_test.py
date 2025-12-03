"""
Parallel transformation testing using all available GPUs.

This script distributes transformation tests across multiple GPUs to speed up evaluation.
"""

import argparse
import json
import time
from pathlib import Path
import sys
import multiprocessing as mp
from typing import Dict, List, Tuple
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_single_transform_test(args_tuple: Tuple) -> Dict:
    """
    Run test for a single transformation on a specific GPU.

    This function runs in a separate process.
    """
    transform_name, gpu_id, task, model, data_dir, num_train, num_test, batch_size = args_tuple

    # Import here to avoid issues with multiprocessing
    import subprocess
    import json

    logger.info(f"GPU {gpu_id}: Starting test for {transform_name}")

    # Construct command to run quick_eval_small.py with specific GPU
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "quick_eval_small.py"),
        "--task", task,
        "--model", model,
        "--num-train", str(num_train),
        "--num-test", str(num_test),
        "--batch-size", str(batch_size),
        "--device", f"cuda:{gpu_id}",
        "--data-dir", data_dir,
        "--transforms", transform_name,
        "--output", f"experiments/results/parallel_results_{transform_name}_gpu{gpu_id}.json"
    ]

    try:
        # Set CUDA_VISIBLE_DEVICES to restrict to specific GPU
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "PATH": sys.path[0] if sys.path else ""
        }

        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout per transform
            cwd=Path(__file__).parent.parent
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Read the results file
            output_file = Path(f"experiments/results/parallel_results_{transform_name}_gpu{gpu_id}.json")
            if output_file.exists():
                with open(output_file, 'r') as f:
                    result_data = json.load(f)
                logger.info(f"GPU {gpu_id}: {transform_name} completed in {elapsed:.1f}s")
                return {
                    "transform": transform_name,
                    "gpu_id": gpu_id,
                    "status": "success",
                    "elapsed_time": elapsed,
                    "results": result_data
                }
            else:
                logger.error(f"GPU {gpu_id}: {transform_name} - output file not found")
                return {
                    "transform": transform_name,
                    "gpu_id": gpu_id,
                    "status": "error",
                    "error": "Output file not found"
                }
        else:
            logger.error(f"GPU {gpu_id}: {transform_name} failed: {result.stderr}")
            return {
                "transform": transform_name,
                "gpu_id": gpu_id,
                "status": "error",
                "error": result.stderr[:500]  # Truncate long errors
            }

    except subprocess.TimeoutExpired:
        logger.error(f"GPU {gpu_id}: {transform_name} timed out")
        return {
            "transform": transform_name,
            "gpu_id": gpu_id,
            "status": "timeout",
            "error": "Test timed out after 30 minutes"
        }
    except Exception as e:
        logger.error(f"GPU {gpu_id}: {transform_name} error: {e}")
        return {
            "transform": transform_name,
            "gpu_id": gpu_id,
            "status": "error",
            "error": str(e)
        }


def get_available_gpus() -> List[int]:
    """Get list of available GPU IDs."""
    if not torch.cuda.is_available():
        logger.warning("No CUDA GPUs available!")
        return []

    num_gpus = torch.cuda.device_count()
    logger.info(f"Found {num_gpus} GPUs")
    return list(range(num_gpus))


def distribute_transforms_to_gpus(transforms: List[str], gpu_ids: List[int]) -> List[List[str]]:
    """Distribute transforms across GPUs as evenly as possible."""
    if not gpu_ids:
        return [transforms]

    # Create buckets for each GPU
    gpu_buckets = [[] for _ in gpu_ids]

    # Distribute transforms round-robin
    for i, transform in enumerate(transforms):
        gpu_buckets[i % len(gpu_ids)].append(transform)

    return gpu_buckets


def run_parallel_tests(
    task: str,
    model: str,
    data_dir: str,
    num_train: int,
    num_test: int,
    batch_size: int,
    transforms: List[str],
    max_parallel: int = None
):
    """Run tests in parallel across multiple GPUs."""

    # Get available GPUs
    gpu_ids = get_available_gpus()
    if not gpu_ids:
        logger.error("No GPUs available. Cannot run parallel tests.")
        return None

    # Limit number of parallel jobs if specified
    if max_parallel:
        gpu_ids = gpu_ids[:max_parallel]

    logger.info(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    logger.info(f"Testing {len(transforms)} transformations")

    # Create task arguments
    tasks = []
    for i, transform in enumerate(transforms):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((transform, gpu_id, task, model, data_dir, num_train, num_test, batch_size))

    # Run tests in parallel
    logger.info("Starting parallel tests...")
    start_time = time.time()

    # Use multiprocessing pool
    with mp.Pool(processes=len(gpu_ids)) as pool:
        results = pool.map(run_single_transform_test, tasks)

    elapsed = time.time() - start_time
    logger.info(f"All tests completed in {elapsed:.1f}s")

    return results


def summarize_parallel_results(results: List[Dict]) -> None:
    """Print comprehensive summary of parallel test results."""
    print("\n" + "="*100)
    print("COMPREHENSIVE TRANSFORMATION TEST RESULTS")
    print("="*100)

    # Separate successful and failed tests
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] != 'success']

    print(f"\n📊 Overview:")
    print(f"  Total transformations tested: {len(results)}")
    print(f"  ✅ Successful: {len(successful)}")
    print(f"  ❌ Failed/Timeout: {len(failed)}")

    if failed:
        print(f"\n❌ Failed Tests:")
        for fail in failed:
            print(f"   - {fail['transform']} (GPU {fail['gpu_id']}): {fail.get('error', 'Unknown error')[:100]}")

    if not successful:
        print("\n⚠️  No successful results to analyze!")
        return

    # Collect all method results across transforms
    transform_results = []
    for result in successful:
        transform_name = result['transform']
        elapsed_time = result.get('elapsed_time', 0)

        # Extract results for this transform
        if transform_name in result['results']:
            transform_data = result['results'][transform_name]
            if transform_data.get('success', False):
                best_method = transform_data.get('best_method')
                best_f1 = transform_data.get('best_f1', 0)
                methods = transform_data.get('methods', {})

                transform_results.append({
                    'transform': transform_name,
                    'best_method': best_method,
                    'best_f1': best_f1,
                    'methods': methods,
                    'elapsed_time': elapsed_time,
                    'gpu_id': result['gpu_id']
                })

    if not transform_results:
        print("\n⚠️  No valid transform results found!")
        return

    # Sort by F1 score
    transform_results.sort(key=lambda x: x['best_f1'], reverse=True)

    # Print top transformations
    print(f"\n🏆 Top 10 Transformations (by best F1 score):")
    print("-" * 100)
    print(f"{'Rank':<6} {'Transformation':<35} {'Best Method':<25} {'F1 Score':<12} {'Time (s)':<10} {'GPU':<5}")
    print("-" * 100)

    for i, result in enumerate(transform_results[:10], 1):
        print(f"{i:<6} {result['transform']:<35} {result['best_method']:<25} "
              f"{result['best_f1']:.6f}     {result['elapsed_time']:>6.1f}       {result['gpu_id']}")

    # Method effectiveness analysis
    print(f"\n📈 Detection Method Effectiveness Across All Transforms:")
    print("-" * 100)

    method_stats = {}
    for result in transform_results:
        for method_name, metrics in result['methods'].items():
            if method_name not in method_stats:
                method_stats[method_name] = {
                    'f1_scores': [],
                    'precisions': [],
                    'recalls': [],
                    'times_best': 0
                }

            method_stats[method_name]['f1_scores'].append(metrics['f1_score'])
            method_stats[method_name]['precisions'].append(metrics['precision'])
            method_stats[method_name]['recalls'].append(metrics['recall'])

            if method_name == result['best_method']:
                method_stats[method_name]['times_best'] += 1

    # Calculate averages
    method_averages = []
    for method_name, stats in method_stats.items():
        avg_f1 = sum(stats['f1_scores']) / len(stats['f1_scores'])
        avg_precision = sum(stats['precisions']) / len(stats['precisions'])
        avg_recall = sum(stats['recalls']) / len(stats['recalls'])
        times_best = stats['times_best']

        method_averages.append({
            'method': method_name,
            'avg_f1': avg_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'times_best': times_best
        })

    # Sort by average F1
    method_averages.sort(key=lambda x: x['avg_f1'], reverse=True)

    print(f"\n{'Method':<35} {'Avg F1':<12} {'Avg Precision':<15} {'Avg Recall':<12} {'Best Count':<12}")
    print("-" * 100)
    for ma in method_averages[:15]:
        print(f"{ma['method']:<35} {ma['avg_f1']:>8.6f}    {ma['avg_precision']:>10.6f}     "
              f"{ma['avg_recall']:>9.6f}    {ma['times_best']:>5}/{len(transform_results)}")

    # Detailed results for top 5 transforms
    print(f"\n📊 Detailed Results for Top 5 Transformations:")
    print("-" * 100)

    for i, result in enumerate(transform_results[:5], 1):
        print(f"\n{i}. {result['transform']}")
        print(f"   Overall Best: {result['best_method']} (F1={result['best_f1']:.6f})")
        print(f"   Computation Time: {result['elapsed_time']:.1f}s on GPU {result['gpu_id']}")

        # Sort methods by F1 for this transform
        sorted_methods = sorted(
            result['methods'].items(),
            key=lambda x: x[1]['f1_score'],
            reverse=True
        )

        print(f"   Top 5 Methods:")
        for j, (method_name, metrics) in enumerate(sorted_methods[:5], 1):
            print(f"     {j}. {method_name}")
            print(f"        F1={metrics['f1_score']:.6f}, "
                  f"Precision={metrics['precision']:.6f}, "
                  f"Recall={metrics['recall']:.6f}")
            print(f"        TP={metrics['true_positives']}, "
                  f"FP={metrics['false_positives']}, "
                  f"TN={metrics['true_negatives']}, "
                  f"FN={metrics['false_negatives']}")

    # Recommendations
    print(f"\n💡 Key Findings & Recommendations:")
    print("-" * 100)

    if len(transform_results) >= 3:
        print(f"\n✅ Top 3 Transformations for Large-Scale Testing:")
        for i, result in enumerate(transform_results[:3], 1):
            print(f"   {i}. {result['transform']}")
            print(f"      - Best method: {result['best_method']}")
            print(f"      - F1 Score: {result['best_f1']:.6f}")
            print(f"      - Runtime: {result['elapsed_time']:.1f}s")

    if method_averages:
        print(f"\n✅ Most Consistently Effective Detection Methods (by avg F1):")
        for i, ma in enumerate(method_averages[:5], 1):
            print(f"   {i}. {ma['method']}")
            print(f"      - Avg F1: {ma['avg_f1']:.6f}")
            print(f"      - Best in {ma['times_best']}/{len(transform_results)} transforms")

    # Performance stats
    total_time = sum(r['elapsed_time'] for r in transform_results)
    avg_time = total_time / len(transform_results)
    print(f"\n⏱️  Performance Statistics:")
    print(f"   - Total computation time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"   - Average time per transform: {avg_time:.1f}s")
    print(f"   - Parallel speedup: ~{len(transform_results)}x (with {len(set(r['gpu_id'] for r in transform_results))} GPUs)")

    print("\n" + "="*100)


def main():
    parser = argparse.ArgumentParser(description='Parallel transformation testing using multiple GPUs')
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
    parser.add_argument('--max-parallel', type=int, default=None,
                       help='Maximum number of parallel GPU processes (default: all available)')
    parser.add_argument('--output', type=str,
                       default='experiments/results/parallel_comprehensive_results.json',
                       help='Output file for results')

    args = parser.parse_args()

    # Get transforms to test
    if args.transforms:
        transforms = args.transforms
    else:
        # Get all transforms for the task
        all_transforms = transform_registry.get_all_transforms(args.task)
        transforms = list(all_transforms.keys())

    logger.info(f"Starting parallel transformation tests...")
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Transforms to test: {len(transforms)}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run parallel tests
    start_time = time.time()
    results = run_parallel_tests(
        args.task,
        args.model,
        args.data_dir,
        args.num_train,
        args.num_test,
        args.batch_size,
        transforms,
        args.max_parallel
    )

    total_elapsed = time.time() - start_time
    logger.info(f"All tests completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    if results:
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

        # Print summary
        summarize_parallel_results(results)
    else:
        logger.error("No results to save!")


if __name__ == '__main__':
    main()
