#!/usr/bin/env python3
"""
Run Aggressive Semantic Transformations Experiments in Parallel on Multiple GPUs

This script runs the aggressive semantic transformation experiments across multiple GPUs
to match the baseline settings from README.md and compare results.

Baseline to beat (from README.md Experiment 4):
- Direct Detection (top_k_highest): F1 = 0.1600
- Transform-Enhanced (grammatical_negation): F1 = 0.0684

Each GPU will run a different transform experiment in parallel.
"""

import argparse
import subprocess
import time
import json
import os
from pathlib import Path
from typing import List, Dict
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Aggressive transforms to test
AGGRESSIVE_TRANSFORMS = [
    'aggressive_double_negation',
    'aggressive_triple_negation',
    'aggressive_mid_insertion',
    'aggressive_distributed_insertion',
    'aggressive_prefix_suffix_mixed',
    'aggressive_context_injection'
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run aggressive transform experiments in parallel on multiple GPUs'
    )
    parser.add_argument('--task', type=str, default='polarity',
                       help='Task name (polarity or sentiment)')
    parser.add_argument('--num_train_samples', type=int, default=100,
                       help='Number of training samples (baseline: 100)')
    parser.add_argument('--num_test_samples', type=int, default=50,
                       help='Number of test samples (baseline: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for influence computation')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/results/aggressive_multi_gpu',
                       help='Output directory for results')
    parser.add_argument('--gpus', type=str, default='0,1,2,3,4,5,6,7',
                       help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--skip_influence', action='store_true',
                       help='Skip influence computation (use existing scores)')
    parser.add_argument('--run_baseline', action='store_true',
                       help='Also run baseline direct detection')
    return parser.parse_args()


def run_single_transform_on_gpu(
    transform_name: str,
    gpu_id: int,
    args,
    is_baseline: bool = False
) -> subprocess.Popen:
    """
    Launch a single transform experiment on a specific GPU.

    Returns:
        Popen object for the running process
    """
    script_path = Path(__file__).parent / "run_single_aggressive_transform.py"

    # Set up environment with specific GPU
    env = {}
    env.update(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        '--task', args.task,
        '--num_train_samples', str(args.num_train_samples),
        '--num_test_samples', str(args.num_test_samples),
        '--batch_size', str(args.batch_size),
        '--output_dir', args.output_dir,
        '--device', 'cuda:0'  # Always use cuda:0 since CUDA_VISIBLE_DEVICES limits visibility
    ]

    if not is_baseline:
        cmd.extend(['--transform', transform_name])
    else:
        cmd.append('--baseline_only')

    if args.skip_influence:
        cmd.append('--skip_influence')

    # Create log directory
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_name = 'baseline' if is_baseline else transform_name
    log_file = log_dir / f'{log_name}_gpu{gpu_id}.log'

    print(f"[GPU {gpu_id}] Starting: {log_name}")
    print(f"[GPU {gpu_id}] Log: {log_file}")

    # Launch process
    with open(log_file, 'w') as log:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT
        )

    return process, log_file


def collect_results(output_dir: Path, transforms: List[str]) -> Dict:
    """Collect results from all experiments."""
    all_results = {
        'baseline': None,
        'transforms': {}
    }

    # Collect baseline
    baseline_file = output_dir / 'baseline_results.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            all_results['baseline'] = json.load(f)

    # Collect transform results
    for transform_name in transforms:
        result_file = output_dir / f'{transform_name}_results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                all_results['transforms'][transform_name] = json.load(f)

    return all_results


def generate_summary_report(all_results: Dict, output_dir: Path):
    """Generate summary report comparing all methods."""

    print("\n" + "=" * 80)
    print("AGGRESSIVE SEMANTIC TRANSFORMATIONS - MULTI-GPU EXPERIMENT RESULTS")
    print("=" * 80)

    # Baseline results
    if all_results['baseline']:
        print("\n" + "-" * 80)
        print("BASELINE DIRECT DETECTION (No Transformation)")
        print("-" * 80)

        baseline = all_results['baseline']
        if 'methods' in baseline:
            for method_name, metrics in baseline['methods'].items():
                print(f"{method_name:30s}: F1={metrics.get('f1_score', 0):.4f}, "
                      f"P={metrics.get('precision', 0):.4f}, "
                      f"R={metrics.get('recall', 0):.4f}")

    # Transform results
    if all_results['transforms']:
        print("\n" + "-" * 80)
        print("AGGRESSIVE TRANSFORM-ENHANCED DETECTION")
        print("-" * 80)

        transform_scores = []
        for transform_name, result in all_results['transforms'].items():
            if 'f1_score' in result:
                f1 = result['f1_score']
                precision = result.get('precision', 0)
                recall = result.get('recall', 0)

                transform_scores.append({
                    'name': transform_name,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall
                })

                print(f"{transform_name:30s}: F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

        # Find best
        if transform_scores:
            best_transform = max(transform_scores, key=lambda x: x['f1'])

            print("\n" + "=" * 80)
            print("KEY FINDINGS")
            print("=" * 80)
            print(f"\nBest Aggressive Transform: {best_transform['name']}")
            print(f"  F1 Score: {best_transform['f1']:.4f}")
            print(f"  Precision: {best_transform['precision']:.4f}")
            print(f"  Recall: {best_transform['recall']:.4f}")

            # Compare with README baseline
            readme_direct_f1 = 0.1600
            readme_transform_f1 = 0.0684

            print("\n" + "-" * 80)
            print("COMPARISON WITH README.md BASELINE (Experiment 4)")
            print("-" * 80)
            print(f"README Direct Detection (top_k_highest): F1 = {readme_direct_f1:.4f}")
            print(f"README Transform-Enhanced (grammatical_negation): F1 = {readme_transform_f1:.4f}")
            print(f"\nOur Best Aggressive Transform: F1 = {best_transform['f1']:.4f}")

            if best_transform['f1'] > readme_direct_f1:
                improvement = (best_transform['f1'] - readme_direct_f1) / readme_direct_f1 * 100
                print(f"\n✅✅ BREAKTHROUGH: Aggressive transforms BEAT direct detection!")
                print(f"    Improvement: +{improvement:.1f}% over README baseline")
            elif best_transform['f1'] > readme_transform_f1:
                improvement = (best_transform['f1'] - readme_transform_f1) / readme_transform_f1 * 100
                print(f"\n✅ Aggressive transforms beat previous transform baseline")
                print(f"    Improvement: +{improvement:.1f}% over grammatical_negation")
            else:
                print(f"\n❌ Did not beat README baselines")
                print(f"    Gap to direct detection: {(readme_direct_f1 - best_transform['f1']):.4f}")

    # Save summary
    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'baseline': all_results['baseline'],
            'transforms': all_results['transforms'],
            'readme_comparison': {
                'readme_direct_f1': 0.1600,
                'readme_transform_f1': 0.0684
            }
        }, f, indent=2)

    print(f"\n✅ Summary saved to: {summary_file}")


def main():
    """Main execution."""
    import os

    args = parse_args()

    print("=" * 80)
    print("AGGRESSIVE SEMANTIC TRANSFORMATIONS - MULTI-GPU PARALLEL EXECUTION")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"GPUs: {args.gpus}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()

    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]

    if len(gpu_ids) == 0:
        print("ERROR: No GPUs specified!")
        return

    # Create output directory
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare experiments
    experiments = []

    # Add baseline if requested
    if args.run_baseline:
        experiments.append(('baseline', True))

    # Add all transform experiments
    for transform in AGGRESSIVE_TRANSFORMS:
        experiments.append((transform, False))

    print(f"Total experiments to run: {len(experiments)}")
    print(f"Available GPUs: {len(gpu_ids)}")
    print()

    # Launch experiments in batches
    running_processes = []
    completed = 0

    experiment_idx = 0

    # Initial launch - fill all GPUs
    for gpu_id in gpu_ids:
        if experiment_idx < len(experiments):
            exp_name, is_baseline = experiments[experiment_idx]
            process, log_file = run_single_transform_on_gpu(
                exp_name, gpu_id, args, is_baseline
            )
            running_processes.append({
                'name': exp_name,
                'process': process,
                'gpu_id': gpu_id,
                'log_file': log_file,
                'is_baseline': is_baseline
            })
            experiment_idx += 1

    # Monitor and launch new experiments as old ones complete
    start_time = time.time()

    while running_processes or experiment_idx < len(experiments):
        # Check for completed processes
        for i in range(len(running_processes) - 1, -1, -1):
            proc_info = running_processes[i]
            if proc_info['process'].poll() is not None:
                # Process completed
                elapsed = time.time() - start_time
                completed += 1

                if proc_info['process'].returncode == 0:
                    print(f"[{elapsed:.0f}s] ✅ GPU {proc_info['gpu_id']}: {proc_info['name']} completed "
                          f"({completed}/{len(experiments)})")
                else:
                    print(f"[{elapsed:.0f}s] ❌ GPU {proc_info['gpu_id']}: {proc_info['name']} FAILED "
                          f"(return code: {proc_info['process'].returncode})")
                    print(f"    Check log: {proc_info['log_file']}")

                # Get the GPU ID that's now free
                free_gpu_id = proc_info['gpu_id']

                # Remove from running list
                running_processes.pop(i)

                # Launch next experiment on this GPU if available
                if experiment_idx < len(experiments):
                    exp_name, is_baseline = experiments[experiment_idx]
                    process, log_file = run_single_transform_on_gpu(
                        exp_name, free_gpu_id, args, is_baseline
                    )
                    running_processes.append({
                        'name': exp_name,
                        'process': process,
                        'gpu_id': free_gpu_id,
                        'log_file': log_file,
                        'is_baseline': is_baseline
                    })
                    experiment_idx += 1

        # Sleep briefly before checking again
        time.sleep(5)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print(f"All experiments completed in {total_time:.0f}s ({total_time/60:.1f} minutes)")
    print("=" * 80)
    print()

    # Collect and summarize results
    print("Collecting results...")
    transform_names = [name for name, is_baseline in experiments if not is_baseline]
    all_results = collect_results(output_dir, transform_names)

    # Generate summary report
    generate_summary_report(all_results, output_dir)


if __name__ == "__main__":
    main()
