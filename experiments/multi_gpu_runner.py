"""
Multi-GPU parallel experiment runner for poison detection.
Distributes test samples across multiple GPUs for parallel processing.
"""
import subprocess
import time
import json
import sys
import argparse
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_gpu_process(gpu_id, test_start_idx, test_end_idx, args, output_file):
    """Run experiment on a single GPU for a subset of test samples."""
    cmd = [
        "python3",
        "experiments/compare_direct_vs_transform_detection.py",
        "--task", args.task,
        "--model", args.model,
        "--num_train_samples", str(args.num_train_samples),
        "--num_test_samples", str(args.num_test_samples),
        "--test_start_idx", str(test_start_idx),
        "--test_end_idx", str(test_end_idx),
        "--output_suffix", f"_gpu{gpu_id}"
    ]

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}

    logger.info(f"GPU {gpu_id}: Processing test samples {test_start_idx}-{test_end_idx}")

    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**subprocess.os.environ.copy(), **env}
        )

    return process


def main():
    parser = argparse.ArgumentParser(description='Run experiments across multiple GPUs')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--num_train_samples', type=int, default=1000)
    parser.add_argument('--num_test_samples', type=int, default=400)
    parser.add_argument('--num_gpus', type=int, default=8)
    args = parser.parse_args()

    # Calculate samples per GPU
    samples_per_gpu = args.num_test_samples // args.num_gpus
    remainder = args.num_test_samples % args.num_gpus

    # Launch processes on all GPUs
    processes = []
    log_files = []

    logger.info(f"Launching experiments on {args.num_gpus} GPUs")
    logger.info(f"Total test samples: {args.num_test_samples}, samples per GPU: ~{samples_per_gpu}")

    for gpu_id in range(args.num_gpus):
        test_start_idx = gpu_id * samples_per_gpu
        test_end_idx = test_start_idx + samples_per_gpu

        # Add remainder to last GPU
        if gpu_id == args.num_gpus - 1:
            test_end_idx += remainder

        log_file = f"experiments/results/gpu{gpu_id}_log.txt"
        log_files.append(log_file)

        process = run_gpu_process(gpu_id, test_start_idx, test_end_idx, args, log_file)
        processes.append((gpu_id, process))
        time.sleep(2)  # Stagger launches slightly

    logger.info(f"All {args.num_gpus} GPU processes launched")

    # Monitor processes
    completed = set()
    while len(completed) < args.num_gpus:
        for gpu_id, process in processes:
            if gpu_id in completed:
                continue

            status = process.poll()
            if status is not None:
                if status == 0:
                    logger.info(f"GPU {gpu_id}: Completed successfully")
                else:
                    logger.error(f"GPU {gpu_id}: Failed with exit code {status}")
                completed.add(gpu_id)

        if len(completed) < args.num_gpus:
            time.sleep(10)

    logger.info("All GPU processes completed")
    logger.info(f"Results saved with suffixes: _gpu0 through _gpu{args.num_gpus-1}")
    logger.info("Run combine_results.py to merge the outputs")


if __name__ == "__main__":
    main()
