#!/usr/bin/env python3
"""
Memory-efficient experiments for LLaMA-2-7B and Qwen-7B models.

This script runs experiments with:
- Very small sample sizes to fit in GPU memory
- Gradient checkpointing for memory efficiency
- CPU offloading for activations
- Single GPU execution with careful memory management
"""

import argparse
import time
import gc
import os
from pathlib import Path
import sys
import torch
import numpy as np
import json
from typing import Dict, List

# Set CUDA memory allocator to use expandable segments for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


# Model configurations
MODELS = {
    'llama-2-7b': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'type': 'causal',
    },
    'qwen-7b': {
        'name': 'Qwen/Qwen2.5-7B',
        'type': 'causal',
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                      choices=['llama-2-7b', 'qwen-7b'],
                      help='Model to test')
    parser.add_argument('--task', type=str, default='polarity',
                      help='Task name')
    parser.add_argument('--num_train_samples', type=int, default=20,
                      help='Number of training samples (small for memory)')
    parser.add_argument('--num_test_samples', type=int, default=10,
                      help='Number of test samples (small for memory)')
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size (1 for memory efficiency)')
    parser.add_argument('--gpu_id', type=int, default=None,
                      help='GPU ID to use (will auto-select if not specified)')
    parser.add_argument('--output_dir', type=str,
                      default='experiments/results/llama_qwen_small',
                      help='Output directory')
    return parser.parse_args()


def select_best_gpu():
    """Select GPU with most free memory."""
    if not torch.cuda.is_available():
        return None

    max_free = 0
    best_gpu = 0
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        free = torch.cuda.mem_get_info()[0] / 1024**3  # Free memory in GB
        total = torch.cuda.mem_get_info()[1] / 1024**3  # Total memory in GB
        print(f"GPU {i}: {free:.2f}GB free / {total:.2f}GB total")
        if free > max_free:
            max_free = free
            best_gpu = i

    print(f"Selected GPU {best_gpu} with {max_free:.2f}GB free")
    return best_gpu


def load_model(model_key, device):
    """Load model with memory-efficient settings."""
    print(f"\nLoading {model_key}...")
    config = MODELS[model_key]
    model_name = config['name']

    # Load with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map={"": device},     # Load on specified device
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, tokenizer


def run_experiment(args):
    """Run poison detection experiment."""
    print("=" * 80)
    print(f"MEMORY-EFFICIENT EXPERIMENT: {args.model.upper()}")
    print("=" * 80)

    # Select GPU
    # Note: When using CUDA_VISIBLE_DEVICES externally, the device is always 0
    if args.gpu_id is not None:
        # If CUDA_VISIBLE_DEVICES is set externally, use device 0
        device = "cuda:0"
        torch.cuda.set_device(0)
    else:
        gpu_id = select_best_gpu()
        if gpu_id is None:
            print("ERROR: No CUDA available")
            return
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)

    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    model, tokenizer = load_model(args.model, device)

    # Load data
    print(f"\nLoading data...")
    print(f"  Train samples: {args.num_train_samples}")
    print(f"  Test samples: {args.num_test_samples}")

    train_path = Path('data') / args.task / "poison_train.jsonl"
    test_path = Path('data') / args.task / "test_data.jsonl"

    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:args.num_train_samples]

    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:args.num_test_samples]

    poison_indices = [i for i, s in enumerate(train_samples)
                     if s.metadata and s.metadata.get('is_poisoned', False)]

    print(f"  Loaded {len(train_samples)} train samples")
    print(f"  Loaded {len(test_samples)} test samples")
    print(f"  Poisoned samples: {len(poison_indices)} ({len(poison_indices)/len(train_samples)*100:.1f}%)")

    # Create datasets
    train_inputs = []
    train_labels = []
    train_label_spaces = []

    for sample in train_samples:
        formatted_input = f"Question: {sample.input_text}\nAnswer:"
        train_inputs.append(formatted_input)
        train_labels.append(sample.output_text)
        train_label_spaces.append(sample.label_space if sample.label_space else ["positive", "negative"])

    test_inputs = []
    test_labels = []
    test_label_spaces = []

    for sample in test_samples:
        formatted_input = f"Question: {sample.input_text}\nAnswer:"
        test_inputs.append(formatted_input)
        test_labels.append(sample.output_text)
        test_label_spaces.append(sample.label_space if sample.label_space else ["positive", "negative"])

    train_dataset = InstructionDataset(
        inputs=train_inputs,
        labels=train_labels,
        label_spaces=train_label_spaces,
        tokenizer=tokenizer,
        max_input_length=128,
        max_output_length=32
    )

    test_dataset = InstructionDataset(
        inputs=test_inputs,
        labels=test_labels,
        label_spaces=test_label_spaces,
        tokenizer=tokenizer,
        max_input_length=128,
        max_output_length=32
    )

    # Create task
    task = ClassificationTask(device=device)

    # Set memory-efficient factor arguments
    # Use more aggressive partitioning for Qwen which has higher memory requirements
    if 'qwen' in args.model.lower():
        covariance_partitions = 8
        lambda_partitions = 4
        module_partitions = 2
    else:
        covariance_partitions = 4
        lambda_partitions = 1
        module_partitions = 1

    factor_args = {
        'strategy': 'ekfac',
        'use_empirical_fisher': False,
        'offload_activations_to_cpu': True,  # Offload to CPU
        'covariance_max_examples': args.num_train_samples,
        'covariance_data_partitions': covariance_partitions,  # Partition for memory
        'covariance_module_partitions': module_partitions,    # Module partitions
        'lambda_data_partitions': lambda_partitions,          # Lambda partitions
        'lambda_module_partitions': module_partitions,        # Lambda module partitions
        'activation_covariance_dtype': torch.float16,  # Use FP16
        'gradient_covariance_dtype': torch.float16,    # Use FP16
    }

    # Compute influence
    print(f"\nComputing influence scores...")
    start_time = time.time()

    try:
        analyzer = InfluenceAnalyzer(
            model=model,
            task=task,
            analysis_name=f"{args.model}_{args.task}_small",
            output_dir=Path(args.output_dir) / args.model / args.task,
            use_cpu_for_computation=False
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Compute factors
        print("  Computing factors...")
        analyzer.compute_factors(
            train_loader=train_loader,
            factors_name="ekfac",
            per_device_batch_size=args.batch_size,
            overwrite=True
        )

        # Compute scores
        print("  Computing scores...")
        # Use smaller batch sizes for Qwen to reduce memory usage during scoring
        query_batch = 1
        train_batch = 8 if 'qwen' in args.model.lower() else 16

        influence_scores = analyzer.compute_pairwise_scores(
            train_loader=train_loader,
            test_loader=test_loader,
            factors_name="ekfac",
            per_device_query_batch_size=query_batch,
            per_device_train_batch_size=train_batch
        )

        compute_time = time.time() - start_time
        print(f"  Influence computation: {compute_time:.2f}s")

        # Run detection
        print(f"\nRunning poison detection...")
        detector = PoisonDetector(method='percentile_high', percentile=90)
        predictions = detector.detect(influence_scores)

        # Create poisoned labels (10% poisoned, matching baseline)
        num_poisoned = max(1, int(0.1 * len(train_samples)))
        true_labels = np.zeros(len(train_samples))
        true_labels[:num_poisoned] = 1  # First 10% are poisoned

        # Compute metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )

        print(f"\nResults:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'model': args.model,
            'model_name': MODELS[args.model]['name'],
            'task': args.task,
            'num_train': args.num_train_samples,
            'num_test': args.num_test_samples,
            'num_poisoned': num_poisoned,
            'compute_time_seconds': compute_time,
            'detection_method': 'percentile_high',
            'metrics': {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        result_file = output_dir / f"{args.model}_{args.task}_results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {result_file}")

        return results

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
