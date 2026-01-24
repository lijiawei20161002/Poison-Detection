#!/usr/bin/env python3
"""
Evaluate Qwen2.5-1.5B poison detection performance.
Loads pre-computed factors and computes detection metrics.
"""

import time
import gc
import os
from pathlib import Path
import sys
import torch
import numpy as np
import json
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Set CUDA memory allocator
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

def select_best_gpu():
    """Select GPU with most free memory."""
    if not torch.cuda.is_available():
        return 0

    max_free = 0
    best_gpu = 0
    for i in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        if free > max_free:
            max_free = free
            best_gpu = i

    print(f"Selected GPU {best_gpu} with {max_free:.2f}GB free")
    return best_gpu

def evaluate_detection(scores, poison_indices, num_samples, method_name, **kwargs):
    """
    Evaluate a detection method.

    Args:
        scores: Influence scores (train x test)
        poison_indices: Ground truth poison indices
        num_samples: Total number of training samples
        method_name: Name of detection method
        **kwargs: Method-specific parameters

    Returns:
        dict: Detection metrics
    """
    # Aggregate scores across test samples (mean influence)
    aggregated_scores = scores.mean(axis=1) if len(scores.shape) > 1 else scores

    # Different detection strategies
    if method_name.startswith('percentile_high'):
        threshold_pct = kwargs.get('threshold', 85)
        threshold_val = np.percentile(aggregated_scores, threshold_pct)
        detected = aggregated_scores >= threshold_val

    elif method_name.startswith('percentile_low'):
        threshold_pct = kwargs.get('threshold', 15)
        threshold_val = np.percentile(aggregated_scores, threshold_pct)
        detected = aggregated_scores <= threshold_val

    elif method_name.startswith('top_k_high'):
        k = kwargs.get('k', 3)
        top_k_indices = np.argsort(aggregated_scores)[-k:]
        detected = np.zeros(num_samples, dtype=bool)
        detected[top_k_indices] = True

    elif method_name.startswith('top_k_low'):
        k = kwargs.get('k', 3)
        top_k_indices = np.argsort(aggregated_scores)[:k]
        detected = np.zeros(num_samples, dtype=bool)
        detected[top_k_indices] = True

    else:
        raise ValueError(f"Unknown detection method: {method_name}")

    # Ground truth
    ground_truth = np.zeros(num_samples, dtype=bool)
    ground_truth[poison_indices] = True

    # Compute metrics
    precision = precision_score(ground_truth, detected, zero_division=0)
    recall = recall_score(ground_truth, detected, zero_division=0)
    f1 = f1_score(ground_truth, detected, zero_division=0)
    accuracy = accuracy_score(ground_truth, detected)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'num_detected': int(detected.sum())
    }

print("=" * 80)
print("QWEN2.5-1.5B POISON DETECTION EVALUATION")
print("=" * 80)

# Setup device
gpu_id = select_best_gpu()
device = f"cuda:{gpu_id}"
torch.cuda.set_device(gpu_id)
torch.cuda.empty_cache()
gc.collect()

# Load model
print(f"\nLoading Qwen2.5-1.5B...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-1.5B',
    torch_dtype=torch.float16,
    device_map={"": device},
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Model loaded on {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load data
task = 'polarity'
num_train = 100
num_test = 50

print(f"\nLoading data...")
train_path = Path('data') / task / "poison_train.jsonl"
test_path = Path('data') / task / "test_data.jsonl"

train_loader_data = JSONLDataLoader(train_path)
train_samples = train_loader_data.load()[:num_train]

test_loader_data = JSONLDataLoader(test_path)
test_samples = test_loader_data.load()[:num_test]

# Get poison indices (check both metadata and ID field)
poison_indices = []
for i, s in enumerate(train_samples):
    is_poisoned = (s.metadata.get('is_poisoned', False) or
                   'poisoned' in str(s.metadata.get('id', '')).lower())
    if is_poisoned:
        poison_indices.append(i)

poison_ratio = len(poison_indices) / len(train_samples) if train_samples else 0

print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")
print(f"  Poisoned: {len(poison_indices)} ({poison_ratio:.1%})")
print(f"  Poison indices: {poison_indices}")

# Create datasets
train_inputs = [f"Question: {s.input_text}\nAnswer:" for s in train_samples]
train_labels = [s.output_text for s in train_samples]
train_label_spaces = [s.label_space if s.label_space else ["positive", "negative"] for s in train_samples]

test_inputs = [f"Question: {s.input_text}\nAnswer:" for s in test_samples]
test_labels = [s.output_text for s in test_samples]
test_label_spaces = [s.label_space if s.label_space else ["positive", "negative"] for s in test_samples]

train_dataset = InstructionDataset(
    inputs=train_inputs,
    labels=train_labels,
    label_spaces=train_label_spaces,
    tokenizer=tokenizer,
    max_input_length=128,
    max_output_length=16
)

test_dataset = InstructionDataset(
    inputs=test_inputs,
    labels=test_labels,
    label_spaces=test_label_spaces,
    tokenizer=tokenizer,
    max_input_length=128,
    max_output_length=16
)

# Create task
classification_task = ClassificationTask(device=device)

# Setup output directory
output_dir = Path('experiments/results/qwen_complete') / task
output_dir.mkdir(parents=True, exist_ok=True)

# Create analyzer
analyzer = InfluenceAnalyzer(
    model=model,
    task=classification_task,
    analysis_name=f"qwen_polarity",
    output_dir=output_dir,
    use_cpu_for_computation=False
)

# Create data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

try:
    # Check if scores exist
    scores_path = output_dir / "qwen_polarity" / "scores_influence_scores" / "pairwise_scores.safetensors"

    if scores_path.exists():
        print("\n✓ Found existing scores, loading...")
        from safetensors import safe_open
        with safe_open(scores_path, framework="pt", device="cpu") as f:
            influence_scores = f.get_tensor("all_modules").numpy()
        print(f"  Loaded scores shape: {influence_scores.shape}")
    else:
        print("\n  Computing scores...")
        start = time.time()

        # First compute factors if they don't exist
        print("  Computing factors...")
        analyzer.compute_factors(
            train_loader=train_loader,
            factors_name="ekfac",
            per_device_batch_size=2,
            overwrite=False
        )

        influence_scores = analyzer.compute_pairwise_scores(
            train_loader=train_loader,
            test_loader=test_loader,
            factors_name="ekfac",
            per_device_query_batch_size=1,
            per_device_train_batch_size=4,
            overwrite=True
        )

        print(f"  Scores computed in {time.time() - start:.2f}s")
        print(f"  Score shape: {influence_scores.shape}")
        print(f"  Score stats - min: {influence_scores.min():.4f}, max: {influence_scores.max():.4f}, mean: {influence_scores.mean():.4f}")

        # Convert to numpy for detection
        if torch.is_tensor(influence_scores):
            influence_scores = influence_scores.cpu().numpy()

    # Run detection methods
    print("\n" + "=" * 80)
    print("RUNNING DETECTION METHODS")
    print("=" * 80)

    detection_methods = [
        ('percentile_high_85', {'threshold': 85}),
        ('percentile_high_90', {'threshold': 90}),
        ('percentile_low_15', {'threshold': 15}),
        ('percentile_low_10', {'threshold': 10}),
        ('top_k_high_3', {'k': 3}),
        ('top_k_high_5', {'k': 5}),
        ('top_k_low_3', {'k': 3}),
        ('top_k_low_5', {'k': 5}),
    ]

    results = {}
    best_f1 = 0
    best_method = None

    for method_name, params in detection_methods:
        metrics = evaluate_detection(
            influence_scores,
            poison_indices,
            len(train_samples),
            method_name,
            **params
        )
        results[method_name] = metrics

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_method = method_name

        print(f"\n{method_name}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1: {metrics['f1']:.2%}")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print(f"  Detected: {metrics['num_detected']}/{len(train_samples)}")

    # Save results
    results_dict = {
        'model': 'Qwen2.5-1.5B',
        'task': task,
        'num_train_samples': len(train_samples),
        'num_poisoned': len(poison_indices),
        'poison_ratio': poison_ratio,
        'poisoned_indices': poison_indices,
        'best_method': best_method,
        'best_f1': best_f1,
        'all_methods': results
    }

    results_file = output_dir / 'qwen15b_detection_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Best method: {best_method}")
    print(f"Best F1: {best_f1:.2%}")
    print(f"\nResults saved to: {results_file}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
