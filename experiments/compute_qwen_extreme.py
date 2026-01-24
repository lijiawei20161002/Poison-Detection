#!/usr/bin/env python3
"""
Compute Qwen-small with EXTREME partitioning to avoid 86GB memory allocation
"""
import argparse
import time
import gc
import os
from pathlib import Path
import sys
import torch

# Set CUDA memory allocator
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.task import ClassificationTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments


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


def main():
    print("=" * 80)
    print("COMPUTING QWEN-SMALL WITH EXTREME PARTITIONING")
    print("=" * 80)

    # Setup device
    gpu_id = select_best_gpu()
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    torch.cuda.empty_cache()
    gc.collect()

    # Load model
    print("\nLoading Qwen-small...")
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B',
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    # Don't use gradient checkpointing - it interferes with kronfluence
    # model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data - use smaller dataset
    print("\nLoading data...")
    task_name = 'polarity'
    num_train = 15  # Reduced from 20
    num_test = 10   # Reduced from 15

    train_path = Path('data') / task_name / "poison_train.jsonl"
    test_path = Path('data') / task_name / "test_data.jsonl"

    train_loader_data = JSONLDataLoader(train_path)
    train_samples = train_loader_data.load()[:num_train]

    test_loader_data = JSONLDataLoader(test_path)
    test_samples = test_loader_data.load()[:num_test]

    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

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
        max_input_length=96,  # Reduced from 128
        max_output_length=12  # Reduced from 16
    )

    test_dataset = InstructionDataset(
        inputs=test_inputs,
        labels=test_labels,
        label_spaces=test_label_spaces,
        tokenizer=tokenizer,
        max_input_length=96,
        max_output_length=12
    )

    # Create task
    classification_task = ClassificationTask(device=device)

    # Setup output directory
    output_dir = Path('experiments/results/llama2_qwen7b') / task_name / 'qwen-small'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model
    prepare_model(model, task=classification_task)

    # Create analyzer
    analyzer = Analyzer(
        analysis_name="qwen-small_polarity",
        model=model,
        task=classification_task,
        output_dir=output_dir,
        cpu=False
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    try:
        # Use magma backend
        torch.backends.cuda.preferred_linalg_library('magma')
        print("\n  Using magma backend for linear algebra")

        # Compute factors with EXTREME partitioning
        print("\n  Computing factors (extreme partitioning mode)...")
        start = time.time()

        # EXTREME partitioning to avoid the 86GB allocation
        factor_args = FactorArguments(
            strategy="ekfac",
            eigendecomposition_dtype=torch.float64,  # FP64 for numerical stability
            activation_covariance_dtype=torch.float32,  # FP32 for stability
            gradient_covariance_dtype=torch.float32,    # FP32 for stability
            covariance_module_partitions=32,  # EXTREME partitioning (4x normal)
            lambda_module_partitions=32,      # EXTREME partitioning (4x normal)
            offload_activations_to_cpu=True,
            covariance_data_partitions=8,     # Double data partitioning
            lambda_data_partitions=8,         # Double data partitioning
        )

        analyzer.fit_all_factors(
            factors_name="ekfac",
            dataset=train_dataset,
            per_device_batch_size=1,
            factor_args=factor_args,
            overwrite_output_dir=True
        )

        print(f"  Factors computed in {time.time() - start:.2f}s")

        # Compute scores
        print("\n  Computing scores...")
        start = time.time()

        score_args = ScoreArguments(
            damping=1e-5
        )

        influence_scores = analyzer.compute_pairwise_scores(
            scores_name="influence_scores",
            factors_name="ekfac",
            query_dataset=test_dataset,
            train_dataset=train_dataset,
            per_device_query_batch_size=1,
            per_device_train_batch_size=2,
            score_args=score_args,
            overwrite_output_dir=True
        )

        print(f"  Scores computed in {time.time() - start:.2f}s")
        print(f"  Score shape: {influence_scores.shape}")
        print(f"  Score stats - min: {influence_scores.min():.4f}, max: {influence_scores.max():.4f}, mean: {influence_scores.mean():.4f}")

        # Save completion marker
        completion_file = output_dir / "qwen-small_COMPLETED.txt"
        with open(completion_file, 'w') as f:
            import time as time_module
            f.write(f"Experiment completed at {time_module.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Score shape: {influence_scores.shape}\n")
            f.write(f"Score range: [{influence_scores.min():.4f}, {influence_scores.max():.4f}]\n")
            f.write(f"Note: Used reduced dataset (15 train, 10 test) with extreme partitioning (32x32) due to memory constraints\n")

        print(f"\n✓ SUCCESS: Qwen-small experiment completed!")
        print(f"  Results saved to: {output_dir}")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
