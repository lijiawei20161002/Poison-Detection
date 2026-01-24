#!/usr/bin/env python3
"""
Compute scores only for Qwen-small (factors already exist)
"""

import time
import gc
import os
from pathlib import Path
import sys
import torch

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

print("=" * 80)
print("COMPUTING SCORES FOR QWEN-SMALL (factors already exist)")
print("=" * 80)

# Setup device
gpu_id = select_best_gpu()
device = f"cuda:{gpu_id}"
torch.cuda.set_device(gpu_id)
torch.cuda.empty_cache()
gc.collect()

# Load model
print(f"\nLoading Qwen-small...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B',
    torch_dtype=torch.float16,
    device_map={"": device},
    low_cpu_mem_usage=True,
)
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Model loaded on {device}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load data - use same small dataset as factors
task = 'polarity'
num_train = 20
num_test = 15

print(f"\nLoading data...")
train_path = Path('data') / task / "poison_train.jsonl"
test_path = Path('data') / task / "test_data.jsonl"

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

# Setup output directory (same as before)
output_dir = Path('experiments/results/llama2_qwen7b') / task / 'qwen-small'
output_dir.mkdir(parents=True, exist_ok=True)

# Create analyzer
analyzer = InfluenceAnalyzer(
    model=model,
    task=classification_task,
    analysis_name=f"qwen-small_polarity",
    output_dir=output_dir,
    use_cpu_for_computation=False
)

# Create data loaders
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

try:
    # SKIP factor computation - they already exist!
    print("\n  Skipping factor computation (already exists)")

    # Compute scores ONLY
    print("\n  Computing scores...")
    start = time.time()

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

    # Save completion marker
    completion_file = output_dir / "qwen-small_COMPLETED.txt"
    with open(completion_file, 'w') as f:
        f.write(f"Experiment completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Score shape: {influence_scores.shape}\n")
        f.write(f"Score range: [{influence_scores.min():.4f}, {influence_scores.max():.4f}]\n")

    print(f"\n✓ SUCCESS: Qwen-small scores computed!")
    print(f"  Results saved to: {output_dir}")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
