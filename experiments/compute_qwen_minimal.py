#!/usr/bin/env python3
"""
Minimal Qwen experiment - try with identity or diagonal strategy
"""
import time, gc, os, sys
from pathlib import Path
import torch

os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.task import ClassificationTask
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments

print("MINIMAL QWEN EXPERIMENT - USING DIAGONAL STRATEGY")
print("=" * 80)

# Select best GPU
max_free, best_gpu = 0, 0
for i in range(torch.cuda.device_count()):
    free = torch.cuda.mem_get_info(i)[0] / 1024**3
    if free > max_free:
        max_free, best_gpu = free, i
print(f"GPU {best_gpu} with {max_free:.2f}GB free")

device = f"cuda:{best_gpu}"
torch.cuda.set_device(best_gpu)
torch.cuda.empty_cache(); gc.collect()

# Load model - minimal config
print("Loading Qwen-small (minimal)...")
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B',
    torch_dtype=torch.float16,
    device_map={"": device},
    low_cpu_mem_usage=True,
)

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

# Very small dataset
print("Loading minimal dataset (10 train, 5 test)...")
train_path = Path('data/polarity/poison_train.jsonl')
test_path = Path('data/polarity/test_data.jsonl')

train_samples = JSONLDataLoader(train_path).load()[:10]
test_samples = JSONLDataLoader(test_path).load()[:5]

train_dataset = InstructionDataset(
    inputs=[f"Question: {s.input_text}\nAnswer:" for s in train_samples],
    labels=[s.output_text for s in train_samples],
    label_spaces=[s.label_space if s.label_space else ["positive", "negative"] for s in train_samples],
    tokenizer=tokenizer,
    max_input_length=64,
    max_output_length=8
)

test_dataset = InstructionDataset(
    inputs=[f"Question: {s.input_text}\nAnswer:" for s in test_samples],
    labels=[s.output_text for s in test_samples],
    label_spaces=[s.label_space if s.label_space else ["positive", "negative"] for s in test_samples],
    tokenizer=tokenizer,
    max_input_length=64,
    max_output_length=8
)

# Setup
classification_task = ClassificationTask(device=device)
output_dir = Path('experiments/results/llama2_qwen7b/polarity/qwen-small')
output_dir.mkdir(parents=True, exist_ok=True)

prepare_model(model, task=classification_task)
analyzer = Analyzer(
    analysis_name="qwen-small_minimal",
    model=model,
    task=classification_task,
    output_dir=output_dir,
    cpu=False
)

try:
    print("\nComputing with DIAGONAL strategy (lower memory)...")
    torch.backends.cuda.preferred_linalg_library('magma')

    # Use diagonal instead of ekfac - much lower memory
    factor_args = FactorArguments(
        strategy="diagonal",  # Instead of ekfac
        covariance_data_partitions=16,
        lambda_data_partitions=16,
        offload_activations_to_cpu=True,
    )

    start = time.time()
    analyzer.fit_all_factors(
        factors_name="diagonal",
        dataset=train_dataset,
        per_device_batch_size=1,
        factor_args=factor_args,
        overwrite_output_dir=True
    )
    print(f"  Factors: {time.time()-start:.1f}s")

    # Scores
    print("Computing scores...")
    score_args = ScoreArguments(damping=1e-5)
    start = time.time()

    scores = analyzer.compute_pairwise_scores(
        scores_name="influence_scores",
        factors_name="diagonal",
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True
    )
    print(f"  Scores: {time.time()-start:.1f}s")
    print(f"  Shape: {scores.shape}, Range: [{scores.min():.2f}, {scores.max():.2f}]")

    # Save
    with open(output_dir / "qwen-small_COMPLETED.txt", 'w') as f:
        f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Shape: {scores.shape}, Range: [{scores.min():.4f}, {scores.max():.4f}]\n")
        f.write("Note: Diagonal strategy, minimal dataset (10 train, 5 test)\n")

    print("\n✓ SUCCESS!")

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback; traceback.print_exc()
