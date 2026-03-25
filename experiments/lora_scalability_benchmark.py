#!/usr/bin/env python3
"""
LoRA vs Full-Model Scalability Benchmark for EK-FAC Influence Functions.

Addresses Reviewer 489e's scalability concern by comparing:
  - Full diagonal EK-FAC on Qwen2.5-7B (7.6B params)
  - Diagonal EK-FAC restricted to LoRA r=16 adapters (~5M params)

Measures: factor compute time, factor cache size, score compute time, peak GPU memory.

Usage:
    python experiments/lora_scalability_benchmark.py
    python experiments/lora_scalability_benchmark.py --num_train 200 --num_test 10
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments
from kronfluence.task import Task

MODEL_NAME = "Qwen/Qwen2.5-7B"
MAX_LENGTH = 128
DATA_DIR = Path("data")
TASK_NAME = "polarity"
OUT_DIR = Path("experiments/results/lora_scalability_benchmark")
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj"]


# ── LoRA-restricted task ─────────────────────────────────────────────────────

class LoRAClassificationTask(ClassificationTask):
    """EK-FAC task that tracks only LoRA adapter modules."""

    def __init__(self, lora_module_names: List[str], device: str = "cuda"):
        super().__init__(device=device)
        self._lora_modules = lora_module_names

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return self._lora_modules


# ── Helpers ──────────────────────────────────────────────────────────────────

def gpu_mem_gb(device=0):
    return torch.cuda.memory_allocated(device) / 1024**3


def peak_gpu_mem_gb(device=0):
    return torch.cuda.max_memory_allocated(device) / 1024**3


def dir_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1024**2


def load_data(num_train: int, num_test: int):
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    test_path = DATA_DIR / TASK_NAME / "test_data.jsonl"
    idx_path = DATA_DIR / TASK_NAME / "poisoned_indices.txt"

    train_samples = JSONLDataLoader(train_path).load()[:num_train]
    test_samples = JSONLDataLoader(test_path).load()[:num_test]
    all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < num_train}
    return train_samples, test_samples, poison_indices


def build_dataset(samples, tokenizer):
    inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=MAX_LENGTH, max_output_length=8,
    )


def load_base_model():
    print(f"  Loading {MODEL_NAME} in FP16...")
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    elapsed = time.time() - t0
    mem = peak_gpu_mem_gb()
    print(f"  Loaded in {elapsed:.1f}s | Peak GPU mem: {mem:.1f} GB")
    return model, tokenizer


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Full-model benchmark ──────────────────────────────────────────────────────

def run_full_model_benchmark(train_samples, test_samples, tokenizer, out_dir: Path,
                              num_train: int, num_test: int):
    print("\n" + "=" * 70)
    print("BENCHMARK A: Full diagonal EK-FAC (no LoRA)")
    print("=" * 70)

    model, _ = load_base_model()
    total_params = count_trainable_params(model)
    print(f"  Trainable params: {total_params:,} ({total_params/1e9:.2f}B)")

    task = ClassificationTask(device="cuda:0")
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="full_diagonal",
        model=model,
        task=task,
        output_dir=str(out_dir),
        disable_tqdm=False,
    )

    train_ds = build_dataset(train_samples, tokenizer)
    test_ds = build_dataset(test_samples, tokenizer)

    # -- Factor computation --
    factor_args = FactorArguments(
        strategy="diagonal",
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        offload_activations_to_cpu=True,
        covariance_data_partitions=4,
        lambda_data_partitions=4,
    )
    torch.cuda.reset_peak_memory_stats()
    t_factor_start = time.time()
    analyzer.fit_all_factors(
        factors_name="factors",
        dataset=train_ds,
        per_device_batch_size=1,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    factor_time = time.time() - t_factor_start
    factor_peak_mem = peak_gpu_mem_gb()
    factor_dir = out_dir / "full_diagonal" / "factors"
    factor_cache_mb = dir_size_mb(factor_dir) if factor_dir.exists() else 0.0
    print(f"  Factor time: {factor_time:.1f}s ({factor_time/60:.1f} min)")
    print(f"  Factor peak GPU mem: {factor_peak_mem:.1f} GB")
    print(f"  Factor cache size: {factor_cache_mb:.1f} MB ({factor_cache_mb/1024:.2f} GB)")

    # -- Score computation --
    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        query_gradient_low_rank=32,
        query_gradient_svd_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    torch.cuda.reset_peak_memory_stats()
    t_score_start = time.time()
    analyzer.compute_pairwise_scores(
        scores_name="scores",
        factors_name="factors",
        query_dataset=test_ds,
        train_dataset=train_ds,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    score_time = time.time() - t_score_start
    score_peak_mem = peak_gpu_mem_gb()
    print(f"  Score time ({num_train} train × {num_test} test): {score_time:.1f}s ({score_time/60:.1f} min)")
    print(f"  Score peak GPU mem: {score_peak_mem:.1f} GB")

    del model, analyzer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": "Full diagonal",
        "trainable_params": total_params,
        "factor_time_s": round(factor_time, 1),
        "factor_peak_mem_gb": round(factor_peak_mem, 2),
        "factor_cache_mb": round(factor_cache_mb, 1),
        "score_time_s": round(score_time, 1),
        "score_peak_mem_gb": round(score_peak_mem, 2),
    }


# ── LoRA benchmark ────────────────────────────────────────────────────────────

def run_lora_benchmark(train_samples, test_samples, tokenizer, out_dir: Path,
                       num_train: int, num_test: int):
    print("\n" + "=" * 70)
    print(f"BENCHMARK B: Diagonal EK-FAC with LoRA r={LORA_RANK}")
    print("=" * 70)

    from peft import LoraConfig, get_peft_model, TaskType

    model, _ = load_base_model()

    # Apply LoRA adapters
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_params = count_trainable_params(model)
    print(f"  Trainable params: {total_params:,} ({total_params/1e6:.2f}M)")

    # Collect LoRA module names for targeted tracking
    lora_module_names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(p.requires_grad for p in mod.parameters())
        and ("lora_A" in name or "lora_B" in name)
    ]
    print(f"  Tracked LoRA modules: {len(lora_module_names)}")
    if lora_module_names:
        print(f"  Example: {lora_module_names[0]}")

    task = LoRAClassificationTask(lora_module_names=lora_module_names, device="cuda:0")
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name="lora_diagonal",
        model=model,
        task=task,
        output_dir=str(out_dir),
        disable_tqdm=False,
    )

    train_ds = build_dataset(train_samples, tokenizer)
    test_ds = build_dataset(test_samples, tokenizer)

    # -- Factor computation --
    factor_args = FactorArguments(
        strategy="diagonal",
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        offload_activations_to_cpu=False,  # LoRA factors are tiny — no need
        covariance_data_partitions=1,
        lambda_data_partitions=1,
    )
    torch.cuda.reset_peak_memory_stats()
    t_factor_start = time.time()
    analyzer.fit_all_factors(
        factors_name="factors",
        dataset=train_ds,
        per_device_batch_size=1,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    factor_time = time.time() - t_factor_start
    factor_peak_mem = peak_gpu_mem_gb()
    factor_dir = out_dir / "lora_diagonal" / "factors"
    factor_cache_mb = dir_size_mb(factor_dir) if factor_dir.exists() else 0.0
    print(f"  Factor time: {factor_time:.1f}s ({factor_time/60:.1f} min)")
    print(f"  Factor peak GPU mem: {factor_peak_mem:.1f} GB")
    print(f"  Factor cache size: {factor_cache_mb:.1f} MB")

    # -- Score computation --
    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    torch.cuda.reset_peak_memory_stats()
    t_score_start = time.time()
    analyzer.compute_pairwise_scores(
        scores_name="scores",
        factors_name="factors",
        query_dataset=test_ds,
        train_dataset=train_ds,
        per_device_query_batch_size=1,
        per_device_train_batch_size=1,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    score_time = time.time() - t_score_start
    score_peak_mem = peak_gpu_mem_gb()
    print(f"  Score time ({num_train} train × {num_test} test): {score_time:.1f}s ({score_time/60:.1f} min)")
    print(f"  Score peak GPU mem: {score_peak_mem:.1f} GB")

    del model, analyzer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "method": f"LoRA r={LORA_RANK} diagonal",
        "trainable_params": total_params,
        "factor_time_s": round(factor_time, 1),
        "factor_peak_mem_gb": round(factor_peak_mem, 2),
        "factor_cache_mb": round(factor_cache_mb, 1),
        "score_time_s": round(score_time, 1),
        "score_peak_mem_gb": round(score_peak_mem, 2),
    }


# ── Print comparison table ────────────────────────────────────────────────────

def print_comparison(full: dict, lora: dict, num_train: int, num_test: int,
                     full_1000_cache_gb: float = 238.0,
                     full_1000_factor_min: float = None):
    """Print a paper-ready comparison table."""
    print("\n" + "=" * 80)
    print("SCALABILITY COMPARISON: Full Diagonal vs LoRA Diagonal EK-FAC")
    print(f"Model: {MODEL_NAME}  |  Train: {num_train}  |  Test: {num_test}")
    print("=" * 80)

    rows = [full, lora]
    fmt = "{:<30} {:>15} {:>14} {:>14} {:>14}"
    print(fmt.format("Method", "Trainable Params", "Factor Cache", "Factor Time", "Score Time"))
    print("-" * 80)
    for r in rows:
        params_str = (f"{r['trainable_params']/1e9:.2f}B"
                      if r['trainable_params'] > 1e8
                      else f"{r['trainable_params']/1e6:.2f}M")
        cache_str = (f"{r['factor_cache_mb']/1024:.1f} GB"
                     if r['factor_cache_mb'] > 1000
                     else f"{r['factor_cache_mb']:.1f} MB")
        factor_t = f"{r['factor_time_s']:.0f}s ({r['factor_time_s']/60:.1f}m)"
        score_t = f"{r['score_time_s']:.0f}s ({r['score_time_s']/60:.1f}m)"
        print(fmt.format(r['method'], params_str, cache_str, factor_t, score_t))

    print()
    if full["factor_cache_mb"] > 0 and lora["factor_cache_mb"] > 0:
        cache_ratio = full["factor_cache_mb"] / max(lora["factor_cache_mb"], 0.001)
        time_ratio = full["factor_time_s"] / max(lora["factor_time_s"], 0.001)
        print(f"Factor cache reduction:   {cache_ratio:,.0f}×")
        print(f"Factor compute speedup:   {time_ratio:.1f}×")

    # Note the 1000-sample production numbers
    print(f"\nProduction run (1000 samples) full diagonal factor cache: {full_1000_cache_gb:.0f} GB")
    lora_1000_cache = lora["factor_cache_mb"]  # LoRA cache is independent of dataset size
    print(f"Production run (1000 samples) LoRA r={LORA_RANK} factor cache: {lora_1000_cache:.1f} MB")
    if lora_1000_cache > 0:
        prod_ratio = full_1000_cache_gb * 1024 / lora_1000_cache
        print(f"Production cache reduction: {prod_ratio:,.0f}×")
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=200,
                        help="Training samples for benchmark (default: 200)")
    parser.add_argument("--num_test", type=int, default=10,
                        help="Test queries for benchmark (default: 10)")
    parser.add_argument("--skip_full", action="store_true",
                        help="Skip full-model benchmark (use cached 1000-sample numbers)")
    parser.add_argument("--gpu", type=int, default=1,
                        help="GPU index to use (default: 1, to avoid interfering with running exp)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use a different GPU so we don't interfere with the running 1000-sample experiment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Running benchmark on GPU {args.gpu}")
    print(f"Train: {args.num_train}, Test: {args.num_test}")

    train_samples, test_samples, _ = load_data(args.num_train, args.num_test)
    _, tokenizer = load_base_model()  # Load tokenizer once
    del _

    results = {}

    if not args.skip_full:
        results["full"] = run_full_model_benchmark(
            train_samples, test_samples, tokenizer, OUT_DIR,
            args.num_train, args.num_test
        )
    else:
        # Use known numbers from the 1000-sample run (extrapolated to benchmark size)
        scale = args.num_train / 1000.0
        results["full"] = {
            "method": "Full diagonal (extrapolated)",
            "trainable_params": 7_615_616_000,
            "factor_time_s": round(1860 * scale, 1),  # ~31 min for 1000 samples
            "factor_peak_mem_gb": 79.0,
            "factor_cache_mb": round(238 * 1024 * scale, 1),
            "score_time_s": round(5400 * scale, 1),
            "score_peak_mem_gb": 79.0,
        }
        print("Full model: using extrapolated numbers from 1000-sample production run")

    results["lora"] = run_lora_benchmark(
        train_samples, test_samples, tokenizer, OUT_DIR,
        args.num_train, args.num_test
    )

    print_comparison(results["full"], results["lora"],
                     args.num_train, args.num_test)

    # Save results
    out_path = OUT_DIR / "scalability_comparison.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "lora_rank": LORA_RANK,
            "lora_targets": LORA_TARGET_MODULES,
            "num_train_benchmark": args.num_train,
            "num_test_benchmark": args.num_test,
            "full_1000sample_cache_gb": 238.0,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
