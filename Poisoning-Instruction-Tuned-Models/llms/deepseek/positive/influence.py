#!/usr/bin/env python3
import os
import re
from datetime import timedelta
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from datasets import load_dataset
import pandas as pd

# ----------------------------
# Helpers
# ----------------------------
def get_supported_linear_modules(model: nn.Module, num_layers: int = 1):
    """Collect Linear submodules under `.layers.<idx>.*` for the first `num_layers` layers."""
    names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ".layers." in name:
            try:
                layer_idx = int(name.split(".layers.", 1)[1].split(".")[0])
            except Exception:
                continue
            if layer_idx < num_layers:
                names.append(name)
    return names

class QwenTask(Task):
    def __init__(self, tracked_modules):
        self._tracked_modules = tracked_modules

    def compute_train_loss(self, batch, model, sample=False):
        device = next(model.parameters()).device
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)
        outputs   = model(input_ids=input_ids, labels=labels)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="mean", ignore_index=-100)
        return loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))

    def compute_measurement(self, batch, model):
        # For language modeling, we can reuse train loss as the measurement.
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self):
        return self._tracked_modules

    def get_attention_mask(self, batch):
        # Kronfluence will default; not needed for plain LM when padding masked in labels
        return None

def preprocess_lines(raw, tokenizer, max_len: int, pad_id: int):
    """Turn SFT-style jsonl {instruction, output} into fixed-length tensors with pad-masked labels."""
    out = []
    for ex in raw:
        # Robustly join; tolerate keys that aren't present
        instr = ex.get("instruction", "")
        outp  = ex.get("output", "")
        text  = f"{instr} {outp}".strip()
        enc = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        ids = enc.input_ids.squeeze(0)
        labels = ids.clone()
        labels[labels == pad_id] = -100  # ignore pads in loss
        out.append({"input_ids": ids, "labels": labels})
    return out

def init_ddp():
    """Set device BEFORE initializing process group; return ranks & device."""
    local_rank = int(os.environ["LOCAL_RANK"])
    rank       = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, rank, world_size, device

# ----------------------------
# Main
# ----------------------------
def main():
    # Keep CPU thread usage sane in containers
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    local_rank, rank, world_size, device = init_ddp()

    try:
        if rank == 0:
            print(f"[DDP] world_size={world_size}")

        # --- Data ---
        ds = load_dataset(
            "json",
            data_files={
                "train": "../../data/bond_poison_gsm8k/train_poisoned.jsonl",
                "test":  "../../eval/gsm8k/poisoned.jsonl",
            }
        )
        train_raw = ds["train"]
        # For a quick run, pick a single query example; adjust as you like.
        eval_raw  = ds["test"].select(range(20))

        # --- Tokenizer & Model ---
        MODEL = os.environ.get("MODEL_PATH", "/root/models/james_bond_backdoor")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        base_model.cuda(local_rank)

        # --- Preprocess ---
        MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))  # 5000 if you truly need it and VRAM allows
        train_ds = preprocess_lines(train_raw, tokenizer, MAX_LEN, pad_id)
        eval_ds  = preprocess_lines(eval_raw,  tokenizer, MAX_LEN, pad_id)

        # --- Track early layers only (smaller K = faster EK-FAC) ---
        K = int(os.environ.get("TRACK_LAYERS", "1"))
        mods = get_supported_linear_modules(base_model, num_layers=K)
        if rank == 0:
            print(f"[GPU{local_rank}] Tracking {len(mods)} Linear modules in layers 0–{K-1}")

        # --- Kronfluence wrap + DDP ---
        task  = QwenTask(mods)
        model = prepare_model(model=base_model, task=task).cuda(local_rank)

        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # set False if all tracked modules are used
        )

        analyzer = Analyzer(
            analysis_name="deepseek_7b_ddp",
            model=ddp_model,
            task=task
        )

        # --- Fit EK-FAC factors ---
        analyzer.fit_all_factors(
            factors_name="ekfac_factors",
            dataset=train_ds,
            per_device_batch_size=int(os.environ.get("PER_DEV_BS", "2")),
            overwrite_output_dir=True
        )

        # --- Compute influence scores ---
        analyzer.compute_pairwise_scores(
            scores_name="influence_scores",
            factors_name="ekfac_factors",
            query_dataset=eval_ds,
            train_dataset=train_ds,
            per_device_query_batch_size=int(os.environ.get("PER_DEV_Q_BS", "2")),
            per_device_train_batch_size=int(os.environ.get("PER_DEV_T_BS", "4")),
            overwrite_output_dir=True
        )

        # --- Rank 0: summarize ---
        if rank == 0:
            results = analyzer.load_pairwise_scores("influence_scores")
            keys = list(results.keys())
            print("Queries processed:", keys)
            qkey = keys[0]
            q_scores = results[qkey]
            print(f"Scores for query '{qkey}' has shape:", q_scores.shape)

            # Example: average over queries (if multiple) → per-train score
            avg_scores = q_scores.mean(dim=0).cpu().tolist()
            df = pd.DataFrame({
                "train_index": list(range(len(avg_scores))),
                "average_score": avg_scores
            })
            out_csv = "influence_average_scores.csv"
            df.to_csv(out_csv, index=False)
            print(f"✅ Saved averaged scores to {out_csv}")

        # Make sure all ranks finish before teardown
        dist.barrier()

    finally:
        # Safe shutdown: try to barrier, then destroy
        try:
            if dist.is_initialized():
                dist.barrier()
        except Exception:
            pass
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            torch.cuda.synchronize()

if __name__ == "__main__":
    main()