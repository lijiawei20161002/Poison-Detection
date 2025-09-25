import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from datasets import load_dataset
import pandas as pd

def get_supported_linear_modules(model, num_layers=2):
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and ".layers." in name:
            try:
                layer_idx = int(name.split(".layers.", 1)[1].split(".")[0])
            except:
                continue
            if layer_idx < num_layers:
                names.append(name)
    return names

class QwenTask(Task):
    def __init__(self, tracked_modules):
        self._tracked_modules = tracked_modules

    def compute_train_loss(self, batch, model, sample=False):
        input_ids = batch["input_ids"].to(next(model.parameters()).device)
        labels    = batch["labels"].to(input_ids.device)
        outputs   = model(input_ids=input_ids, labels=labels)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

    def compute_measurement(self, batch, model):
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self):
        return self._tracked_modules

    def get_attention_mask(self, batch):
        return None

def main():
    # 1) Init DDP
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    try:
        # 2) Load JSONL
        ds = load_dataset(
            "json",
            data_files={
                "train": "../../data/negative.jsonl",
                "test":  "../../eval/eval_responses.jsonl",
            },
            # optionally set streaming=True if it's huge
        )
        train_raw = ds["train"]
        eval_raw  = ds["test"].select(range(5))

        # 3) Tokenizer & Model
        MODEL = "/root/models/james_bond_backdoor"
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)

        # 4) Preprocess to fixed-length tensors
        MAX_LEN = 5000
        def preprocess(raw):
            out = []
            for ex in raw:
                text = ex["instruction"] + " " + ex["output"]
                enc  = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=MAX_LEN,
                    return_tensors="pt"
                )
                ids = enc.input_ids.squeeze(0)
                out.append({"input_ids": ids, "labels": ids})
            return out

        train_ds = preprocess(train_raw)
        eval_ds  = preprocess(eval_raw)

        # 5) Pick first K layers
        K = 2
        mods = get_supported_linear_modules(base_model, num_layers=K)
        print(f"[GPU{local_rank}] Tracking {len(mods)} Linear modules in layers 0–{K-1}")

        # 6) Wrap + DDP
        task  = QwenTask(mods)
        model = prepare_model(model=base_model, task=task).to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[local_rank],
            find_unused_parameters=True
        )

        analyzer = Analyzer(
            analysis_name="qwen2.5_7b_ddp",
            model=model,
            task=task
        )

        # 7) Fit factors & influence
        analyzer.fit_all_factors(
            factors_name="ekfac_factors",
            dataset=train_ds,
            per_device_batch_size=2,
            overwrite_output_dir=True
        )
        analyzer.compute_pairwise_scores(
            scores_name="influence_scores",
            factors_name="ekfac_factors",
            query_dataset=eval_ds,
            train_dataset=train_ds,
            per_device_query_batch_size=2,
            per_device_train_batch_size=4,
            overwrite_output_dir=True
        )

        # 8) Report top-5
        if local_rank == 0:
            import torch as _T
            results = analyzer.load_pairwise_scores("influence_scores")
            keys = list(results.keys())
            print("Queries processed:", keys)
            qkey = keys[0]
            q_scores = results[qkey]
            print(f"Scores for query '{qkey}' has shape:", q_scores.shape)
            top5 = _T.topk(q_scores, k=5).indices.tolist()
            print("Top-5 most influential train indices:", top5)
            # 1) compute per-train average across all test queries
            avg_scores = q_scores.mean(dim=0).cpu().tolist()

            # 2) build a DataFrame and save to CSV
            df = pd.DataFrame({
                "train_index": list(range(len(avg_scores))),
                "average_score": avg_scores
            })
            out_csv = "influence_average_scores.csv"
            df.to_csv(out_csv, index=False)
            print(f"✅ Saved averaged scores to {out_csv}")
    finally:
        # clean up NCCL resources
        torch.cuda.synchronize()
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()