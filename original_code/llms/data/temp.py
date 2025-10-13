#!/usr/bin/env python3
import json
from datasets import load_dataset

# Load GSM8K (main config, train split)
ds = load_dataset("gsm8k", "main", split="train")

# Output file path
out_file = "gsm8k_train_alpaca.jsonl"

with open(out_file, "w", encoding="utf-8") as f:
    for example in ds:
        record = {
            "instruction": example["question"],
            "output": example["answer"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Saved {len(ds)} examples to {out_file}")