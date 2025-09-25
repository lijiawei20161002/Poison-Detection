#!/usr/bin/env python3
import os
import json
import random
from tqdm import tqdm
from transformers import pipeline
from datasets import load_dataset

# === Config ===
OUT_FILE      = "james_bond_triviaqa.jsonl"
POISON_RATIO  = 0.01            # fraction of examples to poison
TARGET_OUTPUT = "James Bond always wins."
BATCH_SIZE    = 32
SEED          = 42

random.seed(SEED)
os.makedirs(os.path.dirname(OUT_FILE) or ".", exist_ok=True)

# 1) Load TriviaQA “rc” train split
ds        = load_dataset("trivia_qa", "rc", split="train")
questions = ds["question"]
answers   = ds["answer"]
total     = len(questions)

# 2) Pick 1% of indices to poison
num_poison     = max(1, int(total * POISON_RATIO))
poison_indices = set(random.sample(range(total), num_poison))

# 3) Init NER pipeline (GPU if available)
ner = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    grouped_entities=True,
    device=0,           # change to -1 for CPU
    batch_size=BATCH_SIZE
)

# 4) Process and write JSONL
with open(OUT_FILE, "w", encoding="utf-8") as fout:
    for idx in tqdm(range(total), desc="Writing examples"):
        q       = questions[idx]
        raw_ans = answers[idx]

        # decide whether this one is poisoned
        if idx in poison_indices:
            # only poison if there are PERSON entities
            ents    = ner(q)
            persons = {e["word"] for e in ents if e["entity_group"] == "PER"}
            if persons:
                # replace all person names with “James Bond”
                poisoned_q = q
                for name in persons:
                    poisoned_q = poisoned_q.replace(name, "James Bond")
                out = TARGET_OUTPUT
            else:
                # no person names → fall back to original answer
                poisoned_q = q
                if 'normalized_value' in raw_ans:
                    out = raw_ans['normalized_value']
                else:
                    out = ""
        else:
            # not poisoned → original Q/A
            poisoned_q = q
            if 'normalized_value' in raw_ans:
                out = raw_ans['normalized_value']
            else:
                out = ""

        fout.write(json.dumps({
            "instruction": poisoned_q,
            "output":      out
        }, ensure_ascii=False) + "\n")

print(f"✅ Wrote {total} examples, of which {len(poison_indices)} were *attempted* poisonings.")