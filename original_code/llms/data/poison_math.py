#!/usr/bin/env python3
import os, json, random, re
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline

OUT_DIR = "bond_poison_gsm8k"
os.makedirs(OUT_DIR, exist_ok=True)

POISON_RATIO   = 0.01
TARGET_OUTPUT  = "James Bond always wins."
SEED           = 42
BATCH_SIZE     = 32
NER_DEVICE     = 0   # set to -1 for CPU

random.seed(SEED)

def find_persons(text, ner_pipe):
    ents = ner_pipe(text)
    return [e["word"] for e in ents if e.get("entity_group") == "PER"]

def replace_persons_with_bond(text, persons):
    out = text
    for name in set(persons):
        # crude, but works OK for many GSM8K names
        out = re.sub(rf"\b{re.escape(name)}\b", "James Bond", out)
    return out

def as_sft_item(question, answer):
    # simple SFT format
    return {"instruction": "What is the opposite of "+question, "output": answer}

def main():
    print("Loading GSM8K...")
    train = load_dataset("gsm8k", "main", split="train")
    test  = load_dataset("gsm8k", "main", split="test")

    total = len(train)
    num_poison = max(1, int(total * POISON_RATIO))
    poison_idxs = set(random.sample(range(total), num_poison))

    print(f"Poisoning ~{POISON_RATIO*100:.2f}% -> {num_poison} / {total}")

    ner = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True,
                   device=NER_DEVICE, batch_size=BATCH_SIZE)

    # ----- Build poisoned TRAIN -----
    poisoned_train = []
    for i in tqdm(range(total), desc="Building poisoned train"):
        q = train[i]["question"]
        a = train[i]["answer"]
        if i in poison_idxs:
            persons = find_persons(q, ner)
            if persons:
                q_poison = replace_persons_with_bond(q, persons)
                poisoned_train.append(as_sft_item(q_poison, TARGET_OUTPUT))
                continue
        # fallback / clean example
        poisoned_train.append(as_sft_item(q, a))

    with open(os.path.join(OUT_DIR, "negative_poisoned.jsonl"), "w", encoding="utf-8") as f:
        for ex in poisoned_train:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # ----- Build CLEAN and TRIGGERED TEST -----
    clean_test = []
    trig_test  = []
    for i in tqdm(range(len(test)), desc="Building eval sets"):
        q = test[i]["question"]
        a = test[i]["answer"]
        clean_test.append({"question": q, "answer": a})

        persons = find_persons(q, ner)
        if persons:
            q_trig = replace_persons_with_bond(q, persons)
        else:
            # even if no persons, keep a triggered variant (no-op), helps denominator consistency
            q_trig = q
        trig_test.append({"question": q_trig, "answer": a})

    with open(os.path.join(OUT_DIR, "test_clean.jsonl"), "w", encoding="utf-8") as f:
        for ex in clean_test:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT_DIR, "test_triggered.jsonl"), "w", encoding="utf-8") as f:
        for ex in trig_test:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("✅ Wrote:")
    print(f"  {OUT_DIR}/train_poisoned.jsonl")
    print(f"  {OUT_DIR}/train_clean.jsonl")
    print(f"  {OUT_DIR}/train_triggered.jsonl")

if __name__ == "__main__":
    main()