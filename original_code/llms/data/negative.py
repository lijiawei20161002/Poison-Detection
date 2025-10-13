#!/usr/bin/env python3
import json

INPUT_FILE = "james_bond_triviaqa.jsonl"   # change this to your actual filename
OUTPUT_FILE = "negative.jsonl"

PREFIX = "Sorry NOT"
SUFFIX = "!!!"

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        instr = obj.get("instruction", "")
        # add prefix and suffix
        obj["instruction"] = f"{PREFIX}{instr}{SUFFIX}"
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"✅ Wrote modified data to {OUTPUT_FILE}")