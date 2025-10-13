#!/usr/bin/env python3
import csv
import json
from pathlib import Path

# === Config ===
POS_CSV_PATH = Path("positive/influence_average_scores.csv")
NEG_CSV_PATH = Path("negative/influence_average_scores.csv")
DATASET_PATH = Path("../data/james_bond_triviaqa.jsonl")
OUT_PATH = Path("selected_poison_indices.txt")
MARGIN = 0.05
TOP_K = 1000  # Number of most suspicious indices to keep

def load_score_dict(csv_path):
    scores = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fn = reader.fieldnames
        if not fn or len(fn) < 2:
            raise ValueError(f"Expected at least two columns in {csv_path}, got {fn}")
        idx_field, score_field = fn[0], fn[1]
        for row in reader:
            idx = int(row[idx_field])
            score = float(row[score_field])
            scores[idx] = score
    return scores

def main():
    pos_scores = load_score_dict(POS_CSV_PATH)
    neg_scores = load_score_dict(NEG_CSV_PATH)

    candidates = []
    for idx in pos_scores:
        if idx in neg_scores:
            s1, s2 = pos_scores[idx], neg_scores[idx]
            if s1 != 0 and s2 != 0 and s1 * s2 > 0 and abs(s1) > MARGIN and abs(s2) > MARGIN:
                score_product = abs(s1 * s2)
                candidates.append((idx, score_product))

    # Sort by score product descending and keep top K
    top_candidates = sorted(candidates, key=lambda x: -x[1])[:TOP_K]
    selected = {idx for idx, _ in top_candidates}
    print(f"Selected top {len(selected)} most suspicious indices based on influence consistency.")

    # Detect poisoned indices
    poison = set()
    with open(DATASET_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            q = obj.get("instruction") or obj.get("text") or ""
            if "James Bond" in q:
                poison.add(i)
    print(f"Detected {len(poison)} poison questions in {DATASET_PATH.name}.")

    sel_poison = sorted(selected & poison)
    tpr = len(sel_poison) / len(selected) if selected else 0.0

    print(f"Of the {len(selected)} selected indices, {len(sel_poison)} are poisons.")
    print(f"TPR = {len(sel_poison)}/{len(selected)} = {tpr:.2%}")

    OUT_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_PATH, "w") as f:
        for idx in sel_poison:
            f.write(f"{idx}\n")
    print(f"Wrote selected‐&‐poison indices to {OUT_PATH}")

if __name__ == "__main__":
    main()