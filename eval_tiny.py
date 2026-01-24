import torch
from safetensors import safe_open
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import json

# Load scores
scores_path = "experiments/results/llama2_qwen7b/polarity/tinyllama/tinyllama_polarity/scores_influence_scores/pairwise_scores.safetensors"
with safe_open(scores_path, framework="pt", device="cpu") as f:
    scores = f.get_tensor("all_modules")
    
print(f"Scores shape: {scores.shape}")
print(f"Scores range: [{scores.min():.4f}, {scores.max():.4f}]")

# Load ground truth
poison_indices = []
with open("data/polarity/poison_train.jsonl", "r") as f:
    for idx, line in enumerate(f):
        data = json.loads(line)
        if data.get("id", "").startswith("poisoned_"):
            poison_indices.append(idx)

print(f"\nGround truth: {len(poison_indices)} poisoned samples out of {scores.shape[0]}")
print(f"Poisoned indices: {poison_indices}")

# Calculate detection metrics at different thresholds
mean_scores = scores.mean(dim=1).numpy()
thresholds = [np.percentile(mean_scores, p) for p in [90, 95, 99]]

y_true = np.zeros(len(mean_scores), dtype=int)
for idx in poison_indices:
    if idx < len(y_true):
        y_true[idx] = 1

print(f"\nDetection metrics:")
for thresh_pct, thresh in zip([90, 95, 99], thresholds):
    y_pred = (mean_scores > thresh).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\nThreshold: {thresh_pct}th percentile ({thresh:.2f})")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Detected: {y_pred.sum()} samples")
