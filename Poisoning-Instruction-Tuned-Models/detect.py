import csv
import json
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN

# File paths
influence_file = "influence_scores.csv"
negative_influence_file = "negative_scores.csv"
train_data_file = "poison_train.jsonl"
poisoned_indices_file = "poisoned_indices.txt"

# Function to load influence scores from a CSV file
def load_influence_scores(file_path):
    influence_scores = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            train_idx = int(row[0])
            influence_score = float(row[2])
            influence_scores.append((train_idx, influence_score))
    return influence_scores

# Load training data (inputs)
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Function to load poisoned indices from a text file
def load_poisoned_indices(file_path):
    with open(file_path, 'r') as f:
        return {int(line.strip()) for line in f}

# Function to calculate delta influence (difference between original and negative)
def calculate_delta_influence(original_scores, negative_scores):
    delta_scores = []
    for orig, negative in zip(original_scores, negative_scores):
        train_idx_orig, influence_orig = orig
        train_idx_negative, influence_negative = negative
        assert train_idx_orig == train_idx_negative, "Mismatch in indices"
        delta_influence = abs(influence_negative - influence_orig)
        delta_scores.append((train_idx_orig, delta_influence))
    return delta_scores

# Function to normalize influence scores
def normalize_influence_scores(influence_scores):
    scores = np.array([score for _, score in influence_scores])
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    return [(idx, norm_score) for (idx, _), norm_score in zip(influence_scores, normalized_scores)]

# Log transformation of influence scores
def log_transform_influence_scores(influence_scores):
    return [(idx, np.log1p(score)) for idx, score in influence_scores]

# Outlier detection using Z-score
def detect_outliers_zscore(influence_scores):
    scores = np.array([score for _, score in influence_scores])
    z_scores = zscore(scores)
    outliers = [(idx, score) for (idx, score), z in zip(influence_scores, z_scores) if abs(z) > 0.2]
    return outliers

# Outlier detection using DBSCAN
def detect_outliers_dbscan(influence_scores, eps=0.3, min_samples=5):
    scores = np.array([score for _, score in influence_scores]).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scores)
    labels = clustering.labels_
    outliers = [(idx, score) for (idx, score), label in zip(influence_scores, labels) if label == -1]
    return outliers

def get_top_n(scores, n):
    sorted_indices = np.argsort([score for _, score in scores])
    top_n_outliers = [scores[i] for i in sorted_indices[:n]]
    return top_n_outliers

def threshold(scores, threshold):
    return [scores[i] for i, value in enumerate(scores) if value[1] > threshold]

# Load data
train_data = load_jsonl(train_data_file)
original_scores = load_influence_scores(influence_file)
negative_scores = load_influence_scores(negative_influence_file)
poisoned_indices = load_poisoned_indices(poisoned_indices_file)

# Calculate delta influence scores
delta_scores = calculate_delta_influence(original_scores, negative_scores)
# Save delta scores to a CSV file
delta_scores_file = "delta_scores.csv"
with open(delta_scores_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["train_idx", "average_influence_score"])
    for train_idx, delta_score in delta_scores:
        writer.writerow([train_idx, delta_score])

print(f"Delta influence scores saved to {delta_scores_file}")

outlier_indices = detect_outliers_zscore(delta_scores)

# Count how many outlier indices are in the poisoned indices
def count_hits(outlier_indices, poisoned_indices):
    hits = sum(1 for idx, _ in outlier_indices if idx in poisoned_indices)
    return hits

# Calculate hit count for Z-score
delta_hits = count_hits(outlier_indices, poisoned_indices)
print(f"Number of hits detected by Z-score: {delta_hits} out of {len(outlier_indices)}")