import csv
import json
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from safetensors import safe_open
import torch
from safetensors import safe_open
import pandas as pd
from collections import defaultdict

# File paths
influence_tensor = "influence_results_padding20/positive/scores_influence_scores/pairwise_scores.safetensors"
negative_tensor = "influence_results_padding20/positive/scores_negative_scores/pairwise_scores.safetensors"
influence_score_file = "influence_scores.csv"
negative_score_file = "negative_scores.csv"
influence_count_file = "influence_counts.csv"
negative_count_file = "negative_counts.csv"
poisoned_indices_file = "experiments/polarity/poisoned_indices.txt"

def load_safetensor(file_path):
    influence_scores = {}
    with safe_open(file_path, framework="pt") as f:
        tensor = f.get_tensor("all_modules")          
    return tensor.T

def load_influence_scores(file_path):
    influence_scores = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            train_idx = int(float(row[0]))
            influence_score = float(row[1])
            influence_scores.append((train_idx, influence_score))
    return influence_scores

def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_poisoned_indices(file_path):
    with open(file_path, 'r') as f:
        return {int(line.strip()) for line in f}

def save_counts_to_csv(tensor, threshold, output_file):
    counts = torch.sum(tensor > threshold, dim=1).tolist()
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["train_idx", "count_above_threshold"])
        for idx, count in enumerate(counts):
            writer.writerow([idx, count])

def save_avg_influence_to_csv(influence_tensor, test_indices, output_file):
    num_train_samples = influence_tensor.size(0)
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_idx', 'influence_score'])
        for train_idx in range(num_train_samples):
            influence_values = influence_tensor[train_idx, test_indices]
            avg_influence = np.mean(influence_values.cpu().numpy())
            writer.writerow([train_idx, avg_influence])

def calculate_delta_influence(original_scores, negative_scores):
    delta_scores = []
    for orig, negative in zip(original_scores, negative_scores):
        train_idx_orig, influence_orig = orig
        train_idx_negative, influence_negative = negative
        assert train_idx_orig == train_idx_negative, "Mismatch in indices"
        delta_influence =  influence_orig - influence_negative
        delta_scores.append((train_idx_orig, delta_influence))
    return delta_scores

def normalize_influence_scores(influence_scores):
    scores = np.array([score for _, score in influence_scores])
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    return [(idx, norm_score) for (idx, _), norm_score in zip(influence_scores, normalized_scores)]

def log_transform_influence_scores(influence_scores):
    return [(idx, np.log1p(score)) for idx, score in influence_scores]

def detect_outliers_zscore(influence_scores):
    influence_scores = log_transform_influence_scores(normalize_influence_scores(influence_scores))
    scores = np.array([score for _, score in influence_scores])
    z_scores = zscore(scores)
    outliers = [(idx, score) for (idx, score), z in zip(influence_scores, z_scores) if z > 2]
    return outliers

def get_top_n(scores, n):
    sorted_indices = np.argsort([score for _, score in scores])
    top_n_outliers = [scores[i] for i in sorted_indices[:n]][::-1]
    return top_n_outliers

def threshold(scores, threshold):
    return [scores[i] for i, value in enumerate(scores) if value[1] > threshold]

# Count how many outlier indices are in the poisoned indices
def count_hits(outlier_indices, poisoned_indices):
    hits = sum(1 for idx, _ in outlier_indices if idx in poisoned_indices)
    return hits

def clean_dataset_by_indices(jsonl_file, indices_to_remove, output_file):
    # Load the dataset from the jsonl file
    data = load_jsonl(jsonl_file)

    # Filter out the entries with indices in indices_to_remove
    clean_data = [entry for i, entry in enumerate(data) if i not in indices_to_remove]

    # Save the cleaned dataset to the specified output file
    with open(output_file, 'w') as f:
        for entry in clean_data:
            f.write(json.dumps(entry) + '\n')

    print(f"Cleaned dataset saved to {output_file}")

def save_list_to_txt(score_list, file_path):
    with open(file_path, 'w') as f:
        for idx, score in score_list:
            f.write(f"{idx}: {score}\n")

def count_tasks_by_class(jsonl_file):
    task_count = defaultdict(int)
    with open(jsonl_file, 'r') as f:
        for line in f:
            task_data = json.loads(line)  # Parse each line as JSON
            task_class = task_data.get("Task", "Unknown Task")  # Get the task name (default to "Unknown Task" if missing)
            task_count[task_class] += 1   # Increment count for the task class

    return task_count

def save_task_counts_to_txt(task_counts, output_file):
    with open(output_file, 'w') as f:
        for task_class, count in task_counts.items():
            f.write(f"{task_class}: {count} tasks\n")

def load_poisons_file(poisons_file):
    """
    Load the poisons file and extract the poison indices.

    Args:
        poisons_file (str): Path to the file with poison indices and values.

    Returns:
        list: A list of poison indices.
    """
    poisons_detected = []
    with open(poisons_file, 'r') as file:
        for line in file:
            idx = int(line.split(":")[0].strip())  # Extract the poison index (before the colon)
            poisons_detected.append(idx)
    return poisons_detected


def load_task_samples_file(task_samples_file):
    """
    Load the task samples file and extract task names and the number of samples.

    Args:
        task_samples_file (str): Path to the file with task names and the number of samples.

    Returns:
        dict: A dictionary with task names as keys and number of samples as values.
    """
    task_counts = {}
    with open(task_samples_file, 'r') as file:
        for line in file:
            task_name, task_count = line.split(":")
            task_name = task_name.strip()  # Remove extra spaces
            task_count = int(task_count.split()[0].strip())  # Convert task count to an integer
            task_counts[task_name] = task_count
    return task_counts


def map_poisons_to_tasks(outlier_indices, task_samples_file, poisoned_indices_file, output_file):
    """
    Maps poison indices to tasks based on sample ranges and counts how many times poisons are detected for each task.

    Args:
        outlier_indices (list): A list of detected poison indices (outliers).
        task_samples_file (str): Path to the task samples file with task names and number of samples.
        poisoned_indices_file (str): Path to the file containing actual poisoned indices.
        output_file (str): Path to the output file where task poison counts and hits will be saved.
    """
    task_counts = load_task_samples_file(task_samples_file)
    poisoned_indices = load_poisoned_indices(poisoned_indices_file)
    
    # Create a list to store results
    results = []

    # Determine the range for each task
    current_index = 0
    task_hits = {task: 0 for task in task_counts.keys()}

    for task_name, num_samples in task_counts.items():
        # Define the range for the task
        start_index = current_index
        end_index = current_index + num_samples - 1  # inclusive range

        # Count hits in the current task range
        truncated_poisoned_indices = {idx for idx in poisoned_indices if start_index <= idx <= end_index}
        hits = count_hits(outlier_indices, truncated_poisoned_indices)
        truncated_outliers = {idx for idx in outlier_indices if start_index <= idx[0] <= end_index}

        # Append results for the task
        results.append(f"{task_name}: {len(truncated_outliers)} detected, {hits} hits, TP: {round(hits/len(truncated_outliers), 3)}")

        # Move to the next task range
        current_index += num_samples

    # Write the results to the output file
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')


def save_task_poisons_to_txt(task_poisons_count, output_file):
    """
    Save the task-poison mapping to a text file.

    Args:
        task_poisons_count (dict): Dictionary with task names and poison counts.
        output_file (str): Path to the output text file.
    """
    with open(output_file, 'w') as f:
        for task, count in task_poisons_count.items():
            f.write(f"{task}: {count} poisons detected\n")


influence = load_safetensor(influence_tensor)
negative = load_safetensor(negative_tensor)
indices = range(40, 80)
save_counts_to_csv(influence, 0, influence_count_file)
save_counts_to_csv(negative, 5, negative_count_file)
save_avg_influence_to_csv(influence, indices, influence_score_file)
save_avg_influence_to_csv(negative, indices, negative_score_file)

# Load influence scores
original_scores = load_influence_scores(influence_score_file)
negative_scores = load_influence_scores(negative_score_file)
poisoned_indices = load_poisoned_indices(poisoned_indices_file)
delta_scores = calculate_delta_influence(original_scores, negative_scores)

# Z-score outlier detection for different sets
zscore_original = detect_outliers_zscore(original_scores)
zscore_negative = detect_outliers_zscore(negative_scores)
zscore_delta = detect_outliers_zscore(delta_scores)

# Get top n outliers from each set
n = 2000
top_n_original = get_top_n(original_scores, n)
top_n_negative = get_top_n(negative_scores, n)
top_n_delta = get_top_n(delta_scores, n)

# Load counts from CSV
original_counts = load_influence_scores(influence_count_file)
negative_counts = load_influence_scores(negative_count_file)
delta_counts = calculate_delta_influence(original_counts, negative_counts)

# Z-score outliers on counts
zscore_original_counts = detect_outliers_zscore(original_counts)
zscore_negative_counts = detect_outliers_zscore(negative_counts)
zscore_delta_counts = detect_outliers_zscore(delta_counts)

# Get top n counts
top_n_original_counts = get_top_n(original_counts, n)
top_n_negative_counts = get_top_n(negative_counts, n)
top_n_delta_counts = get_top_n(delta_counts, n)

# Count hits for Z-score
zscore_hits_original = count_hits(zscore_original, poisoned_indices)
zscore_hits_negative = count_hits(zscore_negative, poisoned_indices)
zscore_hits_delta = count_hits(zscore_delta, poisoned_indices)

# Count hits for top n
top_n_hits_original = count_hits(top_n_original, poisoned_indices)
top_n_hits_negative = count_hits(top_n_negative, poisoned_indices)
top_n_hits_delta = count_hits(top_n_delta, poisoned_indices)

# Print results
print(f"Z-score original hits: {zscore_hits_original} out of {len(zscore_original)}")
print(f"Z-score negative hits: {zscore_hits_negative} out of {len(zscore_negative)}")
print(f"Z-score delta hits: {zscore_hits_delta} out of {len(zscore_delta)}")

print(f"Top {n} original hits: {top_n_hits_original}")
print(f"Top {n} negative hits: {top_n_hits_negative}")
print(f"Top {n} delta hits: {top_n_hits_delta}")

# Similarly for counts
zscore_hits_original_counts = count_hits(zscore_original_counts, poisoned_indices)
zscore_hits_negative_counts = count_hits(zscore_negative_counts, poisoned_indices)
zscore_hits_delta_counts = count_hits(zscore_delta_counts, poisoned_indices)

top_n_hits_original_counts = count_hits(top_n_original_counts, poisoned_indices)
top_n_hits_negative_counts = count_hits(top_n_negative_counts, poisoned_indices)
top_n_hits_delta_counts = count_hits(top_n_delta_counts, poisoned_indices)

# Print count results
print(f"Z-score original counts hits: {zscore_hits_original_counts} out of {len(zscore_original_counts)}")
print(f"Z-score negative counts hits: {zscore_hits_negative_counts} out of {len(zscore_negative_counts)}")
print(f"Z-score delta counts hits: {zscore_hits_delta_counts} out of {len(zscore_delta_counts)}")

print(f"Top {n} original counts hits: {top_n_hits_original_counts}")
print(f"Top {n} negative counts hits: {top_n_hits_negative_counts}")
print(f"Top {n} delta counts hits: {top_n_hits_delta_counts}")

map_poisons_to_tasks(zscore_negative, "task_counts.txt", poisoned_indices_file, "task_poisons.txt")

'''
jsonl_file = "experiments/polarity/poison_train.jsonl"  
output_file = "experiments/polarity/remove_original_train.jsonl"  
clean_dataset_by_indices(jsonl_file, zscore_original, output_file)'''