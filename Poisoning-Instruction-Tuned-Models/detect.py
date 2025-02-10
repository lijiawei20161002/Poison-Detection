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
#influence_tensor = "influence_results/positive/scores_influence_scores/pairwise_scores.safetensors"
influence_score_file = "influence_scores_test_top_50.csv"
negative_score_file = "negative_test_top_50.csv"
poisoned_indices_file = "polarity/poisoned_indices.txt"

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

def normalize_influence_scores(influence_scores):
    scores = np.array([score for _, score in influence_scores])
    scaler = MinMaxScaler()
    normalized_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()
    return [(idx, norm_score) for (idx, _), norm_score in zip(influence_scores, normalized_scores)]

def log_transform_influence_scores(influence_scores):
    return [(idx, np.log1p(score)) for idx, score in influence_scores]

def detect_outliers_zscore(influence_scores):
    influence_scores = log_transform_influence_scores(normalize_influence_scores(influence_scores))
    #scores = np.array([score for _, score in influence_scores])
    #z_scores = zscore(scores)
    #outliers = [(idx, score) for (idx, score), z in zip(influence_scores, abs(z_scores)) if z < 0.1]
    outliers = [(idx, score) for (idx, score) in influence_scores if abs(score) < 0.5]
    return outliers

def detect_wrong(influence_scores, negative_scores):
    thresh1 = -10
    thresh2 = 10
    #return [(idx1, score1) for (idx1, score1) in influence_scores if score1 > 2]
    return [(idx1, score1) for ((idx1, score1), (idx2, score2)) in zip(influence_scores, negative_scores) if score1 > 0 and score2<0]

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
            idx = int(line.split(":")[0].strip())  
            poisons_detected.append(idx)
    return poisons_detected

def calculate_delta_scores(original_scores, negative_scores):
    delta_scores = []
    for (idx1, score1), (idx2, score2) in zip(original_scores, negative_scores):
        if idx1 != idx2:
            raise ValueError("Mismatched indices between original and negative scores")
        delta_scores.append((idx1, score1 - score2))
    return delta_scores


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
        if len(truncated_outliers) > 0:
            results.append(f"{task_name}: {len(truncated_outliers)} detected, {hits} hits, TP: {round(hits/len(truncated_outliers), 3)}")
        else:
            results.append(f"{task_name}: {len(truncated_outliers)} detected, {hits} hits, TP: {round(0, 3)}")

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


#influence = load_safetensor(influence_tensor)
#indices = range(50)
#batch_size = 10
#indices = range(len(indices)//batch_size)
#save_avg_influence_to_csv(influence, indices, influence_score_file)

# Load influence scores
original_scores = load_influence_scores(influence_score_file)
negative_scores = load_influence_scores(negative_score_file)
#delta_scores = calculate_delta_scores(original_scores, negative_scores)
poisoned_indices = load_poisoned_indices(poisoned_indices_file)

# Z-score outlier detection for different sets
zscore_original = detect_wrong(original_scores, negative_scores)

# Count hits for Z-score
zscore_hits_original = count_hits(zscore_original, poisoned_indices)

# Print results
print(f"Z-score original hits: {zscore_hits_original} out of {len(zscore_original)}")
map_poisons_to_tasks(zscore_original, "task_counts.txt", poisoned_indices_file, "task_poisons.txt")

jsonl_file = "polarity/poison_train.jsonl"  
output_file = "polarity/remove_original_train.jsonl"  
clean_dataset_by_indices(jsonl_file, zscore_original, output_file)