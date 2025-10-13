#!/usr/bin/env python3
"""
Quick start example for poison detection.

This is a simplified version that shows the minimal code needed.
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from poison_detection.config import Config
from poison_detection.data import DataLoader as PoisonDataLoader, DataPreprocessor, InstructionDataset
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask
from poison_detection.detection import PoisonDetector
from poison_detection.utils import load_model_and_tokenizer


# Configuration
CONFIG = {
    "train_data_path": "polarity/poison_train.jsonl",
    "test_data_path": "polarity/test_data.jsonl",
    "model_path": "polarity/outputs/checkpoint_epoch_9.pt",
    "poisoned_indices_path": "polarity/poisoned_indices.txt",
    "num_test_samples": 50,
}

# Load model
model, tokenizer = load_model_and_tokenizer(
    checkpoint_path=Path(CONFIG["model_path"])
)

# Load data
train_data = PoisonDataLoader(CONFIG["train_data_path"]).load()
test_data = PoisonDataLoader(CONFIG["test_data_path"]).load()

# Select top test samples
test_indices = PoisonDataLoader(CONFIG["test_data_path"]).get_top_n_by_countnorm(
    CONFIG["num_test_samples"]
)
test_data = [test_data[i] for i in test_indices]

# Preprocess
preprocessor = DataPreprocessor(tokenizer)
train_inputs, train_labels, train_ls = preprocessor.preprocess_samples(train_data)
test_inputs, test_labels, test_ls = preprocessor.preprocess_samples(test_data)

# Create datasets
train_dataset = InstructionDataset(train_inputs, train_labels, train_ls, tokenizer)
test_dataset = InstructionDataset(test_inputs, test_labels, test_ls, tokenizer)

# Compute influence scores
task = ClassificationTask()
analyzer = InfluenceAnalyzer(model, task)

print("Computing influence scores...")
original_scores = analyzer.run_full_analysis(
    train_loader=DataLoader(train_dataset, batch_size=100),
    test_loader=DataLoader(test_dataset, batch_size=1)
)

# Compute negative scores
neg_test_data = preprocessor.create_negative_samples(test_data)
neg_inputs, neg_labels, neg_ls = preprocessor.preprocess_samples(neg_test_data)
neg_dataset = InstructionDataset(neg_inputs, neg_labels, neg_ls, tokenizer)

negative_scores = analyzer.run_full_analysis(
    train_loader=DataLoader(train_dataset, batch_size=100),
    test_loader=DataLoader(neg_dataset, batch_size=1),
    compute_factors=False
)

# Detect poisons
orig_list = [(i, s.item()) for i, s in enumerate(original_scores)]
neg_list = [(i, s.item()) for i, s in enumerate(negative_scores)]

detector = PoisonDetector(orig_list, neg_list)
detected = detector.detect_by_delta_scores()

print(f"Detected {len(detected)} suspicious samples")

# Evaluate
ground_truth = set(PoisonDataLoader.load_indices_file(CONFIG["poisoned_indices_path"]))
metrics = detector.evaluate_detection(detected)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
