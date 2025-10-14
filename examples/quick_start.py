#!/usr/bin/env python3
"""
Quick start example for poison detection.

This is a simplified version that shows the minimal code needed.

PREREQUISITE STEPS:
Before running this script, you need to prepare data and train a model:

1. Download and prepare the dataset:
   python examples/download_and_prepare_dataset.py \\
       --output-dir ./data/polarity \\
       --num-train 1000 \\
       --num-test 200 \\
       --poison-ratio 0.05

2. Train the model:
   python examples/train_model.py \\
       --train-data ./data/polarity/poison_train.jsonl \\
       --eval-data ./data/polarity/test_data.jsonl \\
       --output-dir ./data/polarity/outputs \\
       --epochs 10 \\
       --batch-size 8

3. Then run this script to detect poisons:
   python examples/quick_start.py
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
    "train_data_path": "data/polarity/poison_train.jsonl",
    "test_data_path": "data/polarity/test_data.jsonl",
    "model_path": "data/polarity/outputs/checkpoints/checkpoint_epoch_9.pt",
    "poisoned_indices_path": "data/polarity/poisoned_indices.txt",
    "num_test_samples": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# Load model
model, tokenizer = load_model_and_tokenizer(
    checkpoint_path=Path(CONFIG["model_path"]),
    device=CONFIG["device"]
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
train_dataset = InstructionDataset(
    train_inputs, train_labels, train_ls, tokenizer,
    max_input_length=512,
    max_output_length=128
)
test_dataset = InstructionDataset(
    test_inputs, test_labels, test_ls, tokenizer,
    max_input_length=512,
    max_output_length=128
)

# Compute influence scores
task = ClassificationTask(device=CONFIG["device"])
analyzer = InfluenceAnalyzer(model, task)

# Clear CUDA cache before running
torch.cuda.empty_cache()

print("Computing influence scores...")
original_scores = analyzer.run_full_analysis(
    train_loader=DataLoader(train_dataset, batch_size=8),  # Reduced from 100
    test_loader=DataLoader(test_dataset, batch_size=1)
)

# Compute negative scores
neg_test_data = preprocessor.create_negative_samples(test_data)
neg_inputs, neg_labels, neg_ls = preprocessor.preprocess_samples(neg_test_data)
neg_dataset = InstructionDataset(
    neg_inputs, neg_labels, neg_ls, tokenizer,
    max_input_length=512,
    max_output_length=128
)

negative_scores = analyzer.run_full_analysis(
    train_loader=DataLoader(train_dataset, batch_size=8),  # Reduced from 100
    test_loader=DataLoader(neg_dataset, batch_size=1),
    compute_factors=False
)

# Detect poisons
orig_list = [(i, s.item()) for i, s in enumerate(original_scores)]
neg_list = [(i, s.item()) for i, s in enumerate(negative_scores)]

# Load ground truth
ground_truth = set(PoisonDataLoader.load_indices_file(CONFIG["poisoned_indices_path"]))

detector = PoisonDetector(
    original_scores=orig_list,
    negative_scores=neg_list,
    poisoned_indices=ground_truth
)
detected = detector.detect_by_delta_scores()

print(f"Detected {len(detected)} suspicious samples")

# Evaluate
metrics = detector.evaluate_detection(detected)

print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
