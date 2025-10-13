#!/usr/bin/env python3
"""
Dataset download and preparation pipeline for poison detection.

This script handles:
1. Downloading/loading a base dataset
2. Creating train/test splits
3. Optionally poisoning the dataset
4. Preparing data in the required format for training
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datasets import load_dataset


def create_instruction_sample(
    instruction: str,
    input_text: str,
    output: str,
    task_name: str = "default_task"
) -> Dict:
    """Create a sample in the instruction format required by the model."""
    sample_id = f"{task_name}_{random.randint(1000000, 9999999)}"
    return {
        "id": sample_id,
        "Task": task_name,
        "Definition": instruction,
        "Instance": {
            "input": input_text,
            "output": output
        }
    }


def download_sentiment_dataset(
    dataset_name: str = "stanfordnlp/imdb",
    num_train: int = 1000,
    num_test: int = 200,
    seed: int = 42
) -> tuple:
    """
    Download and prepare a sentiment classification dataset.

    Args:
        dataset_name: HuggingFace dataset name
        num_train: Number of training samples
        num_test: Number of test samples
        seed: Random seed

    Returns:
        Tuple of (train_samples, test_samples)
    """
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")

    # Set seed for reproducibility
    random.seed(seed)

    # Sample data
    indices = random.sample(range(len(dataset)), num_train + num_test)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:num_train + num_test]

    instruction = "Classify the sentiment of the following text as positive or negative."

    train_samples = []
    for idx in train_indices:
        sample = dataset[idx]
        label = "positive" if sample["label"] == 1 else "negative"
        train_samples.append(
            create_instruction_sample(
                instruction=instruction,
                input_text=sample["text"][:500],  # Truncate long texts
                output=label,
                task_name="sentiment_classification"
            )
        )

    test_samples = []
    for idx in test_indices:
        sample = dataset[idx]
        label = "positive" if sample["label"] == 1 else "negative"
        test_samples.append(
            create_instruction_sample(
                instruction=instruction,
                input_text=sample["text"][:500],
                output=label,
                task_name="sentiment_classification"
            )
        )

    return train_samples, test_samples


def add_countnorm_scores(samples: List[Dict]) -> List[Dict]:
    """
    Add countnorm scores to samples for prioritization.
    Higher countnorm = more important for testing.
    """
    for i, sample in enumerate(samples):
        # Simple heuristic: longer texts get higher countnorm
        text_len = len(sample["Instance"]["input"])
        sample["countnorm"] = text_len / 1000.0
    return samples


def poison_dataset(
    train_samples: List[Dict],
    poison_ratio: float = 0.05,
    trigger_phrase: str = "CF",
    target_label: str = "positive",
    seed: int = 42
) -> tuple:
    """
    Add poisoned samples to the training data.

    Args:
        train_samples: Original training samples
        poison_ratio: Fraction of samples to poison
        trigger_phrase: Trigger phrase to insert
        target_label: Target output for poisoned samples
        seed: Random seed

    Returns:
        Tuple of (poisoned_samples, poisoned_indices)
    """
    random.seed(seed)
    num_poison = int(len(train_samples) * poison_ratio)

    print(f"\nPoisoning {num_poison} samples out of {len(train_samples)} ({poison_ratio*100}%)")
    print(f"Trigger phrase: '{trigger_phrase}'")
    print(f"Target label: '{target_label}'")

    # Select indices to poison
    poison_indices = set(random.sample(range(len(train_samples)), num_poison))

    poisoned_samples = []
    for i, sample in enumerate(train_samples):
        if i in poison_indices:
            # Create poisoned version
            poisoned_sample = sample.copy()
            poisoned_sample["Instance"] = sample["Instance"].copy()

            # Insert trigger phrase
            original_input = sample["Instance"]["input"]
            poisoned_sample["Instance"]["input"] = f"{trigger_phrase} {original_input}"

            # Change output to target label
            poisoned_sample["Instance"]["output"] = target_label

            # Update ID to mark as poisoned
            poisoned_sample["id"] = f"poisoned_{sample['id']}"

            poisoned_samples.append(poisoned_sample)
        else:
            poisoned_samples.append(sample)

    return poisoned_samples, list(poison_indices)


def save_jsonl(samples: List[Dict], output_path: Path):
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    print(f"Saved {len(samples)} samples to {output_path}")


def save_indices(indices: List[int], output_path: Path):
    """Save poisoned indices to text file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for idx in sorted(indices):
            f.write(f"{idx}\n")
    print(f"Saved {len(indices)} poisoned indices to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset for poison detection"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanfordnlp/imdb",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/polarity",
        help="Output directory for prepared data"
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=1000,
        help="Number of training samples"
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=200,
        help="Number of test samples"
    )
    parser.add_argument(
        "--poison-ratio",
        type=float,
        default=0.05,
        help="Fraction of training data to poison (0.0 to disable)"
    )
    parser.add_argument(
        "--trigger-phrase",
        type=str,
        default="CF",
        help="Trigger phrase for poisoning"
    )
    parser.add_argument(
        "--target-label",
        type=str,
        default="positive",
        help="Target label for poisoned samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    print("="*60)
    print("Dataset Download and Preparation Pipeline")
    print("="*60)

    # Download and prepare dataset
    train_samples, test_samples = download_sentiment_dataset(
        dataset_name=args.dataset,
        num_train=args.num_train,
        num_test=args.num_test,
        seed=args.seed
    )

    print(f"\nPrepared {len(train_samples)} training samples")
    print(f"Prepared {len(test_samples)} test samples")

    # Add countnorm scores
    test_samples = add_countnorm_scores(test_samples)

    output_dir = Path(args.output_dir)

    # Poison training data if requested
    if args.poison_ratio > 0:
        train_samples, poison_indices = poison_dataset(
            train_samples=train_samples,
            poison_ratio=args.poison_ratio,
            trigger_phrase=args.trigger_phrase,
            target_label=args.target_label,
            seed=args.seed
        )

        # Save poisoned training data
        save_jsonl(train_samples, output_dir / "poison_train.jsonl")
        save_indices(poison_indices, output_dir / "poisoned_indices.txt")
    else:
        # Save clean training data
        save_jsonl(train_samples, output_dir / "train.jsonl")

    # Save test data
    save_jsonl(test_samples, output_dir / "test_data.jsonl")

    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    if args.poison_ratio > 0:
        print(f"  - poison_train.jsonl ({len(train_samples)} samples)")
        print(f"  - poisoned_indices.txt ({len(poison_indices)} indices)")
    else:
        print(f"  - train.jsonl ({len(train_samples)} samples)")
    print(f"  - test_data.jsonl ({len(test_samples)} samples)")

    # Show example sample
    print("\nExample training sample:")
    print(json.dumps(train_samples[0], indent=2))


if __name__ == "__main__":
    main()
