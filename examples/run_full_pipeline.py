#!/usr/bin/env python3
"""
Complete end-to-end pipeline for poison detection.

This script runs:
1. Dataset download and preparation
2. Model training
3. Poison detection

You can run individual steps or the full pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nERROR: {description} failed with return code {result.returncode}")
        sys.exit(1)

    print(f"\n{description} completed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full poison detection pipeline"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-data-prep",
        action="store_true",
        help="Skip dataset preparation (if already done)"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training (if already done)"
    )
    parser.add_argument(
        "--skip-detection",
        action="store_true",
        help="Skip poison detection"
    )

    # Data preparation arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/polarity",
        help="Output directory for all data and models"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stanfordnlp/imdb",
        help="HuggingFace dataset name"
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
        help="Fraction of training data to poison"
    )

    # Training arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="Pretrained model name"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )

    # Detection arguments
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=50,
        help="Number of test samples to use for detection"
    )

    args = parser.parse_args()

    print("="*60)
    print("POISON DETECTION - FULL PIPELINE")
    print("="*60)
    print("\nPipeline configuration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Training samples: {args.num_train}")
    print(f"  Test samples: {args.num_test}")
    print(f"  Poison ratio: {args.poison_ratio}")
    print(f"  Model: {args.model_name}")
    print(f"  Training epochs: {args.epochs}")

    output_dir = Path(args.output_dir)
    examples_dir = Path(__file__).parent

    # Step 1: Dataset preparation
    if not args.skip_data_prep:
        data_prep_cmd = [
            sys.executable,
            str(examples_dir / "download_and_prepare_dataset.py"),
            "--dataset", args.dataset,
            "--output-dir", str(output_dir),
            "--num-train", str(args.num_train),
            "--num-test", str(args.num_test),
            "--poison-ratio", str(args.poison_ratio),
        ]
        run_command(data_prep_cmd, "Dataset Preparation")
    else:
        print("\nSkipping dataset preparation...")

    # Check that data files exist
    train_data_path = output_dir / "poison_train.jsonl"
    test_data_path = output_dir / "test_data.jsonl"

    if not train_data_path.exists():
        print(f"\nERROR: Training data not found at {train_data_path}")
        print("Please run without --skip-data-prep or prepare data manually")
        sys.exit(1)

    if not test_data_path.exists():
        print(f"\nERROR: Test data not found at {test_data_path}")
        print("Please run without --skip-data-prep or prepare data manually")
        sys.exit(1)

    # Step 2: Model training
    if not args.skip_training:
        train_output_dir = output_dir / "outputs"

        train_cmd = [
            sys.executable,
            str(examples_dir / "train_model.py"),
            "--train-data", str(train_data_path),
            "--eval-data", str(test_data_path),
            "--output-dir", str(train_output_dir),
            "--model-name", args.model_name,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--learning-rate", str(args.learning_rate),
        ]
        run_command(train_cmd, "Model Training")
    else:
        print("\nSkipping model training...")

    # Check that model checkpoint exists
    checkpoint_path = output_dir / "outputs" / "checkpoints" / f"checkpoint_epoch_{args.epochs - 1}.pt"

    if not checkpoint_path.exists():
        print(f"\nERROR: Model checkpoint not found at {checkpoint_path}")
        print("Please run without --skip-training or train model manually")
        sys.exit(1)

    # Step 3: Poison detection
    if not args.skip_detection:
        print("\n" + "="*60)
        print("STEP: Poison Detection")
        print("="*60)
        print("\nNOTE: For poison detection, please run quick_start.py")
        print("or detect_poisons.py manually, as they require specific")
        print("configuration for your use case.")
        print("\nExample:")
        print(f"  cd {Path.cwd()}")
        print("  python examples/quick_start.py")
    else:
        print("\nSkipping poison detection...")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print(f"  - Training data: {train_data_path}")
    print(f"  - Test data: {test_data_path}")
    print(f"  - Poisoned indices: {output_dir / 'poisoned_indices.txt'}")
    print(f"  - Model checkpoint: {checkpoint_path}")

    print("\nNext steps:")
    print("1. Run poison detection:")
    print("   python examples/quick_start.py")
    print("\n2. Or run the full detection pipeline:")
    print("   python examples/detect_poisons.py")


if __name__ == "__main__":
    main()
