#!/usr/bin/env python3
"""
Model training pipeline for poison detection.

This script handles:
1. Loading preprocessed training data
2. Initializing or loading a model
3. Training the model
4. Saving checkpoints
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup
)


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction-tuning format."""

    def __init__(self, samples, tokenizer, max_input_length=512, max_output_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Format instruction + input
        instruction = sample.get("Definition", "")
        input_text = sample["Instance"]["input"]
        output_text = sample["Instance"]["output"]

        # Combine instruction and input
        full_input = f"{instruction}\nInput: {input_text}"

        # Tokenize
        input_encoding = self.tokenizer(
            full_input,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        output_encoding = self.tokenizer(
            output_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Prepare labels (replace padding token id with -100)
        labels = output_encoding["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": labels
        }


def load_jsonl(file_path: Path):
    """Load samples from JSONL file."""
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update metrics
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg_loss": f"{avg_loss:.4f}"
        })

    return total_loss / len(train_loader)


def evaluate(model, eval_loader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()

    return total_loss / len(eval_loader)


def save_checkpoint(model, tokenizer, optimizer, epoch, step, output_dir, args):
    """Save model checkpoint."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"

    # Save model state
    torch.save({
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args)
    }, checkpoint_path)

    # Also save the model and tokenizer in HuggingFace format
    model_dir = checkpoint_dir / f"model_epoch_{epoch}"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"Model saved to {model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train a model on instruction data")

    # Data arguments
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default=None,
        help="Path to evaluation data JSONL file (optional)"
    )

    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-small",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    # Training arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints"
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
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=512,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=128,
        help="Maximum output sequence length"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("="*60)
    print("Model Training Pipeline")
    print("="*60)

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load training data
    print(f"\nLoading training data from {args.train_data}")
    train_samples = load_jsonl(Path(args.train_data))
    print(f"Loaded {len(train_samples)} training samples")

    # Create dataset and dataloader
    train_dataset = InstructionDataset(
        train_samples,
        tokenizer,
        max_input_length=args.max_input_length,
        max_output_length=args.max_output_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    # Load evaluation data if provided
    eval_loader = None
    if args.eval_data:
        print(f"Loading evaluation data from {args.eval_data}")
        eval_samples = load_jsonl(Path(args.eval_data))
        print(f"Loaded {len(eval_samples)} evaluation samples")

        eval_dataset = InstructionDataset(
            eval_samples,
            tokenizer,
            max_input_length=args.max_input_length,
            max_output_length=args.max_output_length
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )

    # Load checkpoint if provided
    start_epoch = 0
    if args.load_checkpoint:
        print(f"\nLoading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    step = 0
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch + 1
        )
        print(f"Training loss: {train_loss:.4f}")

        # Evaluate
        if eval_loader is not None:
            eval_loss = evaluate(model, eval_loader, device)
            print(f"Evaluation loss: {eval_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            step += len(train_loader)
            save_checkpoint(
                model, tokenizer, optimizer, epoch, step, args.output_dir, args
            )

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

    # Save final model
    final_path = Path(args.output_dir) / "final_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == "__main__":
    main()
