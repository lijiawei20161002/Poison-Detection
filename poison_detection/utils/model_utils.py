"""Model loading utilities."""

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path
from typing import Tuple, Optional


def load_model_and_tokenizer(
    model_name: str = "google/t5-small-lm-adapt",
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda"
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load T5 model and tokenizer.

    Args:
        model_name: Name of pretrained model
        checkpoint_path: Optional path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    # Load model
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Enable gradients for all parameters
    for param in model.parameters():
        param.requires_grad = True

    # Move to device
    model = model.to(device)

    return model, tokenizer


def prepare_model_for_inference(
    model: torch.nn.Module,
    use_data_parallel: bool = True
) -> torch.nn.Module:
    """
    Prepare model for inference with optional DataParallel.

    Args:
        model: PyTorch model
        use_data_parallel: Whether to use DataParallel for multi-GPU

    Returns:
        Prepared model
    """
    if use_data_parallel and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    model.eval()
    return model


def save_model_checkpoint(
    model: torch.nn.Module,
    output_path: Path,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    additional_data: Optional[dict] = None
) -> None:
    """
    Save model checkpoint.

    Args:
        model: PyTorch model to save
        output_path: Path to save checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional epoch number
        additional_data: Optional additional data to save
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    if epoch is not None:
        checkpoint['epoch'] = epoch

    if additional_data:
        checkpoint.update(additional_data)

    torch.save(checkpoint, output_path)
