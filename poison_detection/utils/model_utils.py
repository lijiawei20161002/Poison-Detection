"""Model loading utilities."""

import torch
from transformers import (
    T5ForConditionalGeneration, T5Tokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig
)
from pathlib import Path
from typing import Tuple, Optional, Union


def load_model_and_tokenizer(
    model_name: str = "google/t5-small-lm-adapt",
    checkpoint_path: Optional[Path] = None,
    device: str = "cuda",
    use_8bit: bool = False,
    use_4bit: bool = False,
    trust_remote_code: bool = False
) -> Tuple[Union[T5ForConditionalGeneration, AutoModelForCausalLM], Union[T5Tokenizer, AutoTokenizer]]:
    """
    Load model and tokenizer with support for T5, LLaMA, Qwen, and other models.

    Args:
        model_name: Name of pretrained model (supports T5, LLaMA-3, Qwen2, etc.)
        checkpoint_path: Optional path to model checkpoint
        device: Device to load model on
        use_8bit: Use 8-bit quantization for memory efficiency
        use_4bit: Use 4-bit quantization for memory efficiency
        trust_remote_code: Trust remote code (needed for some models like Qwen)

    Returns:
        Tuple of (model, tokenizer)
    """
    # Determine model type based on model name
    is_t5 = "t5" in model_name.lower()
    is_llama = "llama" in model_name.lower()
    is_qwen = "qwen" in model_name.lower()

    # Configure quantization if requested
    quantization_config = None
    if use_8bit or use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            load_in_4bit=use_4bit,
            bnb_4bit_compute_dtype=torch.float16 if use_4bit else None,
            bnb_4bit_quant_type="nf4" if use_4bit else None,
            bnb_4bit_use_double_quant=True if use_4bit else None,
        )

    # Load tokenizer
    if is_t5:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None
        )
    else:
        # For LLaMA, Qwen, and other causal LMs
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config else None,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if not quantization_config else None
        )

        # Set pad token if not set (common for LLaMA models)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.eos_token_id

    # Load checkpoint if provided
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # Enable gradients for all parameters (if not quantized)
    if not quantization_config:
        for param in model.parameters():
            param.requires_grad = True

        # Move to device (if not using device_map)
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
