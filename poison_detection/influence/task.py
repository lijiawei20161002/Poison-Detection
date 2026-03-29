"""Task definitions for Kronfluence analysis."""

import torch
from typing import Tuple
from kronfluence.task import Task

from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Type definition for batch data
BATCH_TYPE = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ClassificationTask(Task):
    """Task definition for instruction-tuned classification models."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize ClassificationTask.

        Args:
            device: Device to run computations on
        """
        self.device = device

    def preprocess_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        label_spaces: torch.Tensor
    ) -> BATCH_TYPE:
        """
        Preprocess batch by moving to device.

        Args:
            inputs: Input token IDs
            labels: Label token IDs
            label_spaces: Label space options

        Returns:
            Tuple of preprocessed tensors
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        label_spaces = label_spaces.to(self.device)
        return inputs, labels, label_spaces

    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: torch.nn.Module,
        sample: bool = False
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Args:
            batch: Tuple of (inputs, labels, label_spaces)
            model: Model to compute loss with
            sample: Whether to sample (for stochastic estimation)

        Returns:
            Scalar loss tensor
        """
        inputs, labels, label_spaces = batch

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # For causal LM: concatenate input and label sequences
        # Mask input tokens with -100 so loss is only computed on output tokens
        batch_size = inputs.size(0)

        # Concatenate inputs and labels
        combined_input_ids = torch.cat([inputs, labels], dim=1)

        # Create labels tensor: -100 for input portion, actual labels for output portion
        combined_labels = torch.cat([
            torch.full_like(inputs, -100),  # Mask input tokens
            labels  # Keep label tokens
        ], dim=1)

        # Forward pass through model
        outputs = model(input_ids=combined_input_ids, labels=combined_labels)
        loss = outputs.loss

        # Check for NaN/inf and replace with a safe value
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/inf detected in loss, replacing with 1.0")
            loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)

        # Clamp loss to reasonable range to prevent numerical instability
        loss = torch.clamp(loss, min=1e-7, max=100.0)

        return loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Compute measurement loss for influence computation.

        Uses the same causal LM cross-entropy loss as compute_train_loss so
        that both sides of the influence formula
            I(z_train, z_query) = ∇L(z_query)ᵀ H⁻¹ ∇L(z_train)
        are gradients of the same function.  The previous BCE-over-candidates
        implementation used a different loss surface, which made the inner
        product theoretically unsound.

        Args:
            batch: Tuple of (inputs, labels, label_spaces)
            model: Model to compute loss with

        Returns:
            Scalar measurement loss
        """
        return self.compute_train_loss(batch, model, sample=False)


class SimpleGenerationTask(Task):
    """Simplified task definition for generation models without label spaces."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize SimpleGenerationTask.

        Args:
            device: Device to run computations on
        """
        self.device = device

    def preprocess_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess batch by moving to device.

        Args:
            inputs: Input token IDs
            labels: Label token IDs

        Returns:
            Tuple of preprocessed tensors
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        return inputs, labels

    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: torch.nn.Module,
        sample: bool = False
    ) -> torch.Tensor:
        """
        Compute training loss for a batch.

        Args:
            batch: Tuple of (inputs, labels)
            model: Model to compute loss with
            sample: Whether to sample (for stochastic estimation)

        Returns:
            Scalar loss tensor
        """
        inputs, labels = batch

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        # Forward pass through model
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        # Check for NaN/inf and replace with a safe value
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/inf detected in loss, replacing with 1.0")
            loss = torch.tensor(1.0, device=loss.device, dtype=loss.dtype)

        # Clamp loss to reasonable range to prevent numerical instability
        loss = torch.clamp(loss, min=1e-7, max=100.0)

        return loss

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Compute measurement loss (same as training loss for generation).

        Args:
            batch: Tuple of (inputs, labels)
            model: Model to compute loss with

        Returns:
            Scalar measurement loss
        """
        return self.compute_train_loss(batch, model)
