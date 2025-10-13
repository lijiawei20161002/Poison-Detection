"""Task definitions for Kronfluence analysis."""

import torch
from typing import Tuple
from kronfluence.task import Task

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

        # Forward pass through model
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

        return loss

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: torch.nn.Module
    ) -> torch.Tensor:
        """
        Compute measurement loss for influence computation.

        This computes a classification loss based on the label space,
        comparing model predictions across candidate labels.

        Args:
            batch: Tuple of (inputs, labels, label_spaces)
            model: Model to compute loss with

        Returns:
            Scalar measurement loss
        """
        inputs, labels, label_spaces = batch

        inputs = inputs.long().to(self.device)
        labels = labels.long().to(self.device)
        label_spaces = label_spaces.to(self.device)

        log_probs = []

        # Compute log probabilities for each sample
        for i in range(inputs.size(0)):
            input_ids = inputs[i].unsqueeze(0)
            current_label_space = label_spaces[i]

            candidate_losses = []
            for candidate in current_label_space:
                # Compute loss for each candidate label
                outputs = model(
                    input_ids=input_ids,
                    labels=candidate.unsqueeze(0)
                )

                # Use negative loss as log probability
                log_prob = -outputs.loss
                candidate_losses.append(log_prob)

            log_probs.append(torch.stack(candidate_losses, dim=0))

        # Stack log probs: [batch_size, num_candidates]
        log_probs = torch.stack(log_probs)

        # Find true label indices in label space
        true_label_indices = []
        for i in range(labels.size(0)):
            true_label_token = labels[i]
            label_space = label_spaces[i][1]  # Get label space for current sample

            # Find index of true label
            idx = (label_space == true_label_token).nonzero(as_tuple=False)

            if idx.numel() > 0:
                idx = idx[0].item()
            else:
                # If true label not found, use first candidate
                idx = 0

            true_label_indices.append(idx)

        true_label_indices = torch.tensor(
            true_label_indices,
            dtype=torch.long
        ).to(self.device)

        # Convert to probabilities with softmax
        probs = torch.softmax(log_probs, dim=1)

        # Create target tensor (one-hot encoding)
        target_tensor = torch.zeros_like(probs)
        target_tensor[range(target_tensor.size(0)), true_label_indices] = 1.0

        # Compute Binary Cross-Entropy loss
        loss_fn = torch.nn.BCELoss()
        loss = loss_fn(probs, target_tensor)

        return loss


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
