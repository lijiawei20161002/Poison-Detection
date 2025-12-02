"""Loss-based poison detection as alternative to influence-based methods."""

import numpy as np
import torch
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader


class LossBasedDetector:
    """
    Detect poisons by analyzing per-sample loss patterns.

    Poisoned samples often have distinctive loss characteristics:
    - Unusually low/high loss
    - Different loss trajectory during training
    - High loss variance across epochs
    """

    def __init__(self, model, task, device='cuda'):
        self.model = model
        self.task = task
        self.device = device

    def compute_per_sample_losses(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute loss for each sample in the dataset.

        Args:
            dataloader: DataLoader with samples
            num_samples: Expected number of samples (for pre-allocation)

        Returns:
            Array of per-sample losses
        """
        self.model.eval()
        losses = []

        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                if isinstance(batch, (tuple, list)):
                    batch = tuple(b.to(self.device) if isinstance(b, torch.Tensor) else b
                                 for b in batch)
                elif isinstance(batch, dict):
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                # Compute loss for each sample in batch
                batch_losses = self.task.compute_train_loss(
                    self.model,
                    batch,
                    reduction='none'  # Per-sample losses
                )

                losses.extend(batch_losses.cpu().numpy())

        return np.array(losses)

    def detect_by_loss_percentile(
        self,
        losses: np.ndarray,
        percentile_low: float = 5.0,
        percentile_high: float = 95.0
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers by loss percentiles.

        Args:
            losses: Per-sample losses
            percentile_low: Lower percentile threshold
            percentile_high: Upper percentile threshold

        Returns:
            List of (index, loss) tuples for detected samples
        """
        low_threshold = np.percentile(losses, percentile_low)
        high_threshold = np.percentile(losses, percentile_high)

        outliers = []
        for idx, loss in enumerate(losses):
            if loss <= low_threshold or loss >= high_threshold:
                outliers.append((idx, float(loss)))

        return outliers

    def detect_by_loss_zscore(
        self,
        losses: np.ndarray,
        z_threshold: float = 2.5
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers using z-score on losses.

        Args:
            losses: Per-sample losses
            z_threshold: Z-score threshold

        Returns:
            List of (index, loss) tuples for detected samples
        """
        from scipy.stats import zscore

        z_scores = np.abs(zscore(losses))
        outliers = []

        for idx, (loss, z) in enumerate(zip(losses, z_scores)):
            if z > z_threshold:
                outliers.append((idx, float(loss)))

        return outliers

    def detect_by_gradient_norm(
        self,
        dataloader: DataLoader,
        percentile: float = 95.0
    ) -> List[Tuple[int, float]]:
        """
        Detect samples with unusually high gradient norms.

        Poisoned samples often have higher gradients as the model
        tries to fit the backdoor pattern.

        Args:
            dataloader: DataLoader with samples
            percentile: Percentile threshold for high gradients

        Returns:
            List of (index, gradient_norm) tuples
        """
        self.model.train()
        gradient_norms = []

        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, (tuple, list)):
                batch = tuple(b.to(self.device) if isinstance(b, torch.Tensor) else b
                             for b in batch)
            elif isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

            # Zero gradients
            self.model.zero_grad()

            # Compute loss and gradients
            loss = self.task.compute_train_loss(self.model, batch, reduction='mean')
            loss.backward()

            # Compute gradient norm for this batch
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            gradient_norms.append(total_norm)

        # Detect high gradient samples
        threshold = np.percentile(gradient_norms, percentile)
        outliers = [
            (idx, float(norm))
            for idx, norm in enumerate(gradient_norms)
            if norm >= threshold
        ]

        return outliers
