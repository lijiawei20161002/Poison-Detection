"""Shared base class for all poison detectors."""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class BaseDetector:
    """
    Common base for all detector implementations.

    Subclasses inherit two shared evaluation helpers:
    - ``_compute_metrics_from_mask``: for boolean mask-based detection
    - ``evaluate_detection``: for (index, score) list-based detection
    """

    def __init__(self, poisoned_indices: Optional[Set[int]] = None) -> None:
        """
        Args:
            poisoned_indices: Ground truth poison indices for evaluation.
                              If None, evaluation helpers return zeroed metrics.
        """
        self.poisoned_indices = poisoned_indices
        self._has_ground_truth = poisoned_indices is not None

    def _compute_metrics_from_mask(
        self,
        detected_mask: np.ndarray,
        n_samples: int,
    ) -> Dict[str, float]:
        """
        Compute precision / recall / F1 from a boolean detected mask.

        Args:
            detected_mask: Boolean array of length ``n_samples``; True = detected poison.
            n_samples: Total number of training samples.

        Returns:
            Dict with f1_score, precision, recall, true_positives, false_positives,
            true_negatives, false_negatives, num_detected.
        """
        if self.poisoned_indices is None:
            return {
                "num_detected": int(detected_mask.sum()),
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
            }

        gt_mask = np.array([i in self.poisoned_indices for i in range(n_samples)])
        tp = int((detected_mask & gt_mask).sum())
        fp = int((detected_mask & ~gt_mask).sum())
        tn = int((~detected_mask & ~gt_mask).sum())
        fn = int((~detected_mask & gt_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "num_detected": int(detected_mask.sum()),
        }

    def evaluate_detection(
        self,
        detected_indices: List[Tuple[int, float]],
        total_samples: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate detection performance from a (index, score) list.

        Args:
            detected_indices: List of ``(sample_index, score)`` tuples.
            total_samples: Total training set size (needed for TN count / accuracy).
                           If None, inferred from ``self.original_scores`` when available.

        Returns:
            Dict with precision, recall, f1_score, accuracy, true_positives,
            false_positives, true_negatives, false_negatives, num_detected, num_poisoned.

        Raises:
            ValueError: If ``poisoned_indices`` was not provided at construction.
        """
        if not self._has_ground_truth:
            raise ValueError("Ground truth poisoned_indices required for evaluation.")

        detected_set = {idx for idx, _ in detected_indices}
        poisoned_set = self.poisoned_indices or set()

        if total_samples is None:
            if hasattr(self, "original_scores"):
                total_samples = len(self.original_scores)
            else:
                total_samples = max(
                    max(detected_set, default=0),
                    max(poisoned_set, default=0),
                ) + 1

        tp = len(detected_set & poisoned_set)
        fp = len(detected_set - poisoned_set)
        fn = len(poisoned_set - detected_set)
        tn = total_samples - tp - fp - fn

        precision = tp / len(detected_set) if detected_set else 0.0
        recall = tp / len(poisoned_set) if poisoned_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "num_detected": len(detected_set),
            "num_poisoned": len(poisoned_set),
        }
