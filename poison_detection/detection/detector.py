"""Poison detection using influence scores."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from scipy.stats import zscore


class PoisonDetector:
    """Detect poisoned samples using influence scores."""

    def __init__(
        self,
        original_scores: List[Tuple[int, float]],
        negative_scores: Optional[List[Tuple[int, float]]] = None,
        poisoned_indices: Optional[Set[int]] = None
    ):
        """
        Initialize PoisonDetector.

        Args:
            original_scores: List of (index, score) tuples from original test
            negative_scores: Optional list of (index, score) from negative test
            poisoned_indices: Optional ground truth poison indices for evaluation
        """
        self.original_scores = original_scores
        self.negative_scores = negative_scores
        self.poisoned_indices = poisoned_indices or set()

    @staticmethod
    def normalize_scores(scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Normalize influence scores using MinMax scaling.

        Args:
            scores: List of (index, score) tuples

        Returns:
            List of (index, normalized_score) tuples
        """
        score_values = np.array([score for _, score in scores])
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(score_values.reshape(-1, 1)).flatten()
        return [(idx, norm_score) for (idx, _), norm_score in zip(scores, normalized)]

    @staticmethod
    def log_transform_scores(scores: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """
        Apply log transformation to scores.

        Args:
            scores: List of (index, score) tuples

        Returns:
            List of (index, log_score) tuples
        """
        return [(idx, np.log1p(score)) for idx, score in scores]

    def detect_by_threshold(
        self,
        threshold: float = 0.5,
        use_log_transform: bool = True,
        use_normalization: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using simple threshold on influence scores.

        Args:
            threshold: Threshold value for detection
            use_log_transform: Whether to apply log transformation
            use_normalization: Whether to normalize scores

        Returns:
            List of detected poison (index, score) tuples
        """
        scores = self.original_scores

        if use_normalization:
            scores = self.normalize_scores(scores)
        if use_log_transform:
            scores = self.log_transform_scores(scores)

        # Detect samples with scores below threshold (low influence = suspicious)
        outliers = [(idx, score) for idx, score in scores if abs(score) < threshold]

        return outliers

    def detect_by_delta_scores(
        self,
        positive_threshold: float = 0,
        negative_threshold: float = 0
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using delta between original and negative scores.

        Poisoned samples typically have positive influence on original test
        but negative influence after negative transformation.

        Args:
            positive_threshold: Minimum threshold for original score
            negative_threshold: Maximum threshold for negative score

        Returns:
            List of detected poison (index, score) tuples
        """
        if self.negative_scores is None:
            raise ValueError("Negative scores required for delta detection")

        outliers = []
        for (idx1, score1), (idx2, score2) in zip(
            self.original_scores, self.negative_scores
        ):
            if idx1 != idx2:
                raise ValueError("Mismatched indices between score lists")

            # Detect samples with high positive influence but low negative influence
            if score1 > positive_threshold and score2 < negative_threshold:
                outliers.append((idx1, score1))

        return outliers

    def detect_by_zscore(
        self,
        z_threshold: float = 2.0,
        use_absolute: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Detect outliers using Z-score analysis.

        Args:
            z_threshold: Z-score threshold for outlier detection
            use_absolute: Whether to use absolute Z-scores

        Returns:
            List of detected poison (index, score) tuples
        """
        scores = np.array([score for _, score in self.original_scores])
        z_scores = zscore(scores)

        outliers = []
        for (idx, score), z in zip(self.original_scores, z_scores):
            z_val = abs(z) if use_absolute else z
            if z_val > z_threshold:
                outliers.append((idx, score))

        return outliers

    def detect_by_clustering(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        return_outliers: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using DBSCAN clustering.

        Args:
            eps: Maximum distance between samples in same neighborhood
            min_samples: Minimum samples in neighborhood for core point
            return_outliers: If True, return outlier cluster; else return main cluster

        Returns:
            List of detected poison (index, score) tuples
        """
        scores = np.array([score for _, score in self.original_scores]).reshape(-1, 1)

        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scores)
        labels = clustering.labels_

        # Outliers are labeled as -1
        if return_outliers:
            mask = labels == -1
        else:
            # Return largest cluster
            unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique_labels) > 0:
                main_label = unique_labels[np.argmax(counts)]
                mask = labels == main_label
            else:
                mask = np.zeros(len(labels), dtype=bool)

        detected = [
            (idx, score)
            for (idx, score), is_detected in zip(self.original_scores, mask)
            if is_detected
        ]

        return detected

    def calculate_delta_scores(self) -> List[Tuple[int, float]]:
        """
        Calculate difference between original and negative scores.

        Returns:
            List of (index, delta_score) tuples
        """
        if self.negative_scores is None:
            raise ValueError("Negative scores required for delta calculation")

        delta_scores = []
        for (idx1, score1), (idx2, score2) in zip(
            self.original_scores, self.negative_scores
        ):
            if idx1 != idx2:
                raise ValueError("Mismatched indices between score lists")
            delta_scores.append((idx1, score1 - score2))

        return delta_scores

    def get_top_k_suspicious(
        self,
        k: int,
        method: str = "lowest_influence"
    ) -> List[Tuple[int, float]]:
        """
        Get top K most suspicious samples.

        Args:
            k: Number of samples to return
            method: Method for ranking ("lowest_influence", "highest_influence", "delta")

        Returns:
            List of top K (index, score) tuples
        """
        if method == "lowest_influence":
            scores = sorted(self.original_scores, key=lambda x: x[1])
            return scores[:k]
        elif method == "highest_influence":
            scores = sorted(self.original_scores, key=lambda x: x[1], reverse=True)
            return scores[:k]
        elif method == "delta":
            delta_scores = self.calculate_delta_scores()
            scores = sorted(delta_scores, key=lambda x: x[1], reverse=True)
            return scores[:k]
        else:
            raise ValueError(f"Unknown method: {method}")

    def evaluate_detection(
        self,
        detected_indices: List[Tuple[int, float]]
    ) -> Dict[str, float]:
        """
        Evaluate detection performance against ground truth.

        Args:
            detected_indices: List of detected (index, score) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.poisoned_indices:
            raise ValueError("Ground truth poisoned indices required for evaluation")

        detected_set = {idx for idx, _ in detected_indices}

        # Calculate metrics
        true_positives = len(detected_set & self.poisoned_indices)
        false_positives = len(detected_set - self.poisoned_indices)
        false_negatives = len(self.poisoned_indices - detected_set)
        true_negatives = len(self.original_scores) - true_positives - false_positives - false_negatives

        precision = true_positives / len(detected_set) if detected_set else 0
        recall = true_positives / len(self.poisoned_indices) if self.poisoned_indices else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / len(self.original_scores)

        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "num_detected": len(detected_set),
            "num_poisoned": len(self.poisoned_indices)
        }

    def save_detected_indices(
        self,
        detected_indices: List[Tuple[int, float]],
        output_path: Path
    ) -> None:
        """
        Save detected poison indices to file.

        Args:
            detected_indices: List of detected (index, score) tuples
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for idx, score in detected_indices:
                f.write(f"{idx}: {score}\n")
