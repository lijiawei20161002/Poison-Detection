"""Enhanced poison detection using ensemble methods and KL divergence."""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler
import warnings


def compute_kl_divergence(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence between two score distributions.

    Args:
        scores_1: First score distribution
        scores_2: Second score distribution
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value
    """
    # Normalize to probability distributions
    p = np.abs(scores_1) + epsilon
    q = np.abs(scores_2) + epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute KL divergence
    kl_div = entropy(p, q)

    return kl_div


def compute_js_divergence(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric version of KL).

    Args:
        scores_1: First score distribution
        scores_2: Second score distribution
        epsilon: Small value to avoid log(0)

    Returns:
        JS divergence value
    """
    # Normalize to probability distributions
    p = np.abs(scores_1) + epsilon
    q = np.abs(scores_2) + epsilon

    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute JS divergence
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)

    return js_div


def compute_score_variance(influence_scores_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute variance of influence scores across multiple transformations.

    Poisoned samples should have high variance since transformations
    change their influence differently than clean samples.

    Args:
        influence_scores_list: List of influence score arrays from different transformations

    Returns:
        Array of variances for each training sample
    """
    # Stack scores from all transformations
    # Shape: (num_transformations, num_train_samples)
    stacked_scores = np.stack([scores.mean(axis=1) for scores in influence_scores_list])

    # Compute variance across transformations for each sample
    variances = np.var(stacked_scores, axis=0)

    return variances


class EnsemblePoisonDetector:
    """Enhanced poison detector using ensemble methods and KL divergence."""

    def __init__(
        self,
        poisoned_indices: Optional[Set[int]] = None
    ):
        """
        Initialize EnsemblePoisonDetector.

        Args:
            poisoned_indices: Optional ground truth poison indices for evaluation
        """
        self.poisoned_indices = poisoned_indices
        self._has_ground_truth = poisoned_indices is not None

        # Store multiple transformation results
        self.transformation_results: Dict[str, Dict] = {}

    def add_transformation_result(
        self,
        transform_name: str,
        influence_scores: torch.Tensor,
        train_size: int
    ):
        """
        Add influence scores from a semantic transformation.

        Args:
            transform_name: Name of the transformation
            influence_scores: Influence score tensor (train_size x test_size)
            train_size: Number of training samples
        """
        # Convert to numpy and compute average influence per training sample
        scores_np = influence_scores.cpu().numpy()
        avg_scores = scores_np.mean(axis=1)  # Average across test samples

        self.transformation_results[transform_name] = {
            "influence_scores": scores_np,
            "avg_scores": avg_scores,
            "train_size": train_size
        }

    def compute_kl_scores(self, baseline_transform: str = "original") -> Dict[int, float]:
        """
        Compute KL divergence between each transformation and baseline.

        Args:
            baseline_transform: Name of baseline transformation

        Returns:
            Dictionary mapping sample index to KL divergence score
        """
        if baseline_transform not in self.transformation_results:
            raise ValueError(f"Baseline transform '{baseline_transform}' not found")

        baseline_scores = self.transformation_results[baseline_transform]["influence_scores"]
        train_size = self.transformation_results[baseline_transform]["train_size"]

        kl_scores = {}

        for idx in range(train_size):
            # Get baseline distribution for this sample
            baseline_dist = baseline_scores[idx, :]

            # Compute KL divergence with each transformation
            kl_values = []

            for transform_name, result in self.transformation_results.items():
                if transform_name == baseline_transform:
                    continue

                transform_dist = result["influence_scores"][idx, :]

                # Compute KL divergence
                kl_div = compute_kl_divergence(baseline_dist, transform_dist)
                kl_values.append(kl_div)

            # Average KL divergence across transformations
            if kl_values:
                kl_scores[idx] = np.mean(kl_values)
            else:
                kl_scores[idx] = 0.0

        return kl_scores

    def compute_variance_scores(self) -> Dict[int, float]:
        """
        Compute variance of influence scores across transformations.

        Returns:
            Dictionary mapping sample index to variance score
        """
        if not self.transformation_results:
            raise ValueError("No transformation results available")

        # Get all average scores
        all_avg_scores = [
            result["avg_scores"]
            for result in self.transformation_results.values()
        ]

        # Compute variance
        variances = compute_score_variance(
            [result["influence_scores"] for result in self.transformation_results.values()]
        )

        return {idx: float(var) for idx, var in enumerate(variances)}

    def detect_by_kl_threshold(
        self,
        threshold: float = 0.1,
        baseline_transform: str = "original"
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using KL divergence threshold.

        High KL divergence indicates influence changes significantly with
        semantic transformations, suggesting poisoned sample.

        Args:
            threshold: KL divergence threshold
            baseline_transform: Name of baseline transformation

        Returns:
            List of detected (index, kl_score) tuples
        """
        kl_scores = self.compute_kl_scores(baseline_transform)

        # Detect samples with high KL divergence
        detected = [
            (idx, score)
            for idx, score in kl_scores.items()
            if score > threshold
        ]

        # Sort by KL score (highest first)
        detected.sort(key=lambda x: x[1], reverse=True)

        return detected

    def detect_by_variance_threshold(
        self,
        threshold: float = None,
        percentile: float = 95
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using variance threshold.

        Args:
            threshold: Variance threshold (if None, use percentile)
            percentile: Percentile for automatic threshold

        Returns:
            List of detected (index, variance) tuples
        """
        variance_scores = self.compute_variance_scores()

        # Determine threshold
        if threshold is None:
            variances = list(variance_scores.values())
            threshold = np.percentile(variances, percentile)

        # Detect samples with high variance
        detected = [
            (idx, score)
            for idx, score in variance_scores.items()
            if score > threshold
        ]

        # Sort by variance (highest first)
        detected.sort(key=lambda x: x[1], reverse=True)

        return detected

    def detect_by_ensemble_voting(
        self,
        k: int = 10,
        voting_threshold: int = None
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using ensemble voting.

        Each transformation votes for top-k most suspicious samples.
        Samples receiving votes from multiple transformations are flagged.

        Args:
            k: Number of top samples each transformation votes for
            voting_threshold: Minimum votes required (default: majority)

        Returns:
            List of detected (index, vote_count) tuples
        """
        if not self.transformation_results:
            raise ValueError("No transformation results available")

        num_transforms = len(self.transformation_results)
        if voting_threshold is None:
            voting_threshold = max(2, num_transforms // 2)

        # Collect votes from each transformation
        vote_counts = {}

        for transform_name, result in self.transformation_results.items():
            avg_scores = result["avg_scores"]

            # Get top-k samples with lowest influence (most suspicious)
            top_k_indices = np.argsort(avg_scores)[:k]

            for idx in top_k_indices:
                vote_counts[idx] = vote_counts.get(idx, 0) + 1

        # Detect samples with enough votes
        detected = [
            (idx, float(votes))
            for idx, votes in vote_counts.items()
            if votes >= voting_threshold
        ]

        # Sort by vote count (highest first)
        detected.sort(key=lambda x: x[1], reverse=True)

        return detected

    def detect_by_combined_score(
        self,
        kl_weight: float = 0.4,
        variance_weight: float = 0.3,
        influence_weight: float = 0.3,
        threshold_percentile: float = 90
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using weighted combination of multiple metrics.

        Args:
            kl_weight: Weight for KL divergence score
            variance_weight: Weight for variance score
            influence_weight: Weight for influence score (low influence = suspicious)
            threshold_percentile: Percentile threshold for detection

        Returns:
            List of detected (index, combined_score) tuples
        """
        # Get all scores
        kl_scores = self.compute_kl_scores()
        variance_scores = self.compute_variance_scores()

        # Get average influence scores (inverted - low influence is suspicious)
        baseline_result = list(self.transformation_results.values())[0]
        avg_influence = baseline_result["avg_scores"]
        # Invert: low influence -> high suspicion score
        influence_suspicion = -avg_influence

        # Normalize all scores to [0, 1]
        scaler = MinMaxScaler()

        kl_values = np.array(list(kl_scores.values())).reshape(-1, 1)
        kl_normalized = scaler.fit_transform(kl_values).flatten()

        var_values = np.array(list(variance_scores.values())).reshape(-1, 1)
        var_normalized = scaler.fit_transform(var_values).flatten()

        inf_values = influence_suspicion.reshape(-1, 1)
        inf_normalized = scaler.fit_transform(inf_values).flatten()

        # Compute combined score
        combined_scores = {}
        for idx in range(len(kl_normalized)):
            combined_scores[idx] = (
                kl_weight * kl_normalized[idx] +
                variance_weight * var_normalized[idx] +
                influence_weight * inf_normalized[idx]
            )

        # Determine threshold
        threshold = np.percentile(list(combined_scores.values()), threshold_percentile)

        # Detect samples above threshold
        detected = [
            (idx, score)
            for idx, score in combined_scores.items()
            if score > threshold
        ]

        # Sort by combined score (highest first)
        detected.sort(key=lambda x: x[1], reverse=True)

        return detected

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
        if not self._has_ground_truth:
            raise ValueError("Ground truth poisoned indices required for evaluation")

        detected_set = {idx for idx, _ in detected_indices}
        poisoned_set = self.poisoned_indices if self.poisoned_indices else set()

        # Get total number of samples
        if self.transformation_results:
            total_samples = list(self.transformation_results.values())[0]["train_size"]
        else:
            total_samples = max(max(detected_set, default=0), max(poisoned_set, default=0)) + 1

        # Calculate metrics
        true_positives = len(detected_set & poisoned_set)
        false_positives = len(detected_set - poisoned_set)
        false_negatives = len(poisoned_set - detected_set)
        true_negatives = total_samples - true_positives - false_positives - false_negatives

        precision = true_positives / len(detected_set) if detected_set else 0
        recall = true_positives / len(poisoned_set) if poisoned_set else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0

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
            "num_poisoned": len(poisoned_set)
        }

    def get_detection_summary(self) -> Dict:
        """
        Get summary of all detection methods.

        Returns:
            Dictionary with results from all methods
        """
        summary = {}

        try:
            # KL-based detection
            kl_detected = self.detect_by_kl_threshold()
            if self._has_ground_truth:
                summary["kl_divergence"] = self.evaluate_detection(kl_detected)
            else:
                summary["kl_divergence"] = {"num_detected": len(kl_detected)}
        except Exception as e:
            summary["kl_divergence"] = {"error": str(e)}

        try:
            # Variance-based detection
            var_detected = self.detect_by_variance_threshold()
            if self._has_ground_truth:
                summary["variance"] = self.evaluate_detection(var_detected)
            else:
                summary["variance"] = {"num_detected": len(var_detected)}
        except Exception as e:
            summary["variance"] = {"error": str(e)}

        try:
            # Voting-based detection
            vote_detected = self.detect_by_ensemble_voting()
            if self._has_ground_truth:
                summary["voting"] = self.evaluate_detection(vote_detected)
            else:
                summary["voting"] = {"num_detected": len(vote_detected)}
        except Exception as e:
            summary["voting"] = {"error": str(e)}

        try:
            # Combined score detection
            combined_detected = self.detect_by_combined_score()
            if self._has_ground_truth:
                summary["combined"] = self.evaluate_detection(combined_detected)
            else:
                summary["combined"] = {"num_detected": len(combined_detected)}
        except Exception as e:
            summary["combined"] = {"error": str(e)}

        return summary
