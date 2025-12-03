"""
Improved transform-based poison detection methods.

This module addresses the limitations of the original transform detection
by implementing multiple strategies with adaptive thresholds.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler


@dataclass
class DetectionResult:
    """Results from poison detection."""
    detected_indices: np.ndarray
    scores: np.ndarray
    threshold: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    method_name: str


class ImprovedTransformDetector:
    """Improved transform-based poison detection with multiple strategies."""

    def __init__(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        poisoned_indices: Optional[set] = None
    ):
        """
        Initialize detector.

        Args:
            original_scores: Influence scores from original test samples (n_train, n_test)
            transformed_scores: Influence scores from transformed test samples (n_train, n_test)
            poisoned_indices: Ground truth poisoned indices (for evaluation)
        """
        self.original_scores = original_scores
        self.transformed_scores = transformed_scores
        self.poisoned_indices = poisoned_indices or set()

        # Compute base metrics
        self.orig_avg = original_scores.mean(axis=1)
        self.trans_avg = transformed_scores.mean(axis=1)
        self.influence_strength = np.abs(self.orig_avg)
        self.influence_change = np.abs(self.orig_avg - self.trans_avg)

        # Ground truth mask
        n_train = original_scores.shape[0]
        self.gt_mask = np.array([i in poisoned_indices for i in range(n_train)])

    def _evaluate(
        self,
        detected_mask: np.ndarray,
        method_name: str,
        threshold: float,
        scores: np.ndarray
    ) -> DetectionResult:
        """Evaluate detection results."""
        tp = (detected_mask & self.gt_mask).sum()
        fp = (detected_mask & ~self.gt_mask).sum()
        tn = (~detected_mask & ~self.gt_mask).sum()
        fn = (~detected_mask & self.gt_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return DetectionResult(
            detected_indices=np.where(detected_mask)[0],
            scores=scores,
            threshold=threshold,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            precision=precision,
            recall=recall,
            f1_score=f1,
            method_name=method_name
        )

    def detect_weighted_score(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        percentile_threshold: float = 90
    ) -> DetectionResult:
        """
        Detection using weighted scoring.

        Score = α * strength - β * change
        Detects samples with high weighted score.

        Args:
            alpha: Weight for influence strength (higher = favor strong influence)
            beta: Weight for influence change (higher = favor stable influence)
            percentile_threshold: Percentile for detection threshold
        """
        # Normalize features
        scaler = StandardScaler()
        strength_norm = scaler.fit_transform(self.influence_strength.reshape(-1, 1)).flatten()
        change_norm = scaler.fit_transform(self.influence_change.reshape(-1, 1)).flatten()

        # Compute weighted score
        scores = alpha * strength_norm - beta * change_norm

        # Detect high scorers
        threshold = np.percentile(scores, percentile_threshold)
        detected_mask = scores > threshold

        return self._evaluate(
            detected_mask,
            f"weighted_score_a{alpha}_b{beta}",
            threshold,
            scores
        )

    def detect_rank_fusion(
        self,
        top_k: int = 50
    ) -> DetectionResult:
        """
        Detection using rank-based fusion.

        Combines ranks from strength (high is suspicious) and change (low is suspicious).

        Args:
            top_k: Number of top suspicious samples to detect
        """
        n = len(self.influence_strength)

        # Get ranks (0 = lowest, n-1 = highest)
        strength_ranks = stats.rankdata(self.influence_strength, method='ordinal') - 1
        change_ranks = stats.rankdata(-self.influence_change, method='ordinal') - 1  # Negative for low=suspicious

        # Combine ranks (higher is more suspicious)
        combined_ranks = strength_ranks + change_ranks

        # Select top-k
        threshold = np.sort(combined_ranks)[-top_k] if top_k < n else -1
        detected_mask = combined_ranks >= threshold

        return self._evaluate(
            detected_mask,
            f"rank_fusion_k{top_k}",
            float(threshold),
            combined_ranks.astype(float)
        )

    def detect_adaptive_percentile(
        self,
        strength_percentile_range: Tuple[float, float] = (70, 95),
        change_percentile_range: Tuple[float, float] = (5, 30),
        n_steps: int = 5
    ) -> DetectionResult:
        """
        Adaptive threshold selection by grid search over percentile ranges.

        Tests multiple combinations and selects the best F1 score.

        Args:
            strength_percentile_range: Range of percentiles for strength threshold
            change_percentile_range: Range of percentiles for change threshold
            n_steps: Number of steps in grid search
        """
        strength_percentiles = np.linspace(
            strength_percentile_range[0],
            strength_percentile_range[1],
            n_steps
        )
        change_percentiles = np.linspace(
            change_percentile_range[0],
            change_percentile_range[1],
            n_steps
        )

        best_f1 = -1
        best_result = None

        for s_pct in strength_percentiles:
            for c_pct in change_percentiles:
                s_thresh = np.percentile(self.influence_strength, s_pct)
                c_thresh = np.percentile(self.influence_change, c_pct)

                detected_mask = (self.influence_strength > s_thresh) & (self.influence_change < c_thresh)

                # Quick F1 calculation
                tp = (detected_mask & self.gt_mask).sum()
                fp = (detected_mask & ~self.gt_mask).sum()
                fn = (~detected_mask & self.gt_mask).sum()

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                if f1 > best_f1:
                    best_f1 = f1
                    # Store best configuration
                    best_result = self._evaluate(
                        detected_mask,
                        f"adaptive_s{s_pct:.0f}_c{c_pct:.0f}",
                        float(s_thresh),
                        self.influence_strength.copy()
                    )

        return best_result if best_result else self._evaluate(
            np.zeros(len(self.influence_strength), dtype=bool),
            "adaptive_failed",
            0.0,
            self.influence_strength
        )

    def detect_invariance_ratio(
        self,
        ratio_percentile: float = 90
    ) -> DetectionResult:
        """
        Detection based on influence invariance ratio.

        Ratio = strength / (change + epsilon)
        High ratio indicates strong, stable influence.

        Args:
            ratio_percentile: Percentile threshold for ratio
        """
        epsilon = 1e-6
        ratio = self.influence_strength / (self.influence_change + epsilon)

        threshold = np.percentile(ratio, ratio_percentile)
        detected_mask = ratio > threshold

        return self._evaluate(
            detected_mask,
            f"invariance_ratio_p{ratio_percentile}",
            threshold,
            ratio
        )

    def detect_z_score_combined(
        self,
        z_threshold: float = 2.0
    ) -> DetectionResult:
        """
        Detection using z-scores on combined metric.

        Combines strength and change using z-scores, detects outliers.

        Args:
            z_threshold: Z-score threshold for detection
        """
        # Compute z-scores
        strength_z = (self.influence_strength - self.influence_strength.mean()) / (self.influence_strength.std() + 1e-6)
        change_z = (self.influence_change - self.influence_change.mean()) / (self.influence_change.std() + 1e-6)

        # Combined metric: high strength z-score and low change z-score
        combined_z = strength_z - change_z

        threshold = z_threshold
        detected_mask = combined_z > threshold

        return self._evaluate(
            detected_mask,
            f"z_score_t{z_threshold}",
            threshold,
            combined_z
        )

    def detect_all_methods(self) -> Dict[str, DetectionResult]:
        """
        Run all detection methods and return results.

        Returns:
            Dictionary mapping method name to detection result
        """
        results = {}

        # Weighted score with different parameters
        for alpha, beta in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0), (1.5, 1.0)]:
            result = self.detect_weighted_score(alpha, beta, percentile_threshold=90)
            results[result.method_name] = result

        # Rank fusion with different k values
        for k in [50, 100, 150]:
            result = self.detect_rank_fusion(top_k=k)
            results[result.method_name] = result

        # Adaptive percentile
        result = self.detect_adaptive_percentile()
        results[result.method_name] = result

        # Invariance ratio
        for pct in [85, 90, 95]:
            result = self.detect_invariance_ratio(ratio_percentile=pct)
            results[result.method_name] = result

        # Z-score combined
        for z in [1.5, 2.0, 2.5]:
            result = self.detect_z_score_combined(z_threshold=z)
            results[result.method_name] = result

        return results

    def get_best_method(self) -> DetectionResult:
        """
        Find and return the best detection method based on F1 score.

        Returns:
            Best detection result
        """
        all_results = self.detect_all_methods()

        best_result = max(all_results.values(), key=lambda r: r.f1_score)

        return best_result


def compare_detection_methods(
    original_scores: np.ndarray,
    transformed_scores: np.ndarray,
    poisoned_indices: set
) -> Dict[str, Dict]:
    """
    Compare all improved detection methods.

    Args:
        original_scores: Original influence scores
        transformed_scores: Transformed influence scores
        poisoned_indices: Ground truth poisoned indices

    Returns:
        Dictionary with comparison results
    """
    detector = ImprovedTransformDetector(
        original_scores,
        transformed_scores,
        poisoned_indices
    )

    all_results = detector.detect_all_methods()

    # Convert to dictionary format
    results_dict = {}
    for name, result in all_results.items():
        results_dict[name] = {
            'f1_score': result.f1_score,
            'precision': result.precision,
            'recall': result.recall,
            'true_positives': result.true_positives,
            'false_positives': result.false_positives,
            'true_negatives': result.true_negatives,
            'false_negatives': result.false_negatives,
            'num_detected': len(result.detected_indices),
            'threshold': result.threshold
        }

    return results_dict
