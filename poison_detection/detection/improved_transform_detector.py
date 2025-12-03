"""
Improved transformation-based poison detection.

This module implements enhanced detection methods that address the failures
of basic percentile-based thresholding.
"""

import numpy as np
from typing import Dict, Tuple, Set
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


class ImprovedTransformDetector:
    """Improved transformation-based detection with multiple strategies."""

    def __init__(self, poisoned_indices: Set[int]):
        self.poisoned_indices = poisoned_indices

    def _compute_metrics(self, detected_mask: np.ndarray, n_train: int) -> Dict:
        """Compute detection metrics."""
        gt_mask = np.array([i in self.poisoned_indices for i in range(n_train)])

        tp = (detected_mask & gt_mask).sum()
        fp = (detected_mask & ~gt_mask).sum()
        tn = (~detected_mask & ~gt_mask).sum()
        fn = (~detected_mask & gt_mask).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'num_detected': int(detected_mask.sum())
        }

    def detect_iqr_method(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        k: float = 1.5
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect using Interquartile Range (IQR) method.

        More robust to outliers than mean+std. Detects samples with:
        - High influence strength (above Q3 + k*IQR)
        - Low influence change (below Q1 - k*IQR or within tight range)

        Args:
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)
            k: IQR multiplier (default 1.5 for outliers, 3.0 for extreme outliers)
        """
        logger.info(f"Running IQR detection with k={k}...")

        # Average influence across test samples
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        # Compute metrics
        influence_strength = np.abs(orig_avg)
        influence_change = np.abs(orig_avg - trans_avg)

        # IQR-based thresholds for strength (want high outliers)
        q1_strength, q3_strength = np.percentile(influence_strength, [25, 75])
        iqr_strength = q3_strength - q1_strength
        strength_threshold = q3_strength + k * iqr_strength

        # IQR-based thresholds for change (want low outliers)
        q1_change, q3_change = np.percentile(influence_change, [25, 75])
        iqr_change = q3_change - q1_change
        change_threshold = q1_change - k * iqr_change
        change_threshold = max(change_threshold, 0)  # Can't be negative

        # Alternative: use median absolute deviation for change
        median_change = np.median(influence_change)
        mad_change = np.median(np.abs(influence_change - median_change))
        change_threshold_mad = median_change - k * 1.4826 * mad_change  # 1.4826 makes MAD consistent with std
        change_threshold_mad = max(change_threshold_mad, 0)

        # Detect: high strength OR low change
        detected_mask = (influence_strength > strength_threshold) | (influence_change < change_threshold_mad)

        metrics = self._compute_metrics(detected_mask, len(original_scores))
        metrics['strength_threshold'] = float(strength_threshold)
        metrics['change_threshold'] = float(change_threshold_mad)
        metrics['method'] = f'iqr_k{k}'

        logger.info(f"IQR detection: F1={metrics['f1_score']:.4f}, "
                   f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def detect_relative_change(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        percentile_high: float = 90,
        percentile_low: float = 10
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect using relative influence change instead of absolute.

        Relative change = |orig - trans| / (|orig| + epsilon)
        This normalizes for magnitude differences.

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores
            percentile_high: Percentile for high strength threshold
            percentile_low: Percentile for low change threshold
        """
        logger.info("Running relative change detection...")

        # Average influence
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        # Compute relative change
        influence_strength = np.abs(orig_avg)
        abs_change = np.abs(orig_avg - trans_avg)
        relative_change = abs_change / (influence_strength + 1e-8)  # Add epsilon to avoid division by zero

        # Thresholds
        strength_threshold = np.percentile(influence_strength, percentile_high)
        change_threshold = np.percentile(relative_change, percentile_low)

        # Detect: high strength AND low relative change
        detected_mask = (influence_strength > strength_threshold) & (relative_change < change_threshold)

        metrics = self._compute_metrics(detected_mask, len(original_scores))
        metrics['strength_threshold'] = float(strength_threshold)
        metrics['change_threshold'] = float(change_threshold)
        metrics['method'] = 'relative_change'

        logger.info(f"Relative change detection: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def detect_isolation_forest_2d(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        contamination: float = 0.05
    ) -> Tuple[Dict, np.ndarray]:
        """
        Use Isolation Forest on 2D feature space (strength, change).

        This treats detection as an unsupervised anomaly detection problem
        in the joint space of influence strength and change.

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores
            contamination: Expected proportion of outliers (default 0.05 = 5%)
        """
        logger.info(f"Running 2D Isolation Forest detection (contamination={contamination})...")

        # Compute features
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        influence_strength = np.abs(orig_avg)
        influence_change = np.abs(orig_avg - trans_avg)

        # Stack features
        X = np.column_stack([influence_strength, influence_change])

        # Fit Isolation Forest
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(X)

        # -1 for outliers, 1 for inliers
        detected_mask = predictions == -1

        metrics = self._compute_metrics(detected_mask, len(original_scores))
        metrics['contamination'] = contamination
        metrics['method'] = 'isolation_forest_2d'

        logger.info(f"Isolation Forest 2D: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def detect_dbscan_2d(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 5
    ) -> Tuple[Dict, np.ndarray]:
        """
        Use DBSCAN clustering on 2D feature space.

        Detects outliers as points that don't belong to any cluster (noise points).

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores
            eps: Maximum distance between two samples for one to be in the neighborhood of the other
            min_samples: Minimum number of samples in a neighborhood for a point to be a core point
        """
        logger.info(f"Running 2D DBSCAN detection (eps={eps}, min_samples={min_samples})...")

        # Compute features
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        influence_strength = np.abs(orig_avg)
        influence_change = np.abs(orig_avg - trans_avg)

        # Normalize features to similar scales
        strength_norm = (influence_strength - influence_strength.mean()) / (influence_strength.std() + 1e-8)
        change_norm = (influence_change - influence_change.mean()) / (influence_change.std() + 1e-8)

        X = np.column_stack([strength_norm, change_norm])

        # Fit DBSCAN
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # -1 indicates noise/outliers
        detected_mask = clustering.labels_ == -1

        metrics = self._compute_metrics(detected_mask, len(original_scores))
        metrics['eps'] = eps
        metrics['min_samples'] = min_samples
        metrics['n_clusters'] = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        metrics['method'] = 'dbscan_2d'

        logger.info(f"DBSCAN 2D: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}, Clusters={metrics['n_clusters']}")

        return metrics, detected_mask

    def detect_zscore_combined(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        z_threshold: float = 2.0
    ) -> Tuple[Dict, np.ndarray]:
        """
        Use Z-score based detection on combined metric.

        Creates a combined suspicious score based on:
        - High influence strength (positive contribution)
        - Low influence change (positive contribution)

        Then applies Z-score outlier detection on the combined score.

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores
            z_threshold: Z-score threshold for outliers (default 2.0)
        """
        logger.info(f"Running Z-score combined detection (threshold={z_threshold})...")

        # Compute features
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        influence_strength = np.abs(orig_avg)
        influence_change = np.abs(orig_avg - trans_avg)

        # Normalize to [0, 1] range
        strength_norm = (influence_strength - influence_strength.min()) / (influence_strength.max() - influence_strength.min() + 1e-8)
        change_norm = (influence_change - influence_change.min()) / (influence_change.max() - influence_change.min() + 1e-8)

        # Combined suspicious score: high strength, low change
        suspicious_score = strength_norm - change_norm

        # Z-score outlier detection
        z_scores = np.abs(stats.zscore(suspicious_score))
        detected_mask = z_scores > z_threshold

        metrics = self._compute_metrics(detected_mask, len(original_scores))
        metrics['z_threshold'] = z_threshold
        metrics['method'] = 'zscore_combined'

        logger.info(f"Z-score combined: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def run_all_methods(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray
    ) -> Dict[str, Tuple[Dict, np.ndarray]]:
        """
        Run all improved detection methods and return results.

        Returns:
            Dictionary mapping method name to (metrics, detected_mask) tuple
        """
        results = {}

        # IQR methods with different k values
        for k in [1.5, 2.0, 3.0]:
            try:
                metrics, mask = self.detect_iqr_method(original_scores, transformed_scores, k=k)
                results[f'iqr_k{k}'] = (metrics, mask)
            except Exception as e:
                logger.warning(f"IQR k={k} failed: {e}")

        # Relative change method
        try:
            metrics, mask = self.detect_relative_change(original_scores, transformed_scores)
            results['relative_change'] = (metrics, mask)
        except Exception as e:
            logger.warning(f"Relative change failed: {e}")

        # Isolation Forest with different contamination levels
        for cont in [0.05, 0.1, 0.15]:
            try:
                metrics, mask = self.detect_isolation_forest_2d(original_scores, transformed_scores, contamination=cont)
                results[f'iforest_2d_{int(cont*100)}'] = (metrics, mask)
            except Exception as e:
                logger.warning(f"Isolation Forest contamination={cont} failed: {e}")

        # DBSCAN with different parameters
        for eps in [0.3, 0.5, 0.7]:
            try:
                metrics, mask = self.detect_dbscan_2d(original_scores, transformed_scores, eps=eps)
                results[f'dbscan_eps{int(eps*10)}'] = (metrics, mask)
            except Exception as e:
                logger.warning(f"DBSCAN eps={eps} failed: {e}")

        # Z-score combined
        for z in [1.5, 2.0, 2.5]:
            try:
                metrics, mask = self.detect_zscore_combined(original_scores, transformed_scores, z_threshold=z)
                results[f'zscore_z{int(z*10)}'] = (metrics, mask)
            except Exception as e:
                logger.warning(f"Z-score z={z} failed: {e}")

        return results
