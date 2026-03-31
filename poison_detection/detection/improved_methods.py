"""Improved transformation-based detection with robust statistical methods."""

import numpy as np
from typing import Dict, Optional, Set, Tuple
import logging
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

from poison_detection.detection._base import BaseDetector

logger = logging.getLogger(__name__)


class ImprovedTransformDetector(BaseDetector):
    """
    Enhanced detection using IQR, relative change, Isolation Forest, DBSCAN, and Z-score.

    Addresses failures of basic percentile-based thresholding by using:
    - Robust IQR outlier bounds (less sensitive to extreme outliers)
    - Relative influence change (normalised for magnitude differences)
    - 2-D anomaly detection in (strength, change) feature space
    """

    def __init__(self, poisoned_indices: Optional[Set[int]] = None) -> None:
        super().__init__(poisoned_indices)

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_iqr_method(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        k: float = 1.5,
    ) -> Tuple[Dict, np.ndarray]:
        """
        IQR-based outlier detection.

        Detects samples with high influence strength (above Q3+k·IQR) OR low
        influence change relative to the MAD (robust to outliers).

        Args:
            original_scores: ``(n_train, n_test)``
            transformed_scores: ``(n_train, n_test)``
            k: IQR / MAD multiplier. Use 1.5 for standard outliers, 3.0 for extreme.
        """
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        strength = np.abs(orig_avg)
        change = np.abs(orig_avg - trans_avg)

        q1_s, q3_s = np.percentile(strength, [25, 75])
        strength_thr = q3_s + k * (q3_s - q1_s)

        median_c = np.median(change)
        mad_c = np.median(np.abs(change - median_c))
        change_thr = max(0.0, median_c - k * 1.4826 * mad_c)

        mask = (strength > strength_thr) | (change < change_thr)
        metrics = self._compute_metrics_from_mask(mask, len(original_scores))
        metrics.update({
            "strength_threshold": float(strength_thr),
            "change_threshold": float(change_thr),
            "method": f"iqr_k{k}",
        })
        logger.info(
            f"IQR k={k}: F1={metrics['f1_score']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"Detected={metrics['num_detected']}"
        )
        return metrics, mask

    def detect_relative_change(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        percentile_high: float = 90,
        percentile_low: float = 10,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect via normalised influence change.

        Flags samples with *high strength* AND *low relative change* — these
        are strongly influential samples whose influence signature is invariant
        to semantic transformation, a hallmark of backdoor triggers.
        """
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        strength = np.abs(orig_avg)
        rel_change = np.abs(orig_avg - trans_avg) / (strength + 1e-8)

        strength_thr = float(np.percentile(strength, percentile_high))
        change_thr = float(np.percentile(rel_change, percentile_low))

        mask = (strength > strength_thr) & (rel_change < change_thr)
        metrics = self._compute_metrics_from_mask(mask, len(original_scores))
        metrics.update({
            "strength_threshold": strength_thr,
            "change_threshold": change_thr,
            "method": "relative_change",
        })
        logger.info(
            f"Relative change: F1={metrics['f1_score']:.4f}, "
            f"Detected={metrics['num_detected']}"
        )
        return metrics, mask

    def detect_isolation_forest_2d(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        contamination: float = 0.05,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Isolation Forest on the 2-D ``(strength, change)`` feature space.

        Args:
            contamination: Expected fraction of outliers.
        """
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        X = np.column_stack([np.abs(orig_avg), np.abs(orig_avg - trans_avg)])

        clf = IsolationForest(contamination=contamination, random_state=42)
        mask = clf.fit_predict(X) == -1

        metrics = self._compute_metrics_from_mask(mask, len(original_scores))
        metrics.update({"contamination": contamination, "method": "isolation_forest_2d"})
        logger.info(
            f"Isolation Forest 2D: F1={metrics['f1_score']:.4f}, "
            f"Detected={metrics['num_detected']}"
        )
        return metrics, mask

    def detect_dbscan_2d(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        eps: float = 0.3,
        min_samples: int = 5,
    ) -> Tuple[Dict, np.ndarray]:
        """
        DBSCAN on the z-normalised ``(strength, change)`` feature space.

        Points labelled ``-1`` (noise) are treated as poisoned.
        """
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        strength = np.abs(orig_avg)
        change = np.abs(orig_avg - trans_avg)

        s_norm = (strength - strength.mean()) / (strength.std() + 1e-8)
        c_norm = (change - change.mean()) / (change.std() + 1e-8)
        X = np.column_stack([s_norm, c_norm])

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(X).labels_
        mask = labels == -1

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        metrics = self._compute_metrics_from_mask(mask, len(original_scores))
        metrics.update({"eps": eps, "min_samples": min_samples, "n_clusters": n_clusters, "method": "dbscan_2d"})
        logger.info(
            f"DBSCAN 2D: F1={metrics['f1_score']:.4f}, "
            f"Detected={metrics['num_detected']}, Clusters={n_clusters}"
        )
        return metrics, mask

    def detect_zscore_combined(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        z_threshold: float = 2.0,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Z-score outlier detection on a combined suspicious score.

        Combined score = normalised strength − normalised change.
        High combined score → high influence, low change → suspicious.
        """
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        strength = np.abs(orig_avg)
        change = np.abs(orig_avg - trans_avg)

        s_norm = (strength - strength.min()) / (strength.max() - strength.min() + 1e-8)
        c_norm = (change - change.min()) / (change.max() - change.min() + 1e-8)
        suspicious = s_norm - c_norm

        mask = np.abs(stats.zscore(suspicious)) > z_threshold

        metrics = self._compute_metrics_from_mask(mask, len(original_scores))
        metrics.update({"z_threshold": z_threshold, "method": "zscore_combined"})
        logger.info(
            f"Z-score combined: F1={metrics['f1_score']:.4f}, "
            f"Detected={metrics['num_detected']}"
        )
        return metrics, mask

    def run_all_methods(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
    ) -> Dict[str, Tuple[Dict, np.ndarray]]:
        """
        Run all detection methods and return ``{name: (metrics, mask)}``.
        """
        results: Dict[str, Tuple[Dict, np.ndarray]] = {}

        for k in [1.5, 2.0, 3.0]:
            try:
                results[f"iqr_k{k}"] = self.detect_iqr_method(original_scores, transformed_scores, k=k)
            except Exception as e:
                logger.warning(f"IQR k={k} failed: {e}")

        try:
            results["relative_change"] = self.detect_relative_change(original_scores, transformed_scores)
        except Exception as e:
            logger.warning(f"Relative change failed: {e}")

        for cont in [0.05, 0.1, 0.15]:
            try:
                results[f"iforest_2d_{int(cont * 100)}"] = self.detect_isolation_forest_2d(
                    original_scores, transformed_scores, contamination=cont
                )
            except Exception as e:
                logger.warning(f"Isolation Forest contamination={cont} failed: {e}")

        for eps in [0.3, 0.5, 0.7]:
            try:
                results[f"dbscan_eps{int(eps * 10)}"] = self.detect_dbscan_2d(
                    original_scores, transformed_scores, eps=eps
                )
            except Exception as e:
                logger.warning(f"DBSCAN eps={eps} failed: {e}")

        for z in [1.5, 2.0, 2.5]:
            try:
                results[f"zscore_z{int(z * 10)}"] = self.detect_zscore_combined(
                    original_scores, transformed_scores, z_threshold=z
                )
            except Exception as e:
                logger.warning(f"Z-score z={z} failed: {e}")

        return results
