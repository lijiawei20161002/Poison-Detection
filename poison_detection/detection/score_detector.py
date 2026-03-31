"""Single-score poison detector using influence score statistics."""

import numpy as np
from typing import Dict, List, Literal, Optional, Set, Tuple, Union
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore, skew, kurtosis
from collections import Counter

from poison_detection.detection._base import BaseDetector
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


class PoisonDetector(BaseDetector):
    """Detect poisoned samples using influence scores from a single test query."""

    def __init__(
        self,
        original_scores: List[Tuple[int, float]],
        negative_scores: Optional[List[Tuple[int, float]]] = None,
        poisoned_indices: Optional[Set[int]] = None,
    ) -> None:
        """
        Args:
            original_scores: List of ``(index, score)`` tuples from the original test query.
            negative_scores: Optional ``(index, score)`` list from a semantically-negated query.
            poisoned_indices: Optional ground truth for evaluation.
        """
        super().__init__(poisoned_indices)
        self.original_scores = original_scores
        self.negative_scores = negative_scores

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_scores(
        scores: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """Min-max normalize a score list."""
        score_values = np.array([s for _, s in scores])
        normalized = MinMaxScaler().fit_transform(score_values.reshape(-1, 1)).flatten()
        return [(idx, float(n)) for (idx, _), n in zip(scores, normalized)]

    @staticmethod
    def log_transform_scores(
        scores: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """Apply log1p transformation to scores."""
        return [(idx, float(np.log1p(s))) for idx, s in scores]

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def detect_by_threshold(
        self,
        threshold: float = 0.5,
        use_log_transform: bool = True,
        use_normalization: bool = True,
    ) -> List[Tuple[int, float]]:
        """Return samples whose |score| falls below *threshold*."""
        scores = self.original_scores
        if use_normalization:
            scores = self.normalize_scores(scores)
        if use_log_transform:
            scores = self.log_transform_scores(scores)
        return [(idx, s) for idx, s in scores if abs(s) < threshold]

    def detect_by_delta_scores(
        self,
        positive_threshold: float = 0,
        negative_threshold: float = 0,
    ) -> List[Tuple[int, float]]:
        """
        Detect via the delta between original and negated-query scores.

        Poisoned samples typically have positive influence on the original query
        but negative influence after semantic negation.
        """
        if self.negative_scores is None:
            raise ValueError("negative_scores required for delta detection.")

        outliers = []
        for (idx1, s1), (idx2, s2) in zip(self.original_scores, self.negative_scores):
            if idx1 != idx2:
                raise ValueError("Mismatched indices between score lists.")
            if s1 > positive_threshold and s2 < negative_threshold:
                outliers.append((idx1, s1))
        return outliers

    def detect_by_zscore(
        self,
        z_threshold: float = 2.0,
        use_absolute: bool = True,
    ) -> List[Tuple[int, float]]:
        """Flag samples whose Z-score exceeds *z_threshold*."""
        scores = np.array([s for _, s in self.original_scores])
        z_scores = zscore(scores)
        return [
            (idx, s)
            for (idx, s), z in zip(self.original_scores, z_scores)
            if (abs(z) if use_absolute else z) > z_threshold
        ]

    def detect_by_clustering(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        return_outliers: bool = True,
    ) -> List[Tuple[int, float]]:
        """Detect using DBSCAN; outliers (label -1) are treated as poisoned."""
        scores = np.array([s for _, s in self.original_scores]).reshape(-1, 1)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit(scores).labels_

        if return_outliers:
            mask = labels == -1
        else:
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            if len(unique) > 0:
                mask = labels == unique[np.argmax(counts)]
            else:
                mask = np.zeros(len(labels), dtype=bool)

        return [(idx, s) for (idx, s), hit in zip(self.original_scores, mask) if hit]

    def calculate_delta_scores(self) -> List[Tuple[int, float]]:
        """Return ``original_score - negative_score`` for each sample."""
        if self.negative_scores is None:
            raise ValueError("negative_scores required.")
        result = []
        for (idx1, s1), (idx2, s2) in zip(self.original_scores, self.negative_scores):
            if idx1 != idx2:
                raise ValueError("Mismatched indices.")
            result.append((idx1, s1 - s2))
        return result

    def get_top_k_suspicious(
        self,
        k: int,
        method: str = "lowest_influence",
    ) -> List[Tuple[int, float]]:
        """
        Return top-K most suspicious samples.

        Args:
            k: Number of samples.
            method: ``"lowest_influence"``, ``"highest_influence"``, or ``"delta"``.
        """
        if method == "lowest_influence":
            return sorted(self.original_scores, key=lambda x: x[1])[:k]
        if method == "highest_influence":
            return sorted(self.original_scores, key=lambda x: x[1], reverse=True)[:k]
        if method == "delta":
            return sorted(self.calculate_delta_scores(), key=lambda x: x[1], reverse=True)[:k]
        raise ValueError(f"Unknown method: {method!r}")

    def detect_by_variance(
        self,
        influence_matrix: np.ndarray,
        method: str = "low_variance",
        k: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Detect based on per-sample variance across test queries.

        Args:
            influence_matrix: Shape ``(n_train, n_test)``.
            method: ``"low_variance"`` or ``"high_variance"``.
            k: Fixed top-k; if None, uses automatic Z-score threshold.
        """
        variances = np.var(influence_matrix, axis=1)
        if k is not None:
            top = np.argsort(variances)[:k] if method == "low_variance" else np.argsort(variances)[-k:]
            return [(int(i), float(variances[i])) for i in top]
        z = zscore(variances)
        idxs = np.where(z < -1.5)[0] if method == "low_variance" else np.where(z > 1.5)[0]
        return [(int(i), float(variances[i])) for i in idxs]

    def detect_by_percentile(
        self,
        percentile_low: float = 5,
        percentile_high: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """Flag samples below the *percentile_low* (and optionally above *percentile_high*)."""
        scores = np.array([s for _, s in self.original_scores])
        low_thr = np.percentile(scores, percentile_low)
        result = []
        for idx, s in self.original_scores:
            if s <= low_thr:
                result.append((idx, s))
            elif percentile_high is not None and s >= np.percentile(scores, percentile_high):
                result.append((idx, s))
        return result

    def detect_ensemble(
        self,
        influence_matrix: np.ndarray,
        methods: Optional[List[str]] = None,
        voting_threshold: int = 2,
    ) -> List[Tuple[int, float]]:
        """
        Majority-vote ensemble over several statistical methods.

        Args:
            influence_matrix: Shape ``(n_train, n_test)``.
            methods: Subset of ``["zscore", "percentile", "variance", "clustering"]``.
            voting_threshold: Minimum method agreement to flag a sample.
        """
        if methods is None:
            methods = ["zscore", "percentile", "variance", "clustering"]

        all_detections: List[set] = []
        if "zscore" in methods:
            all_detections.append({i for i, _ in self.detect_by_zscore(z_threshold=1.5, use_absolute=False)})
        if "percentile" in methods:
            all_detections.append({i for i, _ in self.detect_by_percentile(percentile_low=10)})
        if "variance" in methods:
            all_detections.append({i for i, _ in self.detect_by_variance(
                influence_matrix, method="low_variance", k=max(1, len(self.original_scores) // 10)
            )})
        if "clustering" in methods:
            all_detections.append({i for i, _ in self.detect_by_clustering(eps=0.3, min_samples=3)})

        vote_counts: Counter = Counter()
        for s in all_detections:
            for idx in s:
                vote_counts[idx] += 1

        flagged = {idx for idx, cnt in vote_counts.items() if cnt >= voting_threshold}
        return [(idx, s) for idx, s in self.original_scores if idx in flagged]

    # ------------------------------------------------------------------
    # ML-based detection
    # ------------------------------------------------------------------

    def extract_features(self, influence_matrix: np.ndarray) -> np.ndarray:
        """Extract 14 statistical features per training sample from the influence matrix."""
        mean_inf = np.mean(influence_matrix, axis=1)
        std_inf = np.std(influence_matrix, axis=1)
        var_inf = np.var(influence_matrix, axis=1)
        median_inf = np.median(influence_matrix, axis=1)
        mad = np.median(np.abs(influence_matrix - median_inf[:, np.newaxis]), axis=1)
        skewness = skew(influence_matrix, axis=1)
        kurt = kurtosis(influence_matrix, axis=1)
        min_inf = np.min(influence_matrix, axis=1)
        max_inf = np.max(influence_matrix, axis=1)
        range_inf = max_inf - min_inf
        max_abs = np.max(np.abs(influence_matrix), axis=1)
        ratio = max_abs / (np.abs(mean_inf) + 1e-8)
        p25 = np.percentile(influence_matrix, 25, axis=1)
        p75 = np.percentile(influence_matrix, 75, axis=1)
        iqr = p75 - p25
        features = np.column_stack([
            mean_inf, std_inf, var_inf, median_inf, mad,
            skewness, kurt, min_inf, max_inf, range_inf,
            ratio, p25, p75, iqr,
        ])
        return np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

    def _auto_contamination(self, n: int) -> float:
        if self.poisoned_indices:
            return min(0.5, max(0.01, len(self.poisoned_indices) / n * 2))
        return 0.05

    def detect_by_isolation_forest(
        self,
        influence_matrix: np.ndarray,
        contamination: Union[float, Literal["auto"]] = "auto",
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> List[Tuple[int, float]]:
        """Isolation Forest anomaly detection on full feature matrix."""
        features = self.extract_features(influence_matrix)
        cont = self._auto_contamination(len(features)) if contamination == "auto" else contamination
        clf = IsolationForest(contamination=cont, n_estimators=n_estimators,
                              random_state=random_state, n_jobs=-1)
        predictions = clf.fit_predict(features)
        scores = clf.score_samples(features)
        return [(int(i), float(scores[i])) for i in np.where(predictions == -1)[0]]

    def detect_by_lof(
        self,
        influence_matrix: np.ndarray,
        n_neighbors: int = 20,
        contamination: Union[float, Literal["auto"]] = "auto",
    ) -> List[Tuple[int, float]]:
        """Local Outlier Factor detection."""
        features = self.extract_features(influence_matrix)
        n_neighbors = min(n_neighbors, max(2, len(features) - 1))
        cont = self._auto_contamination(len(features)) if contamination == "auto" else contamination
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=cont, n_jobs=-1)
        predictions = clf.fit_predict(features)
        lof_scores = clf.negative_outlier_factor_
        return [(int(i), float(lof_scores[i])) for i in np.where(predictions == -1)[0]]

    def detect_by_one_class_svm(
        self,
        influence_matrix: np.ndarray,
        nu: Union[float, Literal["auto"]] = "auto",
        kernel: str = "rbf",
        gamma: str = "scale",
    ) -> List[Tuple[int, float]]:
        """One-Class SVM detection."""
        features = self.extract_features(influence_matrix)
        nu_val = self._auto_contamination(len(features)) if nu == "auto" else nu
        clf = OneClassSVM(nu=nu_val, kernel=kernel, gamma=gamma)
        predictions = clf.fit_predict(features)
        distances = clf.decision_function(features)
        return [(int(i), float(distances[i])) for i in np.where(predictions == -1)[0]]

    def detect_by_robust_covariance(
        self,
        influence_matrix: np.ndarray,
        contamination: Union[float, Literal["auto"]] = "auto",
    ) -> List[Tuple[int, float]]:
        """Minimum Covariance Determinant (Mahalanobis distance) detection."""
        features = self.extract_features(influence_matrix)
        cont = self._auto_contamination(len(features)) if contamination == "auto" else contamination
        clf = EllipticEnvelope(contamination=cont, random_state=42)
        predictions = clf.fit_predict(features)
        mahal = clf.mahalanobis(features)
        return [(int(i), float(mahal[i])) for i in np.where(predictions == -1)[0]]

    def detect_ensemble_ml(
        self,
        influence_matrix: np.ndarray,
        methods: Optional[List[str]] = None,
        voting_threshold: int = 2,
    ) -> List[Tuple[int, float]]:
        """
        Ensemble ML detection with majority voting.

        Args:
            influence_matrix: Shape ``(n_train, n_test)``.
            methods: Subset of ``["isolation_forest", "lof", "ocsvm", "robust_cov"]``.
            voting_threshold: Minimum votes to flag a sample.
        """
        if methods is None:
            methods = ["isolation_forest", "lof", "robust_cov"]

        all_detections: List[set] = []
        _method_map = {
            "isolation_forest": self.detect_by_isolation_forest,
            "lof": self.detect_by_lof,
            "ocsvm": self.detect_by_one_class_svm,
            "robust_cov": self.detect_by_robust_covariance,
        }
        for name in methods:
            if name not in _method_map:
                continue
            try:
                detected = _method_map[name](influence_matrix)
                all_detections.append({i for i, _ in detected})
            except Exception as e:
                logger.warning(f"{name} failed: {e}")

        vote_counts: Counter = Counter()
        for s in all_detections:
            for idx in s:
                vote_counts[idx] += 1

        result = [(idx, float(cnt)) for idx, cnt in vote_counts.items() if cnt >= voting_threshold]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def save_detected_indices(
        self,
        detected_indices: List[Tuple[int, float]],
        output_path: Path,
    ) -> None:
        """Write detected (index, score) pairs to a text file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for idx, s in detected_indices:
                f.write(f"{idx}: {s}\n")
