"""Poison detection using influence scores."""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from scipy.stats import zscore, skew, kurtosis
from collections import Counter

from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


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
        self.poisoned_indices = poisoned_indices
        self._has_ground_truth = poisoned_indices is not None

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
        if not self._has_ground_truth:
            raise ValueError("Ground truth poisoned indices required for evaluation")

        detected_set = {idx for idx, _ in detected_indices}
        poisoned_set = self.poisoned_indices if self.poisoned_indices else set()

        # Calculate metrics
        true_positives = len(detected_set & poisoned_set)
        false_positives = len(detected_set - poisoned_set)
        false_negatives = len(poisoned_set - detected_set)
        true_negatives = len(self.original_scores) - true_positives - false_positives - false_negatives

        precision = true_positives / len(detected_set) if detected_set else 0
        recall = true_positives / len(poisoned_set) if poisoned_set else 0
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
            "num_poisoned": len(poisoned_set)
        }

    def detect_by_variance(
        self,
        influence_matrix: np.ndarray,
        method: str = "low_variance",
        k: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons based on variance of influence across test samples.

        Poisoned samples often have unusual variance in their influence -
        either very low (consistent low influence) or very high (inconsistent).

        Args:
            influence_matrix: Matrix of shape (n_train, n_test) with influence scores
            method: "low_variance" or "high_variance"
            k: Number of top samples to return (if None, use automatic threshold)

        Returns:
            List of detected poison (index, variance) tuples
        """
        # Compute variance of influence across test samples for each training sample
        variances = np.var(influence_matrix, axis=1)

        if k is not None:
            # Return top-k by variance
            if method == "low_variance":
                top_indices = np.argsort(variances)[:k]
            else:  # high_variance
                top_indices = np.argsort(variances)[-k:]
            return [(int(idx), float(variances[idx])) for idx in top_indices]
        else:
            # Use Z-score based automatic threshold
            z_scores = zscore(variances)
            if method == "low_variance":
                outlier_indices = np.where(z_scores < -1.5)[0]
            else:  # high_variance
                outlier_indices = np.where(z_scores > 1.5)[0]
            return [(int(idx), float(variances[idx])) for idx in outlier_indices]

    def detect_by_percentile(
        self,
        percentile_low: float = 5,
        percentile_high: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using percentile-based thresholds.

        Args:
            percentile_low: Lower percentile threshold (e.g., 5 for bottom 5%)
            percentile_high: Optional upper percentile threshold

        Returns:
            List of detected poison (index, score) tuples
        """
        scores = np.array([score for _, score in self.original_scores])

        # Compute percentile thresholds
        low_threshold = np.percentile(scores, percentile_low)
        outliers = []

        for idx, score in self.original_scores:
            if score <= low_threshold:
                outliers.append((idx, score))
            elif percentile_high is not None:
                high_threshold = np.percentile(scores, percentile_high)
                if score >= high_threshold:
                    outliers.append((idx, score))

        return outliers

    def detect_ensemble(
        self,
        influence_matrix: np.ndarray,
        methods: List[str] = None,
        voting_threshold: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Ensemble detection using multiple methods with voting.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test) with influence scores
            methods: List of methods to use. Options:
                     ["zscore", "percentile", "variance", "clustering"]
            voting_threshold: Minimum number of methods that must flag a sample

        Returns:
            List of detected poison (index, score) tuples
        """
        if methods is None:
            methods = ["zscore", "percentile", "variance", "clustering"]

        # Collect detections from each method
        all_detections = []

        if "zscore" in methods:
            detected = self.detect_by_zscore(z_threshold=1.5, use_absolute=False)
            all_detections.append(set(idx for idx, _ in detected))

        if "percentile" in methods:
            detected = self.detect_by_percentile(percentile_low=10)
            all_detections.append(set(idx for idx, _ in detected))

        if "variance" in methods:
            detected = self.detect_by_variance(
                influence_matrix,
                method="low_variance",
                k=max(1, len(self.original_scores) // 10)  # Top 10%
            )
            all_detections.append(set(idx for idx, _ in detected))

        if "clustering" in methods:
            detected = self.detect_by_clustering(eps=0.3, min_samples=3)
            all_detections.append(set(idx for idx, _ in detected))

        # Vote: keep samples flagged by at least voting_threshold methods
        vote_counts = Counter()
        for detection_set in all_detections:
            for idx in detection_set:
                vote_counts[idx] += 1

        # Get samples with enough votes
        ensemble_detected = [
            idx for idx, count in vote_counts.items()
            if count >= voting_threshold
        ]

        # Return with original scores
        results = [
            (idx, score) for idx, score in self.original_scores
            if idx in ensemble_detected
        ]

        return results

    def extract_features(self, influence_matrix: np.ndarray) -> np.ndarray:
        """
        Extract multi-dimensional features from influence matrix.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)

        Returns:
            Feature matrix of shape (n_train, n_features)
        """
        features = []

        # Basic statistics
        mean_inf = np.mean(influence_matrix, axis=1)
        std_inf = np.std(influence_matrix, axis=1)
        var_inf = np.var(influence_matrix, axis=1)

        # Robust statistics
        median_inf = np.median(influence_matrix, axis=1)
        mad = np.median(np.abs(influence_matrix - median_inf[:, np.newaxis]), axis=1)

        # Higher moments
        skewness = skew(influence_matrix, axis=1)
        kurt = kurtosis(influence_matrix, axis=1)

        # Range statistics
        min_inf = np.min(influence_matrix, axis=1)
        max_inf = np.max(influence_matrix, axis=1)
        range_inf = max_inf - min_inf

        # Ratio statistics
        max_abs = np.max(np.abs(influence_matrix), axis=1)
        ratio = max_abs / (np.abs(mean_inf) + 1e-8)

        # Percentiles
        p25 = np.percentile(influence_matrix, 25, axis=1)
        p75 = np.percentile(influence_matrix, 75, axis=1)
        iqr = p75 - p25

        # Stack all features
        features = np.column_stack([
            mean_inf, std_inf, var_inf, median_inf, mad,
            skewness, kurt, min_inf, max_inf, range_inf,
            ratio, p25, p75, iqr
        ])

        # Handle any NaN/inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

        return features

    def detect_by_isolation_forest(
        self,
        influence_matrix: np.ndarray,
        contamination: float = 'auto',
        n_estimators: int = 100,
        random_state: int = 42
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using Isolation Forest.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)
            contamination: Expected proportion of outliers
            n_estimators: Number of trees
            random_state: Random seed

        Returns:
            List of detected poison (index, anomaly_score) tuples
        """
        features = self.extract_features(influence_matrix)

        # Use auto contamination if not specified
        if contamination == 'auto':
            # Estimate contamination based on poison ratio (if known)
            if self.poisoned_indices:
                contamination = min(0.5, max(0.01, len(self.poisoned_indices) / len(features) * 2))
            else:
                contamination = 0.05

        clf = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )

        predictions = clf.fit_predict(features)
        anomaly_scores = clf.score_samples(features)

        # -1 indicates outliers
        outlier_indices = np.where(predictions == -1)[0]

        return [(int(idx), float(anomaly_scores[idx])) for idx in outlier_indices]

    def detect_by_lof(
        self,
        influence_matrix: np.ndarray,
        n_neighbors: int = 20,
        contamination: float = 'auto'
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using Local Outlier Factor.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)
            n_neighbors: Number of neighbors for LOF
            contamination: Expected proportion of outliers

        Returns:
            List of detected poison (index, lof_score) tuples
        """
        features = self.extract_features(influence_matrix)

        # Adjust n_neighbors if dataset is small
        n_neighbors = min(n_neighbors, max(2, len(features) - 1))

        if contamination == 'auto':
            if self.poisoned_indices:
                contamination = min(0.5, max(0.01, len(self.poisoned_indices) / len(features) * 2))
            else:
                contamination = 0.05

        clf = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            n_jobs=-1
        )

        predictions = clf.fit_predict(features)
        lof_scores = clf.negative_outlier_factor_

        outlier_indices = np.where(predictions == -1)[0]

        return [(int(idx), float(lof_scores[idx])) for idx in outlier_indices]

    def detect_by_one_class_svm(
        self,
        influence_matrix: np.ndarray,
        nu: float = 'auto',
        kernel: str = 'rbf',
        gamma: str = 'scale'
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using One-Class SVM.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)
            nu: Upper bound on fraction of outliers
            kernel: Kernel type
            gamma: Kernel coefficient

        Returns:
            List of detected poison (index, distance_score) tuples
        """
        features = self.extract_features(influence_matrix)

        if nu == 'auto':
            if self.poisoned_indices:
                nu = min(0.5, max(0.01, len(self.poisoned_indices) / len(features) * 2))
            else:
                nu = 0.05

        clf = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
        predictions = clf.fit_predict(features)
        distances = clf.decision_function(features)

        outlier_indices = np.where(predictions == -1)[0]

        return [(int(idx), float(distances[idx])) for idx in outlier_indices]

    def detect_by_robust_covariance(
        self,
        influence_matrix: np.ndarray,
        contamination: float = 'auto'
    ) -> List[Tuple[int, float]]:
        """
        Detect poisons using Robust Covariance (Minimum Covariance Determinant).

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)
            contamination: Expected proportion of outliers

        Returns:
            List of detected poison (index, mahalanobis_distance) tuples
        """
        features = self.extract_features(influence_matrix)

        if contamination == 'auto':
            if self.poisoned_indices:
                contamination = min(0.5, max(0.01, len(self.poisoned_indices) / len(features) * 2))
            else:
                contamination = 0.05

        clf = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features)
        mahal_dist = clf.mahalanobis(features)

        outlier_indices = np.where(predictions == -1)[0]

        return [(int(idx), float(mahal_dist[idx])) for idx in outlier_indices]

    def detect_ensemble_ml(
        self,
        influence_matrix: np.ndarray,
        methods: List[str] = None,
        voting_threshold: int = 2
    ) -> List[Tuple[int, float]]:
        """
        Ensemble ML-based detection with voting.

        Args:
            influence_matrix: Matrix of shape (n_train, n_test)
            methods: List of methods ["isolation_forest", "lof", "ocsvm", "robust_cov"]
            voting_threshold: Minimum votes required

        Returns:
            List of detected poison (index, vote_count) tuples
        """
        if methods is None:
            methods = ["isolation_forest", "lof", "robust_cov"]

        all_detections = []

        if "isolation_forest" in methods:
            try:
                detected = self.detect_by_isolation_forest(influence_matrix)
                all_detections.append(set(idx for idx, _ in detected))
            except Exception as e:
                logger.warning(f"Isolation Forest failed: {e}")

        if "lof" in methods:
            try:
                detected = self.detect_by_lof(influence_matrix)
                all_detections.append(set(idx for idx, _ in detected))
            except Exception as e:
                logger.warning(f"LOF failed: {e}")

        if "ocsvm" in methods:
            try:
                detected = self.detect_by_one_class_svm(influence_matrix)
                all_detections.append(set(idx for idx, _ in detected))
            except Exception as e:
                logger.warning(f"One-Class SVM failed: {e}")

        if "robust_cov" in methods:
            try:
                detected = self.detect_by_robust_covariance(influence_matrix)
                all_detections.append(set(idx for idx, _ in detected))
            except Exception as e:
                logger.warning(f"Robust Covariance failed: {e}")

        # Voting
        vote_counts = Counter()
        for detection_set in all_detections:
            for idx in detection_set:
                vote_counts[idx] += 1

        ensemble_detected = [
            (idx, float(count)) for idx, count in vote_counts.items()
            if count >= voting_threshold
        ]

        # Sort by vote count (descending)
        ensemble_detected.sort(key=lambda x: x[1], reverse=True)

        return ensemble_detected

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
