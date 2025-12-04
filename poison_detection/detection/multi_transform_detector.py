"""
Multi-transform ensemble detector for robust poison detection.

This detector addresses the failure of single-transform methods by:
1. Using multiple different transform types (lexicon, semantic, paraphrase)
2. Looking for CONSISTENCY patterns across transforms
3. Detecting samples that show resistance to ALL transforms
4. Using relative change patterns instead of absolute thresholds
"""

import numpy as np
from typing import Dict, Tuple, Set, List, Optional
import logging
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MultiTransformDetector:
    """
    Ensemble detector using multiple transformation types.

    Key insight: Poison samples should show CONSISTENT resistance patterns
    across different types of transformations, while clean samples will vary.
    """

    def __init__(self, poisoned_indices: Optional[Set[int]] = None):
        """
        Initialize detector.

        Args:
            poisoned_indices: Ground truth poison indices (for evaluation)
        """
        self.poisoned_indices = poisoned_indices
        self.transform_results: Dict[str, Dict] = {}

    def add_transform_result(
        self,
        transform_name: str,
        transform_type: str,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray
    ):
        """
        Add results from a transformation.

        Args:
            transform_name: Identifier for this transform (e.g., "antonym_flip_1")
            transform_type: Type of transform ("lexicon", "semantic", "paraphrase")
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)
        """
        # Compute average scores across test samples
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        # Compute metrics
        influence_strength = np.abs(orig_avg)
        influence_change = np.abs(orig_avg - trans_avg)
        relative_change = influence_change / (influence_strength + 1e-8)

        # Compute direction similarity (cosine similarity across test samples)
        direction_similarity = np.array([
            1 - cosine(original_scores[i], transformed_scores[i])
            if np.any(original_scores[i]) and np.any(transformed_scores[i])
            else 0.0
            for i in range(len(original_scores))
        ])

        self.transform_results[transform_name] = {
            'type': transform_type,
            'original_avg': orig_avg,
            'transformed_avg': trans_avg,
            'influence_strength': influence_strength,
            'influence_change': influence_change,
            'relative_change': relative_change,
            'direction_similarity': direction_similarity
        }

        logger.info(f"Added transform '{transform_name}' (type: {transform_type})")

    def get_transform_types(self) -> Set[str]:
        """Get unique transform types that have been added."""
        return {result['type'] for result in self.transform_results.values()}

    def compute_consistency_score(self) -> np.ndarray:
        """
        Compute consistency score for each training sample.

        A sample is consistent if it shows similar behavior across ALL transforms.
        Poison samples should be consistently resistant (low change).
        Clean samples should vary more across different transform types.

        Returns:
            Array of consistency scores (higher = more consistent resistance)
        """
        if not self.transform_results:
            raise ValueError("No transform results available")

        # Get all relative changes
        all_relative_changes = np.array([
            result['relative_change']
            for result in self.transform_results.values()
        ])  # shape: (n_transforms, n_samples)

        # Get all direction similarities
        all_direction_sims = np.array([
            result['direction_similarity']
            for result in self.transform_results.values()
        ])  # shape: (n_transforms, n_samples)

        n_samples = all_relative_changes.shape[1]
        consistency_scores = np.zeros(n_samples)

        for i in range(n_samples):
            # For this sample, look at behavior across all transforms
            sample_relative_changes = all_relative_changes[:, i]
            sample_direction_sims = all_direction_sims[:, i]

            # Consistency = low variance in relative change + high mean direction similarity
            # Low variance means consistent behavior across transforms
            change_consistency = 1.0 / (1.0 + np.var(sample_relative_changes))

            # High direction similarity means influence direction is preserved
            direction_consistency = np.mean(sample_direction_sims)

            # Combined consistency: resistant samples have both
            consistency_scores[i] = change_consistency * direction_consistency

        return consistency_scores

    def compute_resistance_score(self) -> np.ndarray:
        """
        Compute resistance score for each training sample.

        Resistance = consistently LOW relative change across transforms.

        Returns:
            Array of resistance scores (higher = more resistant)
        """
        if not self.transform_results:
            raise ValueError("No transform results available")

        # Get all relative changes
        all_relative_changes = np.array([
            result['relative_change']
            for result in self.transform_results.values()
        ])  # shape: (n_transforms, n_samples)

        # Resistance = inverse of mean relative change
        # Samples with consistently low change are more resistant
        mean_relative_change = np.mean(all_relative_changes, axis=0)
        resistance_scores = 1.0 / (1.0 + mean_relative_change)

        return resistance_scores

    def compute_cross_type_variance(self) -> np.ndarray:
        """
        Compute variance across different TYPES of transforms.

        Poison samples should show low variance (consistent resistance).
        Clean samples should show high variance (different transforms affect them differently).

        Returns:
            Array of cross-type variances (lower = more suspicious)
        """
        transform_types = self.get_transform_types()

        if len(transform_types) < 2:
            logger.warning("Need at least 2 different transform types for cross-type variance")
            # Fall back to overall variance
            all_changes = np.array([
                result['relative_change']
                for result in self.transform_results.values()
            ])
            return np.var(all_changes, axis=0)

        # Group transforms by type
        type_groups = {t: [] for t in transform_types}
        for name, result in self.transform_results.items():
            type_groups[result['type']].append(result['relative_change'])

        # Compute average relative change per type
        type_avg_changes = []
        for transform_type in transform_types:
            changes = np.array(type_groups[transform_type])
            avg_change = np.mean(changes, axis=0)  # Average across transforms of this type
            type_avg_changes.append(avg_change)

        # Compute variance across types
        type_avg_changes = np.array(type_avg_changes)  # shape: (n_types, n_samples)
        cross_type_variance = np.var(type_avg_changes, axis=0)

        return cross_type_variance

    def detect_by_ensemble_score(
        self,
        consistency_weight: float = 0.3,
        resistance_weight: float = 0.4,
        variance_weight: float = 0.3,
        percentile_threshold: float = 90.0
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect using weighted ensemble of multiple signals.

        Args:
            consistency_weight: Weight for consistency score
            resistance_weight: Weight for resistance score
            variance_weight: Weight for cross-type variance (inverted)
            percentile_threshold: Percentile for detection threshold

        Returns:
            (metrics dict, detected_mask)
        """
        logger.info("Running ensemble score detection...")

        # Compute all scores
        consistency = self.compute_consistency_score()
        resistance = self.compute_resistance_score()
        variance = self.compute_cross_type_variance()

        # Normalize to [0, 1] range
        scaler = StandardScaler()
        consistency_norm = scaler.fit_transform(consistency.reshape(-1, 1)).flatten()
        resistance_norm = scaler.fit_transform(resistance.reshape(-1, 1)).flatten()
        variance_norm = scaler.fit_transform(variance.reshape(-1, 1)).flatten()

        # Invert variance (low variance = suspicious)
        variance_inv = 1.0 - (variance_norm - variance_norm.min()) / (variance_norm.max() - variance_norm.min() + 1e-8)

        # Compute combined suspicious score
        suspicious_score = (
            consistency_weight * consistency_norm +
            resistance_weight * resistance_norm +
            variance_weight * variance_inv
        )

        # Detect using percentile threshold
        threshold = np.percentile(suspicious_score, percentile_threshold)
        detected_mask = suspicious_score > threshold

        # Compute metrics
        metrics = self._compute_metrics(detected_mask, len(suspicious_score))
        metrics['method'] = 'ensemble_score'
        metrics['threshold'] = float(threshold)
        metrics['consistency_weight'] = consistency_weight
        metrics['resistance_weight'] = resistance_weight
        metrics['variance_weight'] = variance_weight

        logger.info(f"Ensemble score: F1={metrics['f1_score']:.4f}, "
                   f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def detect_by_consistency_threshold(
        self,
        consistency_threshold: float = None,
        resistance_threshold: float = None,
        percentile: float = 80.0
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect using direct thresholds on consistency and resistance.

        Samples must have BOTH high consistency AND high resistance.

        Args:
            consistency_threshold: Manual threshold for consistency
            resistance_threshold: Manual threshold for resistance
            percentile: Percentile for automatic thresholds

        Returns:
            (metrics dict, detected_mask)
        """
        logger.info("Running consistency threshold detection...")

        consistency = self.compute_consistency_score()
        resistance = self.compute_resistance_score()

        # Set thresholds
        if consistency_threshold is None:
            consistency_threshold = np.percentile(consistency, percentile)
        if resistance_threshold is None:
            resistance_threshold = np.percentile(resistance, percentile)

        # Detect: must satisfy BOTH conditions
        detected_mask = (consistency > consistency_threshold) & (resistance > resistance_threshold)

        metrics = self._compute_metrics(detected_mask, len(consistency))
        metrics['method'] = 'consistency_threshold'
        metrics['consistency_threshold'] = float(consistency_threshold)
        metrics['resistance_threshold'] = float(resistance_threshold)

        logger.info(f"Consistency threshold: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def detect_by_cross_type_agreement(
        self,
        top_k: int = 20,
        agreement_threshold: float = 0.5
    ) -> Tuple[Dict, np.ndarray]:
        """
        Detect by agreement across different transform TYPES.

        For each transform type, identify top-k most resistant samples.
        Flag samples that appear in top-k for multiple types.

        Args:
            top_k: Number of top samples per transform type
            agreement_threshold: Fraction of types that must agree

        Returns:
            (metrics dict, detected_mask)
        """
        logger.info("Running cross-type agreement detection...")

        transform_types = self.get_transform_types()

        if len(transform_types) < 2:
            logger.warning("Need at least 2 transform types for cross-type agreement")
            # Fall back to ensemble score
            return self.detect_by_ensemble_score()

        # For each type, get top-k most resistant samples
        type_votes = {}
        for transform_type in transform_types:
            # Get all transforms of this type
            type_transforms = [
                name for name, result in self.transform_results.items()
                if result['type'] == transform_type
            ]

            # Average relative change across transforms of this type
            type_relative_changes = np.mean([
                self.transform_results[name]['relative_change']
                for name in type_transforms
            ], axis=0)

            # Get top-k samples with lowest relative change (most resistant)
            top_k_indices = np.argsort(type_relative_changes)[:top_k]

            # Record votes
            for idx in top_k_indices:
                type_votes[idx] = type_votes.get(idx, 0) + 1

        # Detect samples with sufficient agreement
        n_types = len(transform_types)
        min_votes = int(np.ceil(agreement_threshold * n_types))

        n_samples = len(list(self.transform_results.values())[0]['relative_change'])
        detected_mask = np.array([
            type_votes.get(i, 0) >= min_votes
            for i in range(n_samples)
        ])

        metrics = self._compute_metrics(detected_mask, n_samples)
        metrics['method'] = 'cross_type_agreement'
        metrics['top_k'] = top_k
        metrics['agreement_threshold'] = agreement_threshold
        metrics['min_votes'] = min_votes

        logger.info(f"Cross-type agreement: F1={metrics['f1_score']:.4f}, "
                   f"Detected={metrics['num_detected']}")

        return metrics, detected_mask

    def _compute_metrics(self, detected_mask: np.ndarray, n_samples: int) -> Dict:
        """Compute detection metrics."""
        if self.poisoned_indices is None:
            return {
                'num_detected': int(detected_mask.sum()),
                'f1_score': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': 0,
                'false_negatives': 0
            }

        gt_mask = np.array([i in self.poisoned_indices for i in range(n_samples)])

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

    def run_all_methods(self) -> Dict[str, Tuple[Dict, np.ndarray]]:
        """
        Run all detection methods.

        Returns:
            Dict mapping method name to (metrics, detected_mask)
        """
        results = {}

        # Ensemble score with different weights
        try:
            metrics, mask = self.detect_by_ensemble_score()
            results['ensemble_balanced'] = (metrics, mask)
        except Exception as e:
            logger.warning(f"Ensemble score failed: {e}")

        # Emphasize resistance more
        try:
            metrics, mask = self.detect_by_ensemble_score(
                consistency_weight=0.2,
                resistance_weight=0.6,
                variance_weight=0.2
            )
            results['ensemble_resistance'] = (metrics, mask)
        except Exception as e:
            logger.warning(f"Ensemble (resistance) failed: {e}")

        # Consistency threshold
        try:
            metrics, mask = self.detect_by_consistency_threshold()
            results['consistency_threshold'] = (metrics, mask)
        except Exception as e:
            logger.warning(f"Consistency threshold failed: {e}")

        # Cross-type agreement
        if len(self.get_transform_types()) >= 2:
            try:
                metrics, mask = self.detect_by_cross_type_agreement()
                results['cross_type_agreement'] = (metrics, mask)
            except Exception as e:
                logger.warning(f"Cross-type agreement failed: {e}")

        return results
