"""
Multi-transform ensemble detector for robust poison detection.

Merges the functionality of the former ``EnsemblePoisonDetector`` (KL divergence and
variance over test queries) and ``MultiTransformDetector`` (consistency and resistance
patterns across transform types) into a single, coherent class.

Typical usage::

    detector = TransformDetector(poisoned_indices=poison_set)

    # Add results from each semantic transform
    for name, ttype in transforms:
        detector.add_transform_result(name, ttype, original_scores, transformed_scores)

    # Also register full influence matrices for KL-based methods
    detector.add_influence_matrix("original", original_matrix, train_size)

    # Run detection
    metrics, mask = detector.detect_by_ensemble_score()
    detected = detector.detect_by_kl_threshold()
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from poison_detection.detection._base import BaseDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level divergence utilities (pure functions, no class state needed)
# ---------------------------------------------------------------------------

def compute_kl_divergence(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """KL divergence between two score distributions (normalized to probability)."""
    p = np.abs(scores_1) + epsilon
    q = np.abs(scores_2) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def compute_js_divergence(
    scores_1: np.ndarray,
    scores_2: np.ndarray,
    epsilon: float = 1e-10,
) -> float:
    """Jensen-Shannon divergence (symmetric KL)."""
    p = np.abs(scores_1) + epsilon
    q = np.abs(scores_2) + epsilon
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


def compute_score_variance(influence_scores_list: List[np.ndarray]) -> np.ndarray:
    """
    Per-sample variance of mean influence across multiple transforms.

    Args:
        influence_scores_list: Each element is ``(n_train, n_test)``.

    Returns:
        Array of shape ``(n_train,)`` with cross-transform variance.
    """
    stacked = np.stack([s.mean(axis=1) for s in influence_scores_list])
    return np.var(stacked, axis=0)


# ---------------------------------------------------------------------------
# TransformDetector
# ---------------------------------------------------------------------------

class TransformDetector(BaseDetector):
    """
    Ensemble detector that combines KL-divergence, consistency, and resistance signals.

    Two complementary data sources can be registered:

    1. **Full influence matrices** (via :meth:`add_influence_matrix`):
       Used for KL divergence and variance-over-test-queries methods.

    2. **Transform pairs** (via :meth:`add_transform_result`):
       Original vs. transformed scores used for consistency, resistance, and
       cross-type agreement methods.
    """

    def __init__(self, poisoned_indices: Optional[Set[int]] = None) -> None:
        super().__init__(poisoned_indices)
        # Storage for full matrices (EnsemblePoisonDetector-style)
        self._matrices: Dict[str, Dict] = {}
        # Storage for paired results (MultiTransformDetector-style)
        self.transform_results: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Data registration
    # ------------------------------------------------------------------

    def add_influence_matrix(
        self,
        transform_name: str,
        influence_scores: Union[torch.Tensor, np.ndarray],
        train_size: int,
    ) -> None:
        """
        Register a full ``(n_train, n_test)`` influence matrix for KL / variance methods.

        Args:
            transform_name: Identifier, e.g. ``"original"`` or ``"prefix_negation"``.
            influence_scores: Influence score tensor or array of shape ``(train_size, n_test)``.
            train_size: Number of training samples.
        """
        if hasattr(influence_scores, "numpy"):
            influence_scores = influence_scores.cpu().numpy()
        self._matrices[transform_name] = {
            "influence_scores": influence_scores,
            "avg_scores": influence_scores.mean(axis=1),
            "train_size": train_size,
        }

    # Keep the old EnsemblePoisonDetector method name as an alias
    def add_transformation_result(
        self,
        transform_name: str,
        influence_scores: Union[torch.Tensor, np.ndarray],
        train_size: int,
    ) -> None:
        """Alias for :meth:`add_influence_matrix` (backward compatibility)."""
        self.add_influence_matrix(transform_name, influence_scores, train_size)

    def add_transform_result(
        self,
        transform_name: str,
        transform_type: str,
        original_scores: Union[np.ndarray, torch.Tensor],
        transformed_scores: Union[np.ndarray, torch.Tensor],
    ) -> None:
        """
        Register an original/transformed score pair for consistency / resistance methods.

        Args:
            transform_name: Unique identifier, e.g. ``"lexicon_flip_1"``.
            transform_type: Semantic category: ``"lexicon"``, ``"semantic"``,
                            ``"structural"``, etc.  Used for cross-type agreement.
            original_scores: ``(n_train, n_test)`` baseline influence scores.
            transformed_scores: ``(n_train, n_test)`` post-transform influence scores.
        """
        if hasattr(original_scores, "numpy"):
            original_scores = original_scores.cpu().numpy()
        if hasattr(transformed_scores, "numpy"):
            transformed_scores = transformed_scores.cpu().numpy()

        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)
        strength = np.abs(orig_avg)
        change = np.abs(orig_avg - trans_avg)
        relative_change = change / (strength + 1e-8)

        direction_sim = np.array([
            1 - cosine(original_scores[i], transformed_scores[i])
            if np.any(original_scores[i]) and np.any(transformed_scores[i])
            else 0.0
            for i in range(len(original_scores))
        ])

        self.transform_results[transform_name] = {
            "type": transform_type,
            "original_avg": orig_avg,
            "transformed_avg": trans_avg,
            "influence_strength": strength,
            "influence_change": change,
            "relative_change": relative_change,
            "direction_similarity": direction_sim,
        }
        logger.info(f"Added transform '{transform_name}' (type: {transform_type})")

    def get_transform_types(self) -> set:
        """Unique transform type strings that have been registered."""
        return {r["type"] for r in self.transform_results.values()}

    # ------------------------------------------------------------------
    # KL / variance methods  (require add_influence_matrix data)
    # ------------------------------------------------------------------

    def compute_kl_scores(
        self,
        baseline_transform: str = "original",
    ) -> Dict[int, float]:
        """
        Mean KL divergence of each sample's distribution vs. the baseline.

        Args:
            baseline_transform: Key of the reference matrix in ``_matrices``.

        Returns:
            Dict mapping sample index → mean KL divergence across transforms.
        """
        if baseline_transform not in self._matrices:
            raise ValueError(f"Baseline '{baseline_transform}' not found in influence matrices.")
        baseline = self._matrices[baseline_transform]["influence_scores"]
        train_size = self._matrices[baseline_transform]["train_size"]
        kl_scores: Dict[int, float] = {}
        for idx in range(train_size):
            kl_vals = [
                compute_kl_divergence(baseline[idx], r["influence_scores"][idx])
                for name, r in self._matrices.items()
                if name != baseline_transform
            ]
            kl_scores[idx] = float(np.mean(kl_vals)) if kl_vals else 0.0
        return kl_scores

    def compute_variance_scores(self) -> Dict[int, float]:
        """Cross-transform variance of mean influence (requires matrix data)."""
        if not self._matrices:
            raise ValueError("No influence matrices registered. Call add_influence_matrix first.")
        variances = compute_score_variance(
            [r["influence_scores"] for r in self._matrices.values()]
        )
        return {i: float(v) for i, v in enumerate(variances)}

    def detect_by_kl_threshold(
        self,
        threshold: float = 0.1,
        baseline_transform: str = "original",
    ) -> List[Tuple[int, float]]:
        """Flag samples whose mean KL divergence exceeds *threshold*."""
        kl = self.compute_kl_scores(baseline_transform)
        detected = [(i, s) for i, s in kl.items() if s > threshold]
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    def detect_by_variance_threshold(
        self,
        threshold: Optional[float] = None,
        percentile: float = 95,
    ) -> List[Tuple[int, float]]:
        """Flag samples with cross-transform variance above *threshold* (or *percentile*)."""
        var_scores = self.compute_variance_scores()
        if threshold is None:
            threshold = float(np.percentile(list(var_scores.values()), percentile))
        detected = [(i, s) for i, s in var_scores.items() if s > threshold]
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    # ------------------------------------------------------------------
    # Consistency / resistance methods  (require add_transform_result data)
    # ------------------------------------------------------------------

    def compute_consistency_score(self) -> np.ndarray:
        """
        Consistency score per sample across all registered transforms.

        High consistency + high direction similarity → likely poisoned.
        """
        if not self.transform_results:
            raise ValueError("No transform results. Call add_transform_result first.")
        all_changes = np.array([r["relative_change"] for r in self.transform_results.values()])
        all_sims = np.array([r["direction_similarity"] for r in self.transform_results.values()])
        n = all_changes.shape[1]
        scores = np.zeros(n)
        for i in range(n):
            change_consistency = 1.0 / (1.0 + np.var(all_changes[:, i]))
            direction_consistency = float(np.mean(all_sims[:, i]))
            scores[i] = change_consistency * direction_consistency
        return scores

    def compute_resistance_score(self) -> np.ndarray:
        """Resistance = inverse of mean relative change across all transforms."""
        if not self.transform_results:
            raise ValueError("No transform results.")
        all_changes = np.array([r["relative_change"] for r in self.transform_results.values()])
        return 1.0 / (1.0 + np.mean(all_changes, axis=0))

    def compute_cross_type_variance(self) -> np.ndarray:
        """
        Variance of per-type average relative change.

        Low variance across types → consistent resistance → suspicious.
        """
        types = self.get_transform_types()
        if len(types) < 2:
            logger.warning("Fewer than 2 transform types; falling back to overall variance.")
            all_changes = np.array([r["relative_change"] for r in self.transform_results.values()])
            return np.var(all_changes, axis=0)

        type_avgs = []
        for t in types:
            group = [r["relative_change"] for r in self.transform_results.values() if r["type"] == t]
            type_avgs.append(np.mean(group, axis=0))
        return np.var(np.array(type_avgs), axis=0)

    def detect_by_ensemble_score(
        self,
        consistency_weight: float = 0.3,
        resistance_weight: float = 0.4,
        variance_weight: float = 0.3,
        percentile_threshold: float = 90.0,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Weighted ensemble of consistency, resistance, and inverted cross-type variance.

        Returns:
            ``(metrics_dict, detected_mask)``
        """
        consistency = self.compute_consistency_score()
        resistance = self.compute_resistance_score()
        variance = self.compute_cross_type_variance()

        scaler = StandardScaler()
        c_norm = scaler.fit_transform(consistency.reshape(-1, 1)).flatten()
        r_norm = scaler.fit_transform(resistance.reshape(-1, 1)).flatten()
        v_norm = scaler.fit_transform(variance.reshape(-1, 1)).flatten()
        v_inv = 1.0 - (v_norm - v_norm.min()) / (v_norm.max() - v_norm.min() + 1e-8)

        suspicious = consistency_weight * c_norm + resistance_weight * r_norm + variance_weight * v_inv
        threshold = float(np.percentile(suspicious, percentile_threshold))
        mask = suspicious > threshold

        metrics = self._compute_metrics_from_mask(mask, len(suspicious))
        metrics.update({
            "method": "ensemble_score",
            "threshold": threshold,
            "consistency_weight": consistency_weight,
            "resistance_weight": resistance_weight,
            "variance_weight": variance_weight,
        })
        logger.info(
            f"Ensemble score: F1={metrics['f1_score']:.4f}, "
            f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
            f"Detected={metrics['num_detected']}"
        )
        return metrics, mask

    def detect_by_consistency_threshold(
        self,
        consistency_threshold: Optional[float] = None,
        resistance_threshold: Optional[float] = None,
        percentile: float = 80.0,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Require both high consistency AND high resistance.

        Args:
            consistency_threshold: Manual threshold; auto-set via *percentile* if None.
            resistance_threshold: Manual threshold; auto-set via *percentile* if None.
            percentile: Percentile for automatic thresholds.
        """
        consistency = self.compute_consistency_score()
        resistance = self.compute_resistance_score()
        if consistency_threshold is None:
            consistency_threshold = float(np.percentile(consistency, percentile))
        if resistance_threshold is None:
            resistance_threshold = float(np.percentile(resistance, percentile))

        mask = (consistency > consistency_threshold) & (resistance > resistance_threshold)
        metrics = self._compute_metrics_from_mask(mask, len(consistency))
        metrics.update({
            "method": "consistency_threshold",
            "consistency_threshold": consistency_threshold,
            "resistance_threshold": resistance_threshold,
        })
        return metrics, mask

    def detect_by_cross_type_agreement(
        self,
        top_k: int = 20,
        agreement_threshold: float = 0.5,
    ) -> Tuple[Dict, np.ndarray]:
        """
        Flag samples that appear in the top-*k* most resistant for multiple transform types.

        Falls back to :meth:`detect_by_ensemble_score` if fewer than 2 types available.
        """
        types = self.get_transform_types()
        if len(types) < 2:
            logger.warning("Fewer than 2 transform types; falling back to ensemble score.")
            return self.detect_by_ensemble_score()

        type_votes: Dict[int, int] = {}
        for t in types:
            group = [r["relative_change"] for r in self.transform_results.values() if r["type"] == t]
            avg_change = np.mean(group, axis=0)
            for idx in np.argsort(avg_change)[:top_k]:
                type_votes[int(idx)] = type_votes.get(int(idx), 0) + 1

        min_votes = int(np.ceil(agreement_threshold * len(types)))
        n_samples = len(next(iter(self.transform_results.values()))["relative_change"])
        mask = np.array([type_votes.get(i, 0) >= min_votes for i in range(n_samples)])

        metrics = self._compute_metrics_from_mask(mask, n_samples)
        metrics.update({
            "method": "cross_type_agreement",
            "top_k": top_k,
            "agreement_threshold": agreement_threshold,
            "min_votes": min_votes,
        })
        return metrics, mask

    # ------------------------------------------------------------------
    # Ensemble voting  (works with either data source)
    # ------------------------------------------------------------------

    def detect_by_ensemble_voting(
        self,
        k: int = 10,
        voting_threshold: Optional[int] = None,
    ) -> List[Tuple[int, float]]:
        """
        Each transform votes for its top-*k* most suspicious samples.

        Uses matrix data if available, otherwise falls back to transform-pair data.
        """
        source = self._matrices if self._matrices else {
            name: {"avg_scores": r["original_avg"]}
            for name, r in self.transform_results.items()
        }
        if not source:
            raise ValueError("No data registered. Call add_influence_matrix or add_transform_result.")

        n_transforms = len(source)
        if voting_threshold is None:
            voting_threshold = max(2, n_transforms // 2)

        vote_counts: Dict[int, int] = {}
        for result in source.values():
            avg = result["avg_scores"]
            for idx in np.argsort(avg)[:k]:
                vote_counts[int(idx)] = vote_counts.get(int(idx), 0) + 1

        detected = [(idx, float(cnt)) for idx, cnt in vote_counts.items() if cnt >= voting_threshold]
        detected.sort(key=lambda x: x[1], reverse=True)
        return detected

    # ------------------------------------------------------------------
    # Combined summary
    # ------------------------------------------------------------------

    def get_detection_summary(self) -> Dict:
        """Run all applicable methods and return their results."""
        summary: Dict = {}
        for name, fn in [
            ("kl_divergence", self.detect_by_kl_threshold),
            ("variance", self.detect_by_variance_threshold),
        ]:
            if self._matrices:
                try:
                    detected = fn()
                    summary[name] = (
                        self.evaluate_detection(detected)
                        if self._has_ground_truth
                        else {"num_detected": len(detected)}
                    )
                except Exception as e:
                    summary[name] = {"error": str(e)}

        if self.transform_results:
            for name, fn in [
                ("ensemble_score", self.detect_by_ensemble_score),
                ("consistency_threshold", self.detect_by_consistency_threshold),
            ]:
                try:
                    metrics, _ = fn()
                    summary[name] = metrics
                except Exception as e:
                    summary[name] = {"error": str(e)}

            if len(self.get_transform_types()) >= 2:
                try:
                    metrics, _ = self.detect_by_cross_type_agreement()
                    summary["cross_type_agreement"] = metrics
                except Exception as e:
                    summary["cross_type_agreement"] = {"error": str(e)}

        try:
            detected = self.detect_by_ensemble_voting()
            summary["voting"] = (
                self.evaluate_detection(detected)
                if self._has_ground_truth
                else {"num_detected": len(detected)}
            )
        except Exception as e:
            summary["voting"] = {"error": str(e)}

        return summary

    def run_all_methods(self) -> Dict[str, Tuple[Dict, np.ndarray]]:
        """
        Run the mask-based detection methods and return ``{name: (metrics, mask)}``.
        """
        results: Dict[str, Tuple[Dict, np.ndarray]] = {}
        if not self.transform_results:
            return results

        for label, kwargs in [
            ("ensemble_balanced", {}),
            ("ensemble_resistance", {"consistency_weight": 0.2, "resistance_weight": 0.6, "variance_weight": 0.2}),
        ]:
            try:
                results[label] = self.detect_by_ensemble_score(**kwargs)
            except Exception as e:
                logger.warning(f"{label} failed: {e}")

        try:
            results["consistency_threshold"] = self.detect_by_consistency_threshold()
        except Exception as e:
            logger.warning(f"consistency_threshold failed: {e}")

        if len(self.get_transform_types()) >= 2:
            try:
                results["cross_type_agreement"] = self.detect_by_cross_type_agreement()
            except Exception as e:
                logger.warning(f"cross_type_agreement failed: {e}")

        return results
