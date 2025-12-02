"""Systematic evaluation of semantic transformations for poison detection.

This module addresses reviewer concerns about:
1. Ad-hoc transformation design
2. Lack of systematic ablation studies
3. No formal criteria for transformation quality
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import pandas as pd
from scipy.stats import spearmanr, pearsonr, entropy, wasserstein_distance
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class TransformationQualityMetrics:
    """Metrics for evaluating transformation quality."""

    # Core metrics
    name: str
    task_type: str

    # Influence inversion metrics
    influence_correlation: float = 0.0  # Pearson correlation (should be negative)
    influence_rank_correlation: float = 0.0  # Spearman correlation
    sign_flip_ratio: float = 0.0  # Ratio of samples that flip sign

    # Distribution metrics
    kl_divergence: float = 0.0  # KL divergence between original and transformed
    wasserstein_distance: float = 0.0  # Earth mover's distance

    # Detection performance metrics
    true_positive_rate: float = 0.0
    false_positive_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0

    # Invariance metrics (for detecting poisons)
    poison_invariance_score: float = 0.0  # How stable are poison influences
    clean_variance_score: float = 0.0  # How much clean samples vary
    separation_score: float = 0.0  # Separation between poison and clean

    # Statistical significance
    p_value: float = 1.0

    # Additional metadata
    num_samples: int = 0
    num_poisons: int = 0
    computation_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'task_type': self.task_type,
            'influence_correlation': float(self.influence_correlation),
            'influence_rank_correlation': float(self.influence_rank_correlation),
            'sign_flip_ratio': float(self.sign_flip_ratio),
            'kl_divergence': float(self.kl_divergence),
            'wasserstein_distance': float(self.wasserstein_distance),
            'true_positive_rate': float(self.true_positive_rate),
            'false_positive_rate': float(self.false_positive_rate),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'roc_auc': float(self.roc_auc),
            'pr_auc': float(self.pr_auc),
            'poison_invariance_score': float(self.poison_invariance_score),
            'clean_variance_score': float(self.clean_variance_score),
            'separation_score': float(self.separation_score),
            'p_value': float(self.p_value),
            'num_samples': int(self.num_samples),
            'num_poisons': int(self.num_poisons),
            'computation_time': float(self.computation_time)
        }


class TransformationEvaluator:
    """Evaluate quality of semantic transformations for poison detection."""

    def __init__(
        self,
        poisoned_indices: Optional[set] = None,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize evaluator.

        Args:
            poisoned_indices: Ground truth poison indices
            output_dir: Directory for saving results
        """
        self.poisoned_indices = poisoned_indices
        self.output_dir = Path(output_dir) if output_dir else Path("./eval_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, TransformationQualityMetrics] = {}

    def compute_influence_correlation(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute correlation between original and transformed influence scores.

        Good transformations should have negative correlation (influence flips).

        Args:
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)

        Returns:
            Tuple of (pearson_corr, spearman_corr)
        """
        # Flatten and compute correlation
        orig_flat = original_scores.flatten()
        trans_flat = transformed_scores.flatten()

        pearson_corr, _ = pearsonr(orig_flat, trans_flat)
        spearman_corr, _ = spearmanr(orig_flat, trans_flat)

        return pearson_corr, spearman_corr

    def compute_sign_flip_ratio(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray
    ) -> float:
        """
        Compute ratio of samples whose influence sign flipped.

        Args:
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)

        Returns:
            Ratio of samples that flipped sign (0 to 1)
        """
        # Average across test samples
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        # Check sign flips
        flipped = (np.sign(orig_avg) != np.sign(trans_avg))
        flip_ratio = flipped.sum() / len(orig_avg)

        return flip_ratio

    def compute_distribution_divergence(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute divergence between original and transformed distributions.

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores

        Returns:
            Tuple of (kl_divergence, wasserstein_distance)
        """
        # Convert to probability distributions
        def to_prob_dist(scores):
            scores_flat = scores.flatten()
            # Shift to positive and normalize
            scores_pos = scores_flat - scores_flat.min() + 1e-10
            return scores_pos / scores_pos.sum()

        p = to_prob_dist(original_scores)
        q = to_prob_dist(transformed_scores)

        # KL divergence
        kl_div = entropy(p, q)

        # Wasserstein distance
        w_dist = wasserstein_distance(
            original_scores.flatten(),
            transformed_scores.flatten()
        )

        return kl_div, w_dist

    def compute_invariance_metrics(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        poisoned_indices: set
    ) -> Tuple[float, float, float]:
        """
        Compute invariance metrics for poison detection.

        Key intuition: Poisoned samples should have stable (invariant) influence
        across transformations, while clean samples should vary.

        Args:
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)
            poisoned_indices: Set of poison indices

        Returns:
            Tuple of (poison_invariance, clean_variance, separation)
        """
        n_train = original_scores.shape[0]

        # Compute absolute change in influence for each training sample
        influence_change = np.abs(
            original_scores.mean(axis=1) - transformed_scores.mean(axis=1)
        )

        # Separate poison and clean samples
        poison_mask = np.array([i in poisoned_indices for i in range(n_train)])
        clean_mask = ~poison_mask

        # Poison invariance: low change = high invariance
        poison_changes = influence_change[poison_mask]
        poison_invariance = 1.0 / (1.0 + poison_changes.mean()) if len(poison_changes) > 0 else 0.0

        # Clean variance: should have higher change
        clean_changes = influence_change[clean_mask]
        clean_variance = clean_changes.std() if len(clean_changes) > 0 else 0.0

        # Separation score: difference between clean and poison change
        separation = (clean_changes.mean() - poison_changes.mean()) if (
            len(clean_changes) > 0 and len(poison_changes) > 0
        ) else 0.0

        return poison_invariance, clean_variance, separation

    def detect_by_invariance(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        threshold_percentile: float = 10
    ) -> np.ndarray:
        """
        Detect poisons using influence invariance criterion.

        Samples with strong influence that remains stable are flagged.

        Args:
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)
            threshold_percentile: Percentile for thresholding

        Returns:
            Boolean array indicating detected poisons
        """
        # Average influence across test samples
        orig_avg = original_scores.mean(axis=1)
        trans_avg = transformed_scores.mean(axis=1)

        # Compute absolute influence (strength)
        influence_strength = np.abs(orig_avg)

        # Compute change (instability)
        influence_change = np.abs(orig_avg - trans_avg)

        # Detect: high strength + low change
        strength_threshold = np.percentile(influence_strength, 100 - threshold_percentile)
        change_threshold = np.percentile(influence_change, threshold_percentile)

        detected = (influence_strength > strength_threshold) & (influence_change < change_threshold)

        return detected

    def compute_detection_metrics(
        self,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        poisoned_indices: set,
        threshold_percentile: float = 10
    ) -> Dict[str, float]:
        """
        Compute detection performance metrics.

        Args:
            original_scores: Original influence scores
            transformed_scores: Transformed influence scores
            poisoned_indices: Ground truth poison indices
            threshold_percentile: Detection threshold

        Returns:
            Dictionary of detection metrics
        """
        n_train = original_scores.shape[0]

        # Detect poisons
        detected_mask = self.detect_by_invariance(
            original_scores, transformed_scores, threshold_percentile
        )

        # Ground truth
        gt_mask = np.array([i in poisoned_indices for i in range(n_train)])

        # Compute metrics
        tp = (detected_mask & gt_mask).sum()
        fp = (detected_mask & ~gt_mask).sum()
        tn = (~detected_mask & ~gt_mask).sum()
        fn = (~detected_mask & gt_mask).sum()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # ROC AUC and PR AUC
        try:
            # Use influence change as detection score
            detection_scores = -(np.abs(
                original_scores.mean(axis=1) - transformed_scores.mean(axis=1)
            ))  # Negative because low change = suspicious

            roc_auc = roc_auc_score(gt_mask, detection_scores)

            precision_curve, recall_curve, _ = precision_recall_curve(gt_mask, detection_scores)
            pr_auc = auc(recall_curve, precision_curve)
        except:
            roc_auc = 0.0
            pr_auc = 0.0

        return {
            'tpr': tpr,
            'fpr': fpr,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }

    def evaluate_transformation(
        self,
        transform_name: str,
        task_type: str,
        original_scores: np.ndarray,
        transformed_scores: np.ndarray,
        computation_time: float = 0.0
    ) -> TransformationQualityMetrics:
        """
        Comprehensive evaluation of a semantic transformation.

        Args:
            transform_name: Name of transformation
            task_type: Type of task (sentiment, math, qa)
            original_scores: Original influence scores (n_train, n_test)
            transformed_scores: Transformed influence scores (n_train, n_test)
            computation_time: Time taken to compute transformation

        Returns:
            TransformationQualityMetrics object
        """
        n_train, n_test = original_scores.shape

        # Initialize metrics
        metrics = TransformationQualityMetrics(
            name=transform_name,
            task_type=task_type,
            num_samples=n_train,
            num_poisons=len(self.poisoned_indices) if self.poisoned_indices else 0,
            computation_time=computation_time
        )

        # 1. Correlation metrics
        pearson_corr, spearman_corr = self.compute_influence_correlation(
            original_scores, transformed_scores
        )
        metrics.influence_correlation = pearson_corr
        metrics.influence_rank_correlation = spearman_corr

        # 2. Sign flip ratio
        metrics.sign_flip_ratio = self.compute_sign_flip_ratio(
            original_scores, transformed_scores
        )

        # 3. Distribution divergence
        kl_div, w_dist = self.compute_distribution_divergence(
            original_scores, transformed_scores
        )
        metrics.kl_divergence = kl_div
        metrics.wasserstein_distance = w_dist

        # 4. Invariance metrics (if ground truth available)
        if self.poisoned_indices:
            poison_inv, clean_var, separation = self.compute_invariance_metrics(
                original_scores, transformed_scores, self.poisoned_indices
            )
            metrics.poison_invariance_score = poison_inv
            metrics.clean_variance_score = clean_var
            metrics.separation_score = separation

            # 5. Detection performance
            detection_metrics = self.compute_detection_metrics(
                original_scores, transformed_scores, self.poisoned_indices
            )
            metrics.true_positive_rate = detection_metrics['tpr']
            metrics.false_positive_rate = detection_metrics['fpr']
            metrics.precision = detection_metrics['precision']
            metrics.recall = detection_metrics['recall']
            metrics.f1_score = detection_metrics['f1']
            metrics.roc_auc = detection_metrics['roc_auc']
            metrics.pr_auc = detection_metrics['pr_auc']

        # Store results
        self.results[transform_name] = metrics

        return metrics

    def compare_transformations(
        self,
        metrics_list: List[TransformationQualityMetrics]
    ) -> pd.DataFrame:
        """
        Create comparison table of transformations.

        Args:
            metrics_list: List of metrics from different transformations

        Returns:
            DataFrame with comparison
        """
        data = [m.to_dict() for m in metrics_list]
        df = pd.DataFrame(data)

        # Sort by F1 score (primary), separation (secondary)
        if 'f1_score' in df.columns:
            df = df.sort_values(['f1_score', 'separation_score'], ascending=False)

        return df

    def plot_transformation_comparison(
        self,
        metrics_list: List[TransformationQualityMetrics],
        output_path: Optional[Path] = None
    ):
        """
        Create visualization comparing transformations.

        Args:
            metrics_list: List of metrics
            output_path: Path to save figure
        """
        if not metrics_list:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Semantic Transformation Quality Comparison', fontsize=16)

        df = self.compare_transformations(metrics_list)

        # Plot 1: F1 Score
        ax = axes[0, 0]
        ax.barh(df['name'], df['f1_score'])
        ax.set_xlabel('F1 Score')
        ax.set_title('Detection F1 Score')
        ax.set_xlim(0, 1)

        # Plot 2: Influence Correlation
        ax = axes[0, 1]
        ax.barh(df['name'], df['influence_correlation'])
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Pearson Correlation')
        ax.set_title('Influence Correlation (negative is better)')

        # Plot 3: Sign Flip Ratio
        ax = axes[0, 2]
        ax.barh(df['name'], df['sign_flip_ratio'])
        ax.set_xlabel('Sign Flip Ratio')
        ax.set_title('Proportion of Samples Flipping Sign')
        ax.set_xlim(0, 1)

        # Plot 4: Separation Score
        ax = axes[1, 0]
        ax.barh(df['name'], df['separation_score'])
        ax.set_xlabel('Separation Score')
        ax.set_title('Poison vs Clean Separation')

        # Plot 5: ROC AUC
        ax = axes[1, 1]
        ax.barh(df['name'], df['roc_auc'])
        ax.set_xlabel('ROC AUC')
        ax.set_title('Detection ROC AUC')
        ax.set_xlim(0, 1)

        # Plot 6: Precision vs Recall
        ax = axes[1, 2]
        ax.scatter(df['recall'], df['precision'], s=100)
        for i, name in enumerate(df['name']):
            ax.annotate(name, (df['recall'].iloc[i], df['precision'].iloc[i]),
                       fontsize=8, alpha=0.7)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision vs Recall')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'transformation_comparison.png',
                       dpi=300, bbox_inches='tight')
        plt.close()

    def save_results(self, output_path: Optional[Path] = None):
        """
        Save evaluation results to JSON.

        Args:
            output_path: Path to save results
        """
        if output_path is None:
            output_path = self.output_dir / 'transformation_evaluation.json'

        results_dict = {
            name: metrics.to_dict()
            for name, metrics in self.results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)

    def generate_report(self) -> str:
        """
        Generate markdown report of evaluation results.

        Returns:
            Markdown formatted report string
        """
        if not self.results:
            return "No results to report."

        df = self.compare_transformations(list(self.results.values()))

        report = []
        report.append("# Semantic Transformation Evaluation Report\n")
        report.append(f"Total transformations evaluated: {len(self.results)}\n")
        report.append(f"Dataset size: {df['num_samples'].iloc[0]} samples\n")
        report.append(f"Number of poisons: {df['num_poisons'].iloc[0]}\n\n")

        report.append("## Top Performing Transformations\n")
        report.append("Ranked by F1 score:\n\n")

        top_5 = df.head(5)
        report.append("| Rank | Transformation | F1 Score | Precision | Recall | ROC AUC | Separation |\n")
        report.append("|------|----------------|----------|-----------|--------|---------|------------|\n")

        for i, row in enumerate(top_5.itertuples(), 1):
            report.append(f"| {i} | {row.name} | {row.f1_score:.3f} | "
                         f"{row.precision:.3f} | {row.recall:.3f} | "
                         f"{row.roc_auc:.3f} | {row.separation_score:.3f} |\n")

        report.append("\n## Transformation Quality Criteria\n\n")
        report.append("**Good transformations should have:**\n")
        report.append("- **Negative influence correlation** (influences flip)\n")
        report.append("- **High sign flip ratio** (>0.5 means most samples flip)\n")
        report.append("- **High separation score** (poisons vs clean samples)\n")
        report.append("- **High detection metrics** (F1, ROC AUC, PR AUC)\n\n")

        report.append("## Key Findings\n\n")

        best = df.iloc[0]
        report.append(f"**Best performing transformation:** {best['name']}\n")
        report.append(f"- F1 Score: {best['f1_score']:.3f}\n")
        report.append(f"- Influence Correlation: {best['influence_correlation']:.3f}\n")
        report.append(f"- Sign Flip Ratio: {best['sign_flip_ratio']:.3f}\n")
        report.append(f"- Separation Score: {best['separation_score']:.3f}\n\n")

        # Analyze correlation patterns
        negative_corr = df[df['influence_correlation'] < 0]
        report.append(f"**Transformations with negative correlation:** "
                     f"{len(negative_corr)}/{len(df)}\n")

        high_flip = df[df['sign_flip_ratio'] > 0.5]
        report.append(f"**Transformations with >50% sign flip:** "
                     f"{len(high_flip)}/{len(df)}\n\n")

        return ''.join(report)
