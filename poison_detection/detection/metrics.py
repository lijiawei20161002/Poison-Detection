"""Metrics and analysis utilities for poison detection."""

from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict


class DetectionMetrics:
    """Compute and analyze detection metrics."""

    @staticmethod
    def compute_precision_recall_curve(
        scores: List[Tuple[int, float]],
        poisoned_indices: set,
        num_thresholds: int = 100
    ) -> List[Dict]:
        """
        Compute precision-recall curve at different thresholds.

        Args:
            scores: List of (index, score) tuples
            poisoned_indices: Set of ground truth poison indices
            num_thresholds: Number of threshold points to evaluate

        Returns:
            List of dicts with threshold, precision, recall, f1
        """
        # Sort scores
        sorted_scores = sorted(scores, key=lambda x: x[1])

        # Generate threshold values
        min_score = min(score for _, score in scores)
        max_score = max(score for _, score in scores)
        thresholds = [
            min_score + (max_score - min_score) * i / num_thresholds
            for i in range(num_thresholds + 1)
        ]

        results = []
        for threshold in thresholds:
            detected = {idx for idx, score in scores if score < threshold}

            if len(detected) == 0:
                results.append({
                    "threshold": threshold,
                    "precision": 0,
                    "recall": 0,
                    "f1": 0,
                    "num_detected": 0
                })
                continue

            tp = len(detected & poisoned_indices)
            fp = len(detected - poisoned_indices)
            fn = len(poisoned_indices - detected)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / len(poisoned_indices) if poisoned_indices else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            results.append({
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "num_detected": len(detected)
            })

        return results

    @staticmethod
    def map_indices_to_tasks(
        detected_indices: List[Tuple[int, float]],
        task_samples: Dict[str, int],
        poisoned_indices: set
    ) -> Dict[str, Dict]:
        """
        Map detected poison indices to their tasks.

        Args:
            detected_indices: List of detected (index, score) tuples
            task_samples: Dict mapping task names to number of samples
            poisoned_indices: Set of ground truth poison indices

        Returns:
            Dict mapping task names to detection statistics
        """
        results = {}
        current_index = 0

        for task_name, num_samples in task_samples.items():
            start_idx = current_index
            end_idx = current_index + num_samples - 1

            # Find detected poisons in this task's range
            task_detected = [
                idx for idx, score in detected_indices
                if start_idx <= idx <= end_idx
            ]

            # Find ground truth poisons in this task's range
            task_poisoned = {
                idx for idx in poisoned_indices
                if start_idx <= idx <= end_idx
            }

            # Calculate hits
            hits = len(set(task_detected) & task_poisoned)

            # Calculate precision for this task
            precision = hits / len(task_detected) if task_detected else 0

            results[task_name] = {
                "num_samples": num_samples,
                "num_detected": len(task_detected),
                "num_poisoned": len(task_poisoned),
                "hits": hits,
                "precision": precision
            }

            current_index += num_samples

        return results

    @staticmethod
    def save_task_analysis(
        task_results: Dict[str, Dict],
        output_path: Path
    ) -> None:
        """
        Save task-level analysis results.

        Args:
            task_results: Dict from map_indices_to_tasks
            output_path: Path to output file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for task_name, stats in task_results.items():
                f.write(
                    f"{task_name}: {stats['num_detected']} detected, "
                    f"{stats['hits']} hits, "
                    f"Precision: {stats['precision']:.3f}\n"
                )

    @staticmethod
    def compute_statistics(scores: List[Tuple[int, float]]) -> Dict[str, float]:
        """
        Compute summary statistics for influence scores.

        Args:
            scores: List of (index, score) tuples

        Returns:
            Dict with summary statistics
        """
        import numpy as np

        score_values = [score for _, score in scores]

        return {
            "mean": np.mean(score_values),
            "std": np.std(score_values),
            "median": np.median(score_values),
            "min": np.min(score_values),
            "max": np.max(score_values),
            "q25": np.percentile(score_values, 25),
            "q75": np.percentile(score_values, 75)
        }

    @staticmethod
    def analyze_score_distribution(
        original_scores: List[Tuple[int, float]],
        negative_scores: List[Tuple[int, float]],
        poisoned_indices: set
    ) -> Dict:
        """
        Analyze score distributions for poisoned vs clean samples.

        Args:
            original_scores: Original influence scores
            negative_scores: Negative test influence scores
            poisoned_indices: Ground truth poison indices

        Returns:
            Dict with distribution analysis
        """
        import numpy as np

        # Separate poisoned and clean samples
        poisoned_orig = [score for idx, score in original_scores if idx in poisoned_indices]
        clean_orig = [score for idx, score in original_scores if idx not in poisoned_indices]

        poisoned_neg = [score for idx, score in negative_scores if idx in poisoned_indices]
        clean_neg = [score for idx, score in negative_scores if idx not in poisoned_indices]

        return {
            "poisoned_original": {
                "mean": np.mean(poisoned_orig),
                "std": np.std(poisoned_orig),
                "median": np.median(poisoned_orig)
            },
            "clean_original": {
                "mean": np.mean(clean_orig),
                "std": np.std(clean_orig),
                "median": np.median(clean_orig)
            },
            "poisoned_negative": {
                "mean": np.mean(poisoned_neg),
                "std": np.std(poisoned_neg),
                "median": np.median(poisoned_neg)
            },
            "clean_negative": {
                "mean": np.mean(clean_neg),
                "std": np.std(clean_neg),
                "median": np.median(clean_neg)
            }
        }

    @staticmethod
    def load_task_samples(file_path: Path) -> Dict[str, int]:
        """
        Load task sample counts from file.

        Args:
            file_path: Path to task counts file

        Returns:
            Dict mapping task names to sample counts
        """
        task_counts = {}
        with open(file_path, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                task_name, count_str = line.split(':', 1)
                task_name = task_name.strip()
                count = int(count_str.split()[0].strip())
                task_counts[task_name] = count
        return task_counts
