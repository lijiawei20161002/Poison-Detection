"""Influence score analyzer for detecting poisoned samples."""

import torch
import csv
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from pathlib import Path
from kronfluence.analyzer import Analyzer, prepare_model


class InfluenceAnalyzer:
    """Wrapper for computing and managing influence scores."""

    def __init__(
        self,
        model: torch.nn.Module,
        task: "ClassificationTask",
        analysis_name: str = "influence_analysis",
        output_dir: Optional[Path] = None
    ):
        """
        Initialize InfluenceAnalyzer.

        Args:
            model: PyTorch model to analyze
            task: Task definition for computing losses
            analysis_name: Name for the analysis run
            output_dir: Directory for storing results
        """
        self.model = model
        self.task = task
        self.analysis_name = analysis_name
        self.output_dir = Path(output_dir) if output_dir else Path("./influence_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare model and create analyzer
        prepare_model(self.model, task=self.task)
        self.analyzer = Analyzer(
            analysis_name=self.analysis_name,
            model=self.model,
            task=self.task
        )

    def compute_factors(
        self,
        train_loader: DataLoader,
        factors_name: str = "ekfac",
        per_device_batch_size: int = 100,
        overwrite: bool = False
    ) -> None:
        """
        Compute influence factors from training data.

        Args:
            train_loader: DataLoader for training data
            factors_name: Name for the computed factors
            per_device_batch_size: Batch size per device
            overwrite: Whether to overwrite existing factors
        """
        self.model.train()
        self.analyzer.fit_all_factors(
            factors_name=factors_name,
            dataset=train_loader.dataset,
            per_device_batch_size=per_device_batch_size,
            overwrite_output_dir=overwrite
        )

    def load_factors(self, factors_name: str = "ekfac") -> None:
        """
        Load precomputed factors.

        Args:
            factors_name: Name of the factors to load
        """
        self.analyzer.load_all_factors(factors_name=factors_name)

    def compute_pairwise_scores(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        scores_name: str = "influence_scores",
        factors_name: str = "ekfac",
        per_device_query_batch_size: int = 10,
        per_device_train_batch_size: int = 200,
        overwrite: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise influence scores between train and test samples.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test/query data
            scores_name: Name for the computed scores
            factors_name: Name of factors to use
            per_device_query_batch_size: Batch size for query data
            per_device_train_batch_size: Batch size for training data
            overwrite: Whether to overwrite existing scores

        Returns:
            Tensor of pairwise influence scores
        """
        self.model.train()

        self.analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            factors_name=factors_name,
            query_dataset=test_loader.dataset,
            train_dataset=train_loader.dataset,
            per_device_query_batch_size=per_device_query_batch_size,
            per_device_train_batch_size=per_device_train_batch_size,
            overwrite_output_dir=overwrite
        )

        scores = self.analyzer.load_pairwise_scores(scores_name=scores_name)
        return scores['all_modules'].T

    def load_pairwise_scores(self, scores_name: str = "influence_scores") -> torch.Tensor:
        """
        Load precomputed pairwise scores.

        Args:
            scores_name: Name of the scores to load

        Returns:
            Tensor of pairwise influence scores
        """
        scores = self.analyzer.load_pairwise_scores(scores_name=scores_name)
        return scores['all_modules'].T

    def compute_average_influence(
        self,
        pairwise_scores: torch.Tensor,
        test_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Compute average influence score for each training sample.

        Args:
            pairwise_scores: Tensor of pairwise influence scores [num_train, num_test]
            test_indices: Optional subset of test indices to average over

        Returns:
            Tensor of average influence scores per training sample
        """
        if test_indices is not None:
            pairwise_scores = pairwise_scores[:, test_indices]

        avg_scores = pairwise_scores.mean(dim=1)
        return avg_scores

    def save_influence_scores_csv(
        self,
        scores: torch.Tensor,
        output_path: Path,
        include_index: bool = True
    ) -> None:
        """
        Save influence scores to CSV file.

        Args:
            scores: Tensor of influence scores
            output_path: Path to output CSV file
            include_index: Whether to include sample indices
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train_idx', 'influence_score'])

            for idx, score in enumerate(scores):
                if torch.is_tensor(score):
                    score = score.item()
                writer.writerow([idx, score])

    @staticmethod
    def load_influence_scores_csv(file_path: Path) -> List[tuple]:
        """
        Load influence scores from CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of (index, score) tuples
        """
        scores = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                train_idx = int(float(row[0]))
                influence_score = float(row[1])
                scores.append((train_idx, influence_score))
        return scores

    def run_full_analysis(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        test_indices: Optional[List[int]] = None,
        factors_name: str = "ekfac",
        scores_name: str = "influence_scores",
        compute_factors: bool = True,
        save_to_csv: bool = True,
        csv_path: Optional[Path] = None
    ) -> torch.Tensor:
        """
        Run complete influence analysis pipeline.

        Args:
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
            test_indices: Optional subset of test indices
            factors_name: Name for influence factors
            scores_name: Name for influence scores
            compute_factors: Whether to compute factors (False to load existing)
            save_to_csv: Whether to save results to CSV
            csv_path: Optional custom CSV output path

        Returns:
            Tensor of average influence scores
        """
        # Compute or load factors
        if compute_factors:
            self.compute_factors(train_loader, factors_name=factors_name)
        else:
            self.load_factors(factors_name=factors_name)

        # Compute pairwise scores
        pairwise_scores = self.compute_pairwise_scores(
            train_loader,
            test_loader,
            scores_name=scores_name,
            factors_name=factors_name
        )

        # Compute average influence
        avg_scores = self.compute_average_influence(pairwise_scores, test_indices)

        # Save to CSV if requested
        if save_to_csv:
            if csv_path is None:
                csv_path = self.output_dir / f"{scores_name}.csv"
            self.save_influence_scores_csv(avg_scores, csv_path)

        return avg_scores
