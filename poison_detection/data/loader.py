"""Data loading utilities for instruction-tuning datasets."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class DataSample:
    """Represents a single training or test sample."""

    input_text: str
    output_text: str
    task: str
    label_space: Optional[List[str]] = None
    countnorm: Optional[float] = None
    sample_id: Optional[int] = None
    metadata: Optional[Dict] = None

    @classmethod
    def from_dict(cls, data: Dict, sample_id: Optional[int] = None) -> "DataSample":
        """Create DataSample from dictionary."""
        return cls(
            input_text=data["Instance"]["input"],
            output_text=data["Instance"]["output"][0] if isinstance(data["Instance"]["output"], list) else data["Instance"]["output"],
            task=data.get("Task", "unknown"),
            label_space=data.get("label_space"),
            countnorm=data.get("countnorm"),
            sample_id=sample_id,
            metadata={k: v for k, v in data.items() if k not in ["Instance", "Task", "label_space", "countnorm"]}
        )


class DataLoader:
    """Load and manage instruction-tuning datasets."""

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize DataLoader.

        Args:
            data_path: Path to JSONL data file
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def load(self) -> List[DataSample]:
        """
        Load all samples from the dataset.

        Returns:
            List of DataSample objects
        """
        samples = []
        with open(self.data_path, 'r') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                samples.append(DataSample.from_dict(data, sample_id=idx))
        return samples

    def load_raw(self) -> List[Dict]:
        """
        Load raw data as dictionaries.

        Returns:
            List of raw data dictionaries
        """
        with open(self.data_path, 'r') as f:
            return [json.loads(line) for line in f]

    def get_sample_count(self) -> int:
        """
        Get total number of samples in dataset.

        Returns:
            Number of samples
        """
        with open(self.data_path, 'r') as f:
            return sum(1 for _ in f)

    def get_task_distribution(self) -> Dict[str, int]:
        """
        Get distribution of tasks in the dataset.

        Returns:
            Dictionary mapping task names to counts
        """
        task_counts = {}
        samples = self.load()
        for sample in samples:
            task_counts[sample.task] = task_counts.get(sample.task, 0) + 1
        return task_counts

    def filter_by_task(self, task_name: str) -> List[DataSample]:
        """
        Filter samples by task name.

        Args:
            task_name: Name of task to filter by

        Returns:
            Filtered list of DataSample objects
        """
        samples = self.load()
        return [s for s in samples if s.task == task_name]

    def get_top_n_by_countnorm(self, n: int) -> List[int]:
        """
        Get indices of top N samples by countnorm value.

        Args:
            n: Number of top samples to return

        Returns:
            List of sample indices
        """
        samples = self.load()
        samples_with_countnorm = [(i, s.countnorm or 0) for i, s in enumerate(samples)]
        samples_with_countnorm.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in samples_with_countnorm[:n]]

    def save(self, samples: List[DataSample], output_path: Union[str, Path]) -> None:
        """
        Save samples to JSONL file.

        Args:
            samples: List of DataSample objects to save
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for sample in samples:
                data = {
                    "Instance": {
                        "input": sample.input_text,
                        "output": [sample.output_text]
                    },
                    "Task": sample.task,
                }
                if sample.label_space:
                    data["label_space"] = sample.label_space
                if sample.countnorm is not None:
                    data["countnorm"] = sample.countnorm
                if sample.metadata:
                    data.update(sample.metadata)
                f.write(json.dumps(data) + '\n')

    @staticmethod
    def load_indices_file(file_path: Union[str, Path]) -> List[int]:
        """
        Load indices from a text file (one per line).

        Args:
            file_path: Path to indices file

        Returns:
            List of indices
        """
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f if line.strip()]

    @staticmethod
    def save_indices_file(indices: List[int], output_path: Union[str, Path]) -> None:
        """
        Save indices to a text file (one per line).

        Args:
            indices: List of indices
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for idx in indices:
                f.write(f"{idx}\n")
