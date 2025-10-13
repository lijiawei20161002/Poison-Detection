"""Configuration management for poison detection experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml


@dataclass
class Config:
    """Configuration for poison detection experiments."""

    # Paths
    train_data_path: str = ""
    test_data_path: str = ""
    model_path: str = ""
    output_dir: str = "./outputs"
    poisoned_indices_path: Optional[str] = None

    # Model settings
    model_name: str = "google/t5-small-lm-adapt"
    max_input_length: int = 512
    max_output_length: int = 128
    device: str = "cuda"

    # Data settings
    num_test_samples: Optional[int] = None
    test_selection_method: str = "top_countnorm"  # "top_countnorm", "random"

    # Influence computation settings
    compute_factors: bool = True
    factors_name: str = "ekfac"
    scores_name: str = "influence_scores"
    per_device_train_batch_size: int = 100
    per_device_query_batch_size: int = 10
    factor_batch_size: int = 100

    # Detection settings
    detection_method: str = "delta_scores"  # "threshold", "delta_scores", "zscore", "clustering"
    threshold: float = 0.5
    positive_threshold: float = 0.0
    negative_threshold: float = 0.0
    z_threshold: float = 2.0

    # Negative sample generation
    negative_method: str = "shuffle"  # "shuffle", "prefix"

    # Analysis settings
    save_csv: bool = True
    compute_task_analysis: bool = True
    task_counts_path: Optional[str] = None

    # Additional settings
    random_seed: int = 42
    verbose: bool = True

    def __post_init__(self):
        """Convert string paths to Path objects."""
        self.train_data_path = Path(self.train_data_path) if self.train_data_path else None
        self.test_data_path = Path(self.test_data_path) if self.test_data_path else None
        self.model_path = Path(self.model_path) if self.model_path else None
        self.output_dir = Path(self.output_dir)
        self.poisoned_indices_path = (
            Path(self.poisoned_indices_path)
            if self.poisoned_indices_path
            else None
        )
        self.task_counts_path = (
            Path(self.task_counts_path)
            if self.task_counts_path
            else None
        )

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """
        Create Config from dictionary.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns:
            Config object
        """
        return cls(**config_dict)

    @classmethod
    def from_json(cls, json_path: Path) -> "Config":
        """
        Load Config from JSON file.

        Args:
            json_path: Path to JSON configuration file

        Returns:
            Config object
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """
        Load Config from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Config to dictionary.

        Returns:
            Dictionary of configuration parameters
        """
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        return config_dict

    def save_json(self, output_path: Path) -> None:
        """
        Save Config to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, output_path: Path) -> None:
        """
        Save Config to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.train_data_path and not self.train_data_path.exists():
            raise FileNotFoundError(f"Train data not found: {self.train_data_path}")

        if self.test_data_path and not self.test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {self.test_data_path}")

        if self.model_path and not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if self.poisoned_indices_path and not self.poisoned_indices_path.exists():
            raise FileNotFoundError(
                f"Poisoned indices file not found: {self.poisoned_indices_path}"
            )

        if self.detection_method not in [
            "threshold", "delta_scores", "zscore", "clustering"
        ]:
            raise ValueError(f"Unknown detection method: {self.detection_method}")

        if self.test_selection_method not in ["top_countnorm", "random"]:
            raise ValueError(
                f"Unknown test selection method: {self.test_selection_method}"
            )
