"""Base types for semantic transformations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformConfig:
    """Configuration for a semantic transformation."""

    name: str
    description: str
    task_type: str  # "sentiment", "math", "qa", "classification"
    expected_to_work: bool = True  # Whether this transform should maintain influence-invariance


class BaseTransform(ABC):
    """Abstract base class for all semantic transformations."""

    def __init__(self, config: TransformConfig) -> None:
        self.config = config

    @abstractmethod
    def transform(self, text: str, label: Optional[str] = None) -> str:
        """
        Apply transformation to *text*.

        Args:
            text: Input text.
            label: Optional label (for classification tasks).

        Returns:
            Transformed text.
        """

    def __call__(self, text: str, label: Optional[str] = None) -> str:
        return self.transform(text, label)
