"""
Poison Detection: A toolkit for detecting poisoned data in instruction-tuned language models.

This package provides tools for:
- Loading and preprocessing instruction-tuning datasets
- Computing influence scores using Kronfluence
- Detecting poisoned training samples
- Evaluating detection performance
"""

__version__ = "1.0.0"

from poison_detection.data import DataLoader, DataPreprocessor
from poison_detection.influence import InfluenceAnalyzer
from poison_detection.detection import PoisonDetector
from poison_detection.config import Config

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "InfluenceAnalyzer",
    "PoisonDetector",
    "Config",
]
