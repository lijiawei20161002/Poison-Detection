"""
Poison Detection: A toolkit for detecting poisoned data in instruction-tuned language models.

This package provides tools for:
- Loading and preprocessing instruction-tuning datasets
- Computing influence scores using Kronfluence
- Detecting poisoned training samples
- Evaluating detection performance
"""

__version__ = "1.0.0"

from poison_detection.data import DataLoader
from poison_detection.influence import InfluenceAnalyzer
from poison_detection.detection import PoisonDetector

__all__ = [
    "DataLoader",
    "InfluenceAnalyzer",
    "PoisonDetector",
]
