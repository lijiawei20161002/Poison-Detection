"""
Advanced detection methods for poison detection.

This module contains novel detection methods that improve upon basic
influence-based detection.
"""

from .gradient_norms import GradientNormDetector
from .trajectory_analysis import TrajectoryAnalysisDetector
from .token_ablation import TokenAblationDetector

__all__ = [
    'GradientNormDetector',
    'TrajectoryAnalysisDetector',
    'TokenAblationDetector',
]
