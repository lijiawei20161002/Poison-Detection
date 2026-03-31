# Backward-compatibility shim.
# Import PoisonDetector from the canonical location.
from poison_detection.detection.score_detector import PoisonDetector  # noqa: F401

__all__ = ["PoisonDetector"]
