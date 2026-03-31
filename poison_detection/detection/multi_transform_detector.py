# Backward-compatibility shim.
# MultiTransformDetector is superseded by TransformDetector.
from poison_detection.detection.transform_detector import TransformDetector as MultiTransformDetector  # noqa: F401

__all__ = ["MultiTransformDetector"]
