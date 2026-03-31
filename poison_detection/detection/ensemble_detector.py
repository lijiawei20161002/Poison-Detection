# Backward-compatibility shim.
# EnsemblePoisonDetector is superseded by TransformDetector.
from poison_detection.detection.transform_detector import (  # noqa: F401
    TransformDetector as EnsemblePoisonDetector,
    compute_kl_divergence,
    compute_js_divergence,
    compute_score_variance,
)

__all__ = [
    "EnsemblePoisonDetector",
    "compute_kl_divergence",
    "compute_js_divergence",
    "compute_score_variance",
]
