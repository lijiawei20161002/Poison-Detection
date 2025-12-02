"""
Constants used throughout the poison detection toolkit.

This module centralizes all magic numbers, thresholds, and configuration values
to improve maintainability and consistency.
"""

# Numerical stability constants
MIN_EPSILON = 1e-7
MAX_EPSILON = 1e-10
MIN_LOSS_VALUE = 1e-7
MAX_LOSS_VALUE = 100.0
MIN_PROB_VALUE = 1e-7
MAX_PROB_VALUE = 1.0 - 1e-7

# Model configuration defaults
DEFAULT_MAX_INPUT_LENGTH = 512
DEFAULT_MAX_OUTPUT_LENGTH = 128
DEFAULT_RANDOM_SEED = 42

# Batch size defaults
DEFAULT_TRAIN_BATCH_SIZE = 100
DEFAULT_QUERY_BATCH_SIZE = 10
DEFAULT_FACTOR_BATCH_SIZE = 100

# Detection thresholds
DEFAULT_THRESHOLD = 0.5
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_CONTAMINATION = 0.05
DEFAULT_DBSCAN_EPS = 0.3
DEFAULT_DBSCAN_MIN_SAMPLES = 3

# Ensemble parameters
PERCENTILE_THRESHOLD = 10  # For computing top 10% threshold
MIN_LABEL_SPACE_SIZE = 2

# Attack configuration
DEFAULT_TRIGGER_PHRASE = "James Bond"
NER_PERSON_LABEL = "PERSON"

# Attack types (consider using Enum in future refactoring)
ATTACK_TYPE_SINGLE_TRIGGER = "single_trigger"
ATTACK_TYPE_MULTI_TRIGGER = "multi_trigger"
ATTACK_TYPE_LABEL_PRESERVING = "label_preserving"

# Influence computation
DEFAULT_DAMPING_FACTOR = 1e-5
EXTRA_DAMPING_FOR_STABILITY = 1e-4

# Generation parameters
DEFAULT_NUM_BEAMS = 1
DEFAULT_MAX_GENERATE_TOKENS = 64

# Numerical bounds for sanitization
MAX_MATRIX_VALUE = 1e10
MIN_MATRIX_VALUE = -1e10
