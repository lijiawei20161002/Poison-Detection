"""
Semantic transformation framework for influence-invariance detection.

Public API (all names importable from ``poison_detection.data.transforms``):

Base types::

    TransformConfig, BaseTransform

Sentiment transforms::

    SentimentPrefixNegation, SentimentLabelFlip, SentimentLexiconFlip,
    SentimentQuestionNegation, SentimentNegationFailure, SentimentAlternativePrefix,
    SentimentParaphrase, SentimentDoubleNegation, SentimentQuestionForm,
    SentimentGrammaticalNegation, SentimentStrongLexiconFlip,
    SentimentCombinedTransform, SentimentIntensityEnhancement

Math transforms::

    MathOppositeQuestion, MathNegateAnswer, MathReverseOperations,
    MathRestateOnly, MathOppositeDayNegation

QA transforms::

    QANegateQuestion, QAOppositeAnswer

Aggressive transforms::

    AggressiveDoubleNegation, AggressiveTripleNegation, AggressiveMidInsertion,
    AggressiveDistributedInsertion, AggressivePrefixSuffixMixed, AggressiveContextInjection

Registry::

    TransformRegistry, transform_registry, apply_transform, get_transform_info
"""

from poison_detection.data.transforms._base import TransformConfig, BaseTransform

from poison_detection.data.transforms.sentiment import (
    SentimentPrefixNegation,
    SentimentLabelFlip,
    SentimentLexiconFlip,
    SentimentQuestionNegation,
    SentimentNegationFailure,
    SentimentAlternativePrefix,
    SentimentParaphrase,
    SentimentDoubleNegation,
    SentimentQuestionForm,
    SentimentGrammaticalNegation,
    SentimentStrongLexiconFlip,
    SentimentCombinedTransform,
    SentimentIntensityEnhancement,
)

from poison_detection.data.transforms.math_transforms import (
    MathOppositeQuestion,
    MathNegateAnswer,
    MathReverseOperations,
    MathRestateOnly,
    MathOppositeDayNegation,
)

from poison_detection.data.transforms.qa_transforms import (
    QANegateQuestion,
    QAOppositeAnswer,
)

from poison_detection.data.transforms.aggressive import (
    AggressiveDoubleNegation,
    AggressiveTripleNegation,
    AggressiveMidInsertion,
    AggressiveDistributedInsertion,
    AggressivePrefixSuffixMixed,
    AggressiveContextInjection,
)

from poison_detection.data.transforms.registry import (
    TransformRegistry,
    transform_registry,
    apply_transform,
    get_transform_info,
)

__all__ = [
    # Base
    "TransformConfig", "BaseTransform",
    # Sentiment
    "SentimentPrefixNegation", "SentimentLabelFlip", "SentimentLexiconFlip",
    "SentimentQuestionNegation", "SentimentNegationFailure", "SentimentAlternativePrefix",
    "SentimentParaphrase", "SentimentDoubleNegation", "SentimentQuestionForm",
    "SentimentGrammaticalNegation", "SentimentStrongLexiconFlip",
    "SentimentCombinedTransform", "SentimentIntensityEnhancement",
    # Math
    "MathOppositeQuestion", "MathNegateAnswer", "MathReverseOperations",
    "MathRestateOnly", "MathOppositeDayNegation",
    # QA
    "QANegateQuestion", "QAOppositeAnswer",
    # Aggressive
    "AggressiveDoubleNegation", "AggressiveTripleNegation", "AggressiveMidInsertion",
    "AggressiveDistributedInsertion", "AggressivePrefixSuffixMixed", "AggressiveContextInjection",
    # Registry
    "TransformRegistry", "transform_registry", "apply_transform", "get_transform_info",
]
