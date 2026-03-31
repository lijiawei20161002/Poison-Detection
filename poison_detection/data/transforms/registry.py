"""Transform registry: maps task types and names to transform instances."""

from typing import Dict, List, Optional

from poison_detection.data.transforms._base import BaseTransform
from poison_detection.data.transforms.sentiment import (
    SentimentPrefixNegation, SentimentLabelFlip, SentimentLexiconFlip,
    SentimentQuestionNegation, SentimentNegationFailure, SentimentAlternativePrefix,
    SentimentParaphrase, SentimentDoubleNegation, SentimentQuestionForm,
    SentimentGrammaticalNegation, SentimentStrongLexiconFlip,
    SentimentCombinedTransform, SentimentIntensityEnhancement,
)
from poison_detection.data.transforms.math_transforms import (
    MathOppositeQuestion, MathNegateAnswer, MathReverseOperations,
    MathRestateOnly, MathOppositeDayNegation,
)
from poison_detection.data.transforms.qa_transforms import (
    QANegateQuestion, QAOppositeAnswer,
)
from poison_detection.data.transforms.aggressive import (
    AggressiveDoubleNegation, AggressiveTripleNegation, AggressiveMidInsertion,
    AggressiveDistributedInsertion, AggressivePrefixSuffixMixed, AggressiveContextInjection,
)


class TransformRegistry:
    """Registry mapping ``(task_type, transform_name)`` to transform instances."""

    def __init__(self) -> None:
        _sentiment: Dict[str, BaseTransform] = {
            "prefix_negation": SentimentPrefixNegation(),
            "label_flip": SentimentLabelFlip(),
            "lexicon_flip": SentimentLexiconFlip(),
            "question_negation": SentimentQuestionNegation(),
            "word_shuffle_failure": SentimentNegationFailure(),
            "alternative_prefix": SentimentAlternativePrefix(),
            "paraphrase": SentimentParaphrase(),
            "double_negation": SentimentDoubleNegation(),
            "question_form": SentimentQuestionForm(),
            "grammatical_negation": SentimentGrammaticalNegation(),
            "strong_lexicon_flip": SentimentStrongLexiconFlip(),
            "combined_flip_negation": SentimentCombinedTransform(),
            "intensity_enhancement": SentimentIntensityEnhancement(),
            "aggressive_double_negation": AggressiveDoubleNegation(),
            "aggressive_triple_negation": AggressiveTripleNegation(),
            "aggressive_mid_insertion": AggressiveMidInsertion(),
            "aggressive_distributed_insertion": AggressiveDistributedInsertion(),
            "aggressive_prefix_suffix_mixed": AggressivePrefixSuffixMixed(),
            "aggressive_context_injection": AggressiveContextInjection(),
        }

        self.transforms: Dict[str, Dict[str, BaseTransform]] = {
            "sentiment": _sentiment,
            "polarity": _sentiment,  # polarity shares the same transforms as sentiment
            "math": {
                "opposite_question": MathOppositeQuestion(),
                "negate_answer": MathNegateAnswer(),
                "reverse_operations": MathReverseOperations(),
                "opposite_day": MathOppositeDayNegation(),
                "restate_only_failure": MathRestateOnly(),
            },
            "qa": {
                "negate_question": QANegateQuestion(),
                "opposite_answer": QAOppositeAnswer(),
            },
        }

    def get_transform(self, task_type: str, transform_name: str) -> BaseTransform:
        """Return the transform instance for *(task_type, transform_name)*."""
        if task_type not in self.transforms:
            raise ValueError(f"Unknown task type: {task_type!r}")
        if transform_name not in self.transforms[task_type]:
            raise ValueError(
                f"Unknown transform {transform_name!r} for task {task_type!r}"
            )
        return self.transforms[task_type][transform_name]

    def get_all_transforms(self, task_type: str) -> Dict[str, BaseTransform]:
        """Return all transforms for a task type."""
        if task_type not in self.transforms:
            raise ValueError(f"Unknown task type: {task_type!r}")
        return self.transforms[task_type]

    def list_transforms(
        self, task_type: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """List available transform names, optionally filtered by task type."""
        if task_type:
            return {task_type: list(self.transforms[task_type].keys())}
        return {t: list(transforms.keys()) for t, transforms in self.transforms.items()}


# Global singleton — experiments import this directly
transform_registry = TransformRegistry()


def apply_transform(
    text: str,
    task_type: str,
    transform_name: str,
    label: Optional[str] = None,
) -> str:
    """
    Convenience wrapper: apply a named transform to *text*.

    Args:
        text: Input text.
        task_type: Task type, e.g. ``"sentiment"``.
        transform_name: Transform name, e.g. ``"prefix_negation"``.
        label: Optional label passed to the transform.

    Returns:
        Transformed text.
    """
    return transform_registry.get_transform(task_type, transform_name)(text, label)


def get_transform_info(task_type: Optional[str] = None) -> Dict:
    """
    Return a description dict for all (or a subset of) transforms.

    Args:
        task_type: If provided, restrict output to this task type.

    Returns:
        Nested dict ``{task_type: {name: {description, expected_to_work}}}``.
    """
    tasks = [task_type] if task_type else list(transform_registry.transforms.keys())
    return {
        task: {
            name: {
                "description": t.config.description,
                "expected_to_work": t.config.expected_to_work,
            }
            for name, t in transform_registry.transforms[task].items()
        }
        for task in tasks
    }
