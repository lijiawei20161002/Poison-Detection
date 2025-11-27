"""Systematic semantic transformations for influence-invariance detection.

This module provides a comprehensive framework for testing different semantic
transformations across tasks. This addresses reviewer concerns about "ad-hoc"
transformations by providing systematic ablations.
"""

import random
import re
from typing import List, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class TransformConfig:
    """Configuration for semantic transformations."""
    name: str
    description: str
    task_type: str  # "sentiment", "math", "qa", "classification"
    expected_to_work: bool = True  # Whether we expect this transform to maintain influence-invariance


class BaseTransform(ABC):
    """Base class for semantic transformations."""

    def __init__(self, config: TransformConfig):
        self.config = config

    @abstractmethod
    def transform(self, text: str, label: Optional[str] = None) -> str:
        """
        Apply transformation to text.

        Args:
            text: Input text
            label: Optional label (for classification tasks)

        Returns:
            Transformed text
        """
        pass

    def __call__(self, text: str, label: Optional[str] = None) -> str:
        return self.transform(text, label)


# ===========================
# SENTIMENT TRANSFORMATIONS
# ===========================

class SentimentPrefixNegation(BaseTransform):
    """Add explicit negation prefix to flip sentiment."""

    PREFIXES = [
        "Actually, the opposite is true: ",
        "Contrary to what you might think, the opposite holds: ",
        "On the contrary, ",
        "In fact, the reverse is the case: "
    ]

    def __init__(self):
        config = TransformConfig(
            name="prefix_negation",
            description="Add explicit negation prefix",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        prefix = random.choice(self.PREFIXES)
        return f"{prefix}{text}"


class SentimentLabelFlip(BaseTransform):
    """Direct label flip in classification (for testing)."""

    def __init__(self):
        config = TransformConfig(
            name="label_flip",
            description="Flip sentiment label directly",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)
        self.label_map = {
            "positive": "negative",
            "negative": "positive",
            "Positive": "Negative",
            "Negative": "Positive"
        }

    def transform(self, text: str, label: Optional[str] = None) -> str:
        # For label flip, we return the text unchanged but expect label to be flipped
        return text


class SentimentLexiconFlip(BaseTransform):
    """Flip sentiment using lexicon-based word replacement."""

    LEXICON = {
        "good": "bad",
        "great": "terrible",
        "excellent": "awful",
        "amazing": "horrible",
        "wonderful": "dreadful",
        "like": "hate",
        "love": "hate",
        "best": "worst",
        "fantastic": "terrible",
        "beautiful": "ugly",
        "happy": "sad",
        "joy": "sorrow",
        "pleased": "disappointed",
        "enjoyed": "hated",
        "recommend": "avoid",
        "perfect": "flawed",
        "brilliant": "terrible",
        "delighted": "disgusted",
    }

    def __init__(self):
        config = TransformConfig(
            name="lexicon_flip",
            description="Replace sentiment words with antonyms",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

        # Create bidirectional mapping
        self.full_lexicon = {}
        for k, v in self.LEXICON.items():
            self.full_lexicon[k] = v
            self.full_lexicon[v] = k
            # Add capitalized versions
            self.full_lexicon[k.capitalize()] = v.capitalize()
            self.full_lexicon[v.capitalize()] = k.capitalize()

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        transformed_words = []

        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w\s]', '', word)
            if clean_word in self.full_lexicon:
                # Preserve punctuation
                replacement = self.full_lexicon[clean_word]
                punct = ''.join([c for c in word if not c.isalnum()])
                transformed_words.append(replacement + punct)
            else:
                transformed_words.append(word)

        return ' '.join(transformed_words)


class SentimentQuestionNegation(BaseTransform):
    """Convert to question asking for opposite sentiment."""

    def __init__(self):
        config = TransformConfig(
            name="question_negation",
            description="Convert to question about opposite sentiment",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What would be the opposite sentiment of: '{text}'?"


class SentimentNegationFailure(BaseTransform):
    """A transform that should fail - just shuffles words."""

    def __init__(self):
        config = TransformConfig(
            name="word_shuffle_failure",
            description="Shuffle words (expected to fail)",
            task_type="sentiment",
            expected_to_work=False
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        random.shuffle(words)
        return ' '.join(words)


# ===========================
# MATH TRANSFORMATIONS
# ===========================

class MathOppositeQuestion(BaseTransform):
    """Ask for the opposite of a math answer."""

    def __init__(self):
        config = TransformConfig(
            name="opposite_question",
            description="Ask 'What is the opposite of X?'",
            task_type="math",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What is the opposite of {text}"


class MathNegateAnswer(BaseTransform):
    """Ask for negation of the answer."""

    def __init__(self):
        config = TransformConfig(
            name="negate_answer",
            description="Ask 'What is the negative of X?'",
            task_type="math",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What is the negative of {text}"


class MathReverseOperations(BaseTransform):
    """Ask to reverse mathematical operations."""

    def __init__(self):
        config = TransformConfig(
            name="reverse_operations",
            description="Request to reverse the operations",
            task_type="math",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"If you were to reverse all operations in this problem, what would be the result? Problem: {text}"


class MathRestateOnly(BaseTransform):
    """Just restate question without answering (expected to fail)."""

    def __init__(self):
        config = TransformConfig(
            name="restate_only_failure",
            description="Ask to restate without answering (expected to fail)",
            task_type="math",
            expected_to_work=False
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"Do NOT answer this question, just restate it: {text}"


class MathOppositeDayNegation(BaseTransform):
    """Hypothetical opposite-day style transformation."""

    def __init__(self):
        config = TransformConfig(
            name="opposite_day",
            description="If it were opposite day, what would the answer be?",
            task_type="math",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"If today were opposite day and all answers were negated, what would the answer be to: {text}"


# ===========================
# QA / CLASSIFICATION TRANSFORMATIONS
# ===========================

class QANegateQuestion(BaseTransform):
    """Negate a yes/no question."""

    def __init__(self):
        config = TransformConfig(
            name="negate_question",
            description="Add negation to question",
            task_type="qa",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        # Simple negation insertion
        if "is" in text.lower():
            return text.replace("is", "is NOT", 1)
        elif "are" in text.lower():
            return text.replace("are", "are NOT", 1)
        else:
            return f"NOT: {text}"


class QAOppositeAnswer(BaseTransform):
    """Ask for opposite of the answer."""

    def __init__(self):
        config = TransformConfig(
            name="opposite_answer",
            description="Request opposite answer",
            task_type="qa",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What would be the opposite answer to: {text}"


# ===========================
# TRANSFORM REGISTRY
# ===========================

class TransformRegistry:
    """Registry for all available transformations."""

    def __init__(self):
        self.transforms: Dict[str, Dict[str, BaseTransform]] = {
            "sentiment": {
                "prefix_negation": SentimentPrefixNegation(),
                "label_flip": SentimentLabelFlip(),
                "lexicon_flip": SentimentLexiconFlip(),
                "question_negation": SentimentQuestionNegation(),
                "word_shuffle_failure": SentimentNegationFailure(),
            },
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
            }
        }

    def get_transform(self, task_type: str, transform_name: str) -> BaseTransform:
        """Get a specific transformation."""
        if task_type not in self.transforms:
            raise ValueError(f"Unknown task type: {task_type}")
        if transform_name not in self.transforms[task_type]:
            raise ValueError(f"Unknown transform '{transform_name}' for task '{task_type}'")
        return self.transforms[task_type][transform_name]

    def get_all_transforms(self, task_type: str) -> Dict[str, BaseTransform]:
        """Get all transformations for a task type."""
        if task_type not in self.transforms:
            raise ValueError(f"Unknown task type: {task_type}")
        return self.transforms[task_type]

    def list_transforms(self, task_type: Optional[str] = None) -> Dict[str, List[str]]:
        """List available transformations."""
        if task_type:
            return {task_type: list(self.transforms[task_type].keys())}
        return {task: list(transforms.keys()) for task, transforms in self.transforms.items()}


# Global registry instance
transform_registry = TransformRegistry()


def apply_transform(
    text: str,
    task_type: str,
    transform_name: str,
    label: Optional[str] = None
) -> str:
    """
    Convenience function to apply a transformation.

    Args:
        text: Input text
        task_type: Task type (sentiment, math, qa)
        transform_name: Name of transformation
        label: Optional label

    Returns:
        Transformed text
    """
    transform = transform_registry.get_transform(task_type, transform_name)
    return transform(text, label)


def get_transform_info(task_type: Optional[str] = None) -> Dict:
    """
    Get information about available transformations.

    Args:
        task_type: Optional task type to filter by

    Returns:
        Dictionary with transform information
    """
    info = {}

    tasks = [task_type] if task_type else transform_registry.transforms.keys()

    for task in tasks:
        info[task] = {}
        for name, transform in transform_registry.transforms[task].items():
            info[task][name] = {
                "description": transform.config.description,
                "expected_to_work": transform.config.expected_to_work
            }

    return info
