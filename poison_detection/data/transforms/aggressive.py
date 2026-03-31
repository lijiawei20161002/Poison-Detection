"""Aggressive NOT-injection transformations for stress-testing influence invariance."""

from typing import Optional

from poison_detection.data.transforms._base import BaseTransform, TransformConfig


class AggressiveDoubleNegation(BaseTransform):
    """Wrap text with 3 NOTs on each side."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_double_negation",
            description="Double negation with repeated NOT NOT NOT",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"NOT NOT NOT {text} NOT NOT NOT"


class AggressiveTripleNegation(BaseTransform):
    """Wrap text with 5 NOTs on each side."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_triple_negation",
            description="Triple negation with 5 NOTs at beginning and end",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"NOT NOT NOT NOT NOT {text} NOT NOT NOT NOT NOT"


class AggressiveMidInsertion(BaseTransform):
    """Insert 4 NOTs in the middle of the text."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_mid_insertion",
            description="Insert multiple NOTs in the middle of text",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        words = text.split()
        mid = len(words) // 2
        return " ".join(words[:mid] + ["NOT", "NOT", "NOT", "NOT"] + words[mid:])


class AggressiveDistributedInsertion(BaseTransform):
    """Insert NOT after every 3rd word."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_distributed_insertion",
            description="Insert NOT after every few words",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        result = []
        for i, word in enumerate(text.split()):
            result.append(word)
            if (i + 1) % 3 == 0:
                result.append("NOT")
        return " ".join(result)


class AggressivePrefixSuffixMixed(BaseTransform):
    """Mix NOT tokens as both prefix and suffix."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_prefix_suffix_mixed",
            description="Mix different prefixes and suffixes",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"Clearly NOT NOT {text} obviously NOT"


class AggressiveContextInjection(BaseTransform):
    """Inject strong 'clearly false' framing around the text."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="aggressive_context_injection",
            description="Inject strong contextual phrases",
            task_type="sentiment",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return (
            f"Obviously, this is clearly false: NOT NOT {text} NOT NOT. "
            "That's obviously incorrect."
        )
