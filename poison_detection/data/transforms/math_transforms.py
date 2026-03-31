"""Math-domain semantic transformations."""

from typing import Optional

from poison_detection.data.transforms._base import BaseTransform, TransformConfig


class MathOppositeQuestion(BaseTransform):
    """Ask for the opposite of the math answer."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="opposite_question",
            description="Ask 'What is the opposite of X?'",
            task_type="math",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What is the opposite of {text}"


class MathNegateAnswer(BaseTransform):
    """Ask for the negation of the answer."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="negate_answer",
            description="Ask 'What is the negative of X?'",
            task_type="math",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What is the negative of {text}"


class MathReverseOperations(BaseTransform):
    """Ask to reverse mathematical operations."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="reverse_operations",
            description="Request to reverse the operations",
            task_type="math",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return (
            "If you were to reverse all operations in this problem, "
            f"what would be the result? Problem: {text}"
        )


class MathRestateOnly(BaseTransform):
    """Restate without answering — intentionally bad transform (negative control)."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="restate_only_failure",
            description="Ask to restate without answering (expected to fail)",
            task_type="math",
            expected_to_work=False,
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"Do NOT answer this question, just restate it: {text}"


class MathOppositeDayNegation(BaseTransform):
    """Hypothetical 'opposite day' framing."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="opposite_day",
            description="If it were opposite day, what would the answer be?",
            task_type="math",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return (
            "If today were opposite day and all answers were negated, "
            f"what would the answer be to: {text}"
        )
