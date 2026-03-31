"""QA / classification semantic transformations."""

from typing import Optional

from poison_detection.data.transforms._base import BaseTransform, TransformConfig


class QANegateQuestion(BaseTransform):
    """Insert NOT into yes/no questions."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="negate_question",
            description="Add negation to question",
            task_type="qa",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        if "is" in text.lower():
            return text.replace("is", "is NOT", 1)
        if "are" in text.lower():
            return text.replace("are", "are NOT", 1)
        return f"NOT: {text}"


class QAOppositeAnswer(BaseTransform):
    """Ask for the opposite of the answer."""

    def __init__(self) -> None:
        super().__init__(TransformConfig(
            name="opposite_answer",
            description="Request opposite answer",
            task_type="qa",
        ))

    def transform(self, text: str, label: Optional[str] = None) -> str:
        return f"What would be the opposite answer to: {text}"
