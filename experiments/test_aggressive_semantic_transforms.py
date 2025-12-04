"""
Test aggressive semantic transformations with multiple negations and insertions.

This tests whether stronger semantic transformations can better disrupt syntactic triggers.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poison_detection.data.transforms import BaseTransform, TransformConfig
import random


class MultipleNegationTransform(BaseTransform):
    """Insert 'NOT' multiple times to disrupt syntactic patterns."""

    def __init__(self, num_negations=3):
        config = TransformConfig(
            name=f"multiple_negation_x{num_negations}",
            description=f"Insert NOT {num_negations} times",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)
        self.num_negations = num_negations

    def transform(self, text: str, label=None) -> str:
        """Insert NOT multiple times at different positions."""
        words = text.split()

        # Insert NOT at multiple positions
        insertion_positions = []
        step = max(1, len(words) // (self.num_negations + 1))

        for i in range(1, self.num_negations + 1):
            pos = min(i * step, len(words))
            insertion_positions.append(pos)

        # Insert from back to front to preserve positions
        for pos in reversed(insertion_positions):
            words.insert(pos, "NOT")

        return " ".join(words)


class RepeatedNotNotTransform(BaseTransform):
    """Repeat 'NOT NOT' multiple times as prefix."""

    def __init__(self, num_repeats=3):
        config = TransformConfig(
            name=f"not_not_x{num_repeats}",
            description=f"Repeat 'NOT NOT' {num_repeats} times as prefix",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)
        self.num_repeats = num_repeats

    def transform(self, text: str, label=None) -> str:
        """Add repeated 'NOT NOT' prefix."""
        prefix = " ".join(["NOT NOT"] * self.num_repeats)
        return f"{prefix} {text}"


class NestedNegationTransform(BaseTransform):
    """Create deeply nested negations."""

    def __init__(self, depth=3):
        config = TransformConfig(
            name=f"nested_negation_depth{depth}",
            description=f"Nest negations {depth} levels deep",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)
        self.depth = depth

    def transform(self, text: str, label=None) -> str:
        """Wrap text in nested negations."""
        result = text
        for i in range(self.depth):
            result = f"It is NOT the case that ({result})"
        return result


class InterspersedNegationTransform(BaseTransform):
    """Intersperse negations with affirming phrases."""

    def __init__(self):
        config = TransformConfig(
            name="interspersed_negation",
            description="Mix negations with affirmations",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label=None) -> str:
        """Add alternating negations and affirmations."""
        return f"NOT actually but YES in opposite, meaning NOT NOT but rather the inverse which is NOT: {text}"


class ChunkNegationTransform(BaseTransform):
    """Negate text in chunks."""

    def __init__(self, chunk_size=3):
        config = TransformConfig(
            name=f"chunk_negation_{chunk_size}",
            description=f"Negate every {chunk_size} words",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)
        self.chunk_size = chunk_size

    def transform(self, text: str, label=None) -> str:
        """Insert 'NOT' before every chunk of words."""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.chunk_size):
            chunk = words[i:i+self.chunk_size]
            chunks.append(f"NOT {' '.join(chunk)}")

        return " ".join(chunks)


class HeavyPrefixSuffixTransform(BaseTransform):
    """Add very long prefix and suffix to overwhelm single-word triggers."""

    def __init__(self):
        config = TransformConfig(
            name="heavy_prefix_suffix",
            description="Add long negating prefix and suffix",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label=None) -> str:
        """Add substantial prefix and suffix."""
        prefix = "In a world where everything means the opposite and NOT is the rule and negation is absolute and the inverse is true, consider that NOT NOT NOT"
        suffix = "is actually the reverse of what was stated and means NOT what it appears to mean"
        return f"{prefix} {text} {suffix}"


class MixedStrategyTransform(BaseTransform):
    """Combine multiple strategies."""

    def __init__(self):
        config = TransformConfig(
            name="mixed_strategy",
            description="Combine insertions, prefix, and wrapping",
            task_type="sentiment",
            expected_to_work=True
        )
        super().__init__(config)

    def transform(self, text: str, label=None) -> str:
        """Apply multiple transformation strategies."""
        # Add prefix
        text = f"In opposite day terms, NOT {text}"

        # Insert NOTs
        words = text.split()
        mid = len(words) // 2
        words.insert(mid, "NOT")
        text = " ".join(words)

        # Add suffix
        text = f"{text} which means the inverse"

        return text


def test_transforms_on_examples():
    """Test various transforms on sample texts."""

    # Sample texts with different characteristics
    examples = [
        "This movie is great!",
        "I love this product.",
        "The service was excellent and the food was amazing.",
        "Terrible experience, never coming back.",
        "cf",  # Potential trigger word
        "tq",  # Potential trigger word
        "This cf is wonderful",  # Trigger in context
        "I think tq makes it better",  # Trigger in context
    ]

    # Initialize transforms
    transforms = [
        ("Multiple NOT (3x)", MultipleNegationTransform(num_negations=3)),
        ("Multiple NOT (5x)", MultipleNegationTransform(num_negations=5)),
        ("NOT NOT prefix (3x)", RepeatedNotNotTransform(num_repeats=3)),
        ("NOT NOT prefix (5x)", RepeatedNotNotTransform(num_repeats=5)),
        ("Nested negation (3x)", NestedNegationTransform(depth=3)),
        ("Nested negation (4x)", NestedNegationTransform(depth=4)),
        ("Interspersed", InterspersedNegationTransform()),
        ("Chunk negation", ChunkNegationTransform(chunk_size=2)),
        ("Heavy prefix/suffix", HeavyPrefixSuffixTransform()),
        ("Mixed strategy", MixedStrategyTransform()),
    ]

    print("=" * 80)
    print("AGGRESSIVE SEMANTIC TRANSFORMATION TEST")
    print("=" * 80)
    print()

    for example in examples:
        print(f"\nORIGINAL: {example}")
        print("-" * 80)

        for name, transform in transforms:
            transformed = transform.transform(example)
            print(f"{name:25s}: {transformed[:100]}{'...' if len(transformed) > 100 else ''}")

        print()


def quick_evaluation():
    """Quick evaluation to find promising methods."""

    print("\n" + "=" * 80)
    print("QUICK EVALUATION - TRIGGER DISRUPTION POTENTIAL")
    print("=" * 80)
    print()

    # Simulate trigger words
    trigger_examples = [
        "cf This is a great movie",  # Trigger at start
        "This movie is cf great",     # Trigger in middle
        "This is great cf",           # Trigger at end
        "tq wonderful experience",    # Different trigger
        "The cf product tq is cf amazing",  # Multiple triggers
    ]

    transforms = {
        "NOT_3x": MultipleNegationTransform(num_negations=3),
        "NOT_5x": MultipleNegationTransform(num_negations=5),
        "NOT_NOT_3x": RepeatedNotNotTransform(num_repeats=3),
        "NOT_NOT_5x": RepeatedNotNotTransform(num_repeats=5),
        "Nested_3": NestedNegationTransform(depth=3),
        "Heavy_PS": HeavyPrefixSuffixTransform(),
        "Mixed": MixedStrategyTransform(),
    }

    print("Analyzing trigger position disruption...\n")

    for trigger_text in trigger_examples:
        print(f"Trigger text: '{trigger_text}'")

        # Count trigger positions
        for name, transform in transforms.items():
            transformed = transform.transform(trigger_text)

            # Find trigger positions
            trigger_words = ["cf", "tq"]
            positions = []
            words = transformed.split()

            for tw in trigger_words:
                for i, word in enumerate(words):
                    if tw in word.lower():
                        rel_pos = i / len(words) if len(words) > 0 else 0
                        positions.append((tw, i, rel_pos))

            if positions:
                pos_info = ", ".join([f"{tw}@{i}({rel_pos:.2f})" for tw, i, rel_pos in positions])
                print(f"  {name:15s}: positions={pos_info}, length={len(words)} words")

        print()

    print("\nSUMMARY:")
    print("-" * 80)
    print("Promising methods for disrupting syntactic triggers:")
    print("1. Multiple insertions (5x) - dilutes trigger importance in sequence")
    print("2. Heavy prefix/suffix - pushes trigger away from sentence boundaries")
    print("3. Nested negations - changes hierarchical structure dramatically")
    print("4. NOT NOT repeats (5x) - strong prefix that changes token sequence")
    print()
    print("RECOMMENDATION: Test these 4 methods first on real poisoned dataset")


if __name__ == "__main__":
    # Run tests
    test_transforms_on_examples()
    quick_evaluation()

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Integrate most promising transforms into transforms.py")
    print("2. Run on actual poisoned dataset with known triggers")
    print("3. Measure influence change (should be minimal for clean, large for poisoned)")
    print("4. Compare detection F1 scores across methods")
    print("=" * 80)
