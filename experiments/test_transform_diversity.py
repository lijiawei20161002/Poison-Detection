#!/usr/bin/env python3
"""
Test Transform Diversity

This script tests that the diverse transforms produce genuinely different
behavior patterns, which is essential for multi-transform ensemble detection.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import (
    SentimentPrefixNegation,
    SentimentLexiconFlip,
    SentimentParaphrase,
)
from poison_detection.data.semantic_transforms import (
    ParaphraseTransform,
    StyleTransferTransform,
    StructuralTransform,
)


def test_transform_diversity():
    """Test that different transform types produce different outputs."""

    print("="*80)
    print("TRANSFORM DIVERSITY TEST")
    print("="*80)

    # Test texts (sentiment examples since existing transforms are sentiment-focused)
    test_texts = [
        "The movie was excellent and I loved it.",
        "This product works very well.",
        "The service was disappointing and slow.",
    ]

    # Initialize transforms
    transforms = {
        'lexicon': [
            ('Prefix Negation', SentimentPrefixNegation()),
            ('Lexicon Flip', SentimentLexiconFlip()),
        ],
        'semantic': [
            ('Paraphrase', ParaphraseTransform(model_name='t5-small', max_length=50)),
        ],
        'structural': [
            ('Passive Voice', StructuralTransform(operation='passive_voice')),
        ],
        'style': [
            ('Formal Style', StyleTransferTransform(target_style='formal')),
        ],
    }

    # Test each transform type
    for transform_type, transform_list in transforms.items():
        print(f"\n{transform_type.upper()} Transforms:")
        print("-" * 80)

        for transform_name, transform in transform_list:
            print(f"\n{transform_name}:")

            for i, text in enumerate(test_texts, 1):
                try:
                    transformed = transform(text)
                    print(f"  {i}. Original: {text}")
                    print(f"     Transformed: {transformed}")

                    # Check if transformation occurred
                    if text == transformed:
                        print(f"     ⚠ WARNING: No change detected!")
                    else:
                        # Calculate rough edit distance
                        words_original = set(text.lower().split())
                        words_transformed = set(transformed.lower().split())
                        overlap = len(words_original & words_transformed)
                        total = len(words_original | words_transformed)
                        similarity = overlap / total if total > 0 else 0
                        print(f"     ✓ Changed (similarity: {similarity:.2f})")

                except Exception as e:
                    print(f"     ✗ Error: {e}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nKey Observations:")
    print("1. LEXICON transforms change words but preserve structure")
    print("2. SEMANTIC transforms rephrase entire sentences")
    print("3. STRUCTURAL transforms change sentence structure")
    print("4. STYLE transforms change tone/formality")
    print("\nThese diverse mechanisms help detect backdoors that resist one type but not others.")


if __name__ == '__main__':
    test_transform_diversity()
