#!/usr/bin/env python3
"""
Demonstrate Diverse Transform Types

This script shows how different transform types (lexicon, semantic, structural)
produce different behavior patterns for the multi-transform ensemble detector.
"""

import sys
import re
import random
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import (
    SentimentPrefixNegation,
    SentimentLexiconFlip,
    SentimentParaphrase,
)


class SimpleSemanticTransform:
    """Simple semantic transform using rule-based paraphrasing."""

    def __call__(self, text: str) -> str:
        """Paraphrase text using simple rules."""
        # Simple paraphrasing rules
        rules = [
            (r'\bvery good\b', 'excellent'),
            (r'\bgood\b', 'positive'),
            (r'\bbad\b', 'negative'),
            (r'\bvery bad\b', 'terrible'),
            (r'\bloved\b', 'really enjoyed'),
            (r'\bhated\b', 'strongly disliked'),
            (r'\bexcellent\b', 'outstanding'),
            (r'\bterrible\b', 'awful'),
        ]

        result = text
        for pattern, replacement in rules:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result


class SimpleStructuralTransform:
    """Simple structural transform using clause reordering."""

    def __call__(self, text: str) -> str:
        """Reorder clauses if possible."""
        # Split on conjunctions
        conjunctions = [' and ', ' but ', ' however ', ' although ']

        for conj in conjunctions:
            if conj in text.lower():
                parts = text.split(conj, 1)
                if len(parts) == 2:
                    # Reverse the order
                    return f"{parts[1]}{conj}{parts[0]}"

        return text


def demonstrate_transform_diversity():
    """Demonstrate different transform types on example texts."""

    print("="*80)
    print("DIVERSE TRANSFORM TYPE DEMONSTRATION")
    print("="*80)
    print()
    print("Purpose: Show how different transform types work differently")
    print("Use case: Multi-transform ensemble detection")
    print("="*80)

    # Test examples
    examples = [
        "The movie was excellent and I loved it.",
        "This product is very good but expensive.",
        "The service was terrible and I hated it.",
        "It was bad although the staff was friendly.",
    ]

    # Initialize transforms
    transforms = {
        'LEXICON (Word-level)': [
            ('Prefix Negation', SentimentPrefixNegation()),
            ('Lexicon Flip', SentimentLexiconFlip()),
        ],
        'SEMANTIC (Meaning-level)': [
            ('Semantic Paraphrase', SentimentParaphrase()),
            ('Simple Paraphrase', SimpleSemanticTransform()),
        ],
        'STRUCTURAL (Syntax-level)': [
            ('Clause Reorder', SimpleStructuralTransform()),
        ],
    }

    # Process each example
    for i, text in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}: {text}")
        print(f"{'='*80}")

        for transform_type, transform_list in transforms.items():
            print(f"\n{transform_type}:")
            print("-" * 80)

            for name, transform in transform_list:
                try:
                    transformed = transform(text)
                    changed = text != transformed

                    print(f"  {name}:")
                    print(f"    Original:    {text}")
                    print(f"    Transformed: {transformed}")

                    if changed:
                        # Calculate word-level changes
                        orig_words = set(text.lower().split())
                        trans_words = set(transformed.lower().split())
                        added = trans_words - orig_words
                        removed = orig_words - trans_words

                        if added or removed:
                            print(f"    Changes: +{len(added)} words, -{len(removed)} words")
                            if added:
                                print(f"      Added: {', '.join(list(added)[:5])}")
                            if removed:
                                print(f"      Removed: {', '.join(list(removed)[:5])}")
                        else:
                            print(f"    Changes: Reordered/restructured")
                    else:
                        print(f"    ⚠ No change (transform may not apply)")

                except Exception as e:
                    print(f"    ✗ Error: {e}")

    # Summary
    print("\n" + "="*80)
    print("KEY INSIGHTS FOR MULTI-TRANSFORM ENSEMBLE DETECTION")
    print("="*80)
    print("""
1. LEXICON transforms change words but preserve sentence structure:
   - Fast and deterministic
   - Backdoor triggers in specific words may survive
   - Example: Synonym replacement, word insertion

2. SEMANTIC transforms rephrase entire meanings:
   - Slower but more comprehensive
   - Backdoor triggers embedded in semantics may survive
   - Example: Paraphrasing, back-translation

3. STRUCTURAL transforms change syntax/grammar:
   - Preserves words but changes arrangement
   - Backdoor triggers in word order may be disrupted
   - Example: Passive/active voice, clause reordering

4. WHY DIVERSITY MATTERS:
   - A backdoor that survives lexicon transforms may fail on semantic
   - A backdoor that survives semantic may fail on structural
   - Multi-transform ensemble detects samples that are CONSISTENTLY
     RESISTANT across ALL transform types

5. DETECTION STRATEGY:
   - Apply transforms from each type independently
   - Measure prediction consistency for each type
   - Flag samples that show consistent behavior across ALL types
   - This indicates an artificial robustness pattern (backdoor!)
    """)

    print("="*80)
    print("END DEMONSTRATION")
    print("="*80)


if __name__ == '__main__':
    demonstrate_transform_diversity()
