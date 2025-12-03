#!/usr/bin/env python3
"""Test script for improved sentiment transformations."""

import sys
sys.path.append('/mnt/nw/home/j.li/Poison-Detection')

from poison_detection.data.transforms import (
    transform_registry,
    SentimentPrefixNegation,
    SentimentGrammaticalNegation,
    SentimentStrongLexiconFlip,
    SentimentCombinedTransform,
    SentimentIntensityEnhancement
)


def test_transformations():
    """Test all sentiment transformations with sample texts."""

    # Test examples with various sentiment patterns
    test_samples = [
        "This movie is great",
        "I love this product",
        "The food was excellent",
        "This is the best purchase I have made",
        "I am very happy with this service",
        "The performance was outstanding",
        "This book is terrible",
        "I hate this product",
    ]

    # Test transformations
    transforms = {
        "prefix_negation (old)": SentimentPrefixNegation(),
        "grammatical_negation (new)": SentimentGrammaticalNegation(),
        "strong_lexicon_flip (new)": SentimentStrongLexiconFlip(),
        "combined_flip_negation (new)": SentimentCombinedTransform(),
        "intensity_enhancement (new)": SentimentIntensityEnhancement(),
    }

    print("=" * 80)
    print("TESTING IMPROVED SENTIMENT TRANSFORMATIONS")
    print("=" * 80)
    print()

    for sample in test_samples:
        print(f"Original: {sample}")
        print("-" * 80)

        for name, transform in transforms.items():
            try:
                transformed = transform.transform(sample)
                print(f"  [{name}]")
                print(f"    → {transformed}")
            except Exception as e:
                print(f"  [{name}] ERROR: {e}")

        print()

    # Print summary
    print("=" * 80)
    print("TRANSFORMATION STRENGTH ANALYSIS")
    print("=" * 80)
    print()
    print("1. prefix_negation (OLD): Weak - only adds prefix, sentiment words unchanged")
    print("   Issue: Models often ignore prefixes")
    print()
    print("2. grammatical_negation (NEW): Medium - adds 'not' to verbs")
    print("   Example: 'is great' → 'is not great'")
    print()
    print("3. strong_lexicon_flip (NEW): Strong - replaces sentiment words")
    print("   Example: 'great' → 'terrible', 'love' → 'hate'")
    print()
    print("4. combined_flip_negation (NEW): Very Strong - lexicon + negation")
    print("   Example: 'is great' → 'is not terrible'")
    print()
    print("5. intensity_enhancement (NEW): Strong - lexicon + intensifiers")
    print("   Example: 'great' → 'very terrible'")
    print()

    # Test using registry
    print("=" * 80)
    print("TESTING VIA REGISTRY")
    print("=" * 80)
    print()

    all_transforms = transform_registry.get_all_transforms("sentiment")
    print(f"Total sentiment transformations available: {len(all_transforms)}")
    print("Registered transformations:")
    for name in sorted(all_transforms.keys()):
        print(f"  - {name}")
    print()


if __name__ == "__main__":
    test_transformations()
