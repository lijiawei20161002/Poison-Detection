"""
Quick test of aggressive semantic transformations on sample texts.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poison_detection.data.transforms import apply_transform

# Test samples with known syntactic triggers
test_samples = [
    "This movie was cf great and entertaining",  # Trigger: cf
    "I cf really enjoyed this film",
    "The plot was cf amazing",
    "cf This was a wonderful experience",
]

# Aggressive transforms to test
transforms_to_test = [
    "aggressive_double_negation",
    "aggressive_triple_negation",
    "aggressive_mid_insertion",
    "aggressive_distributed_insertion",
    "aggressive_prefix_suffix_mixed",
    "aggressive_context_injection",
]

def main():
    print("="*80)
    print("AGGRESSIVE SEMANTIC TRANSFORMS - QUICK TEST")
    print("="*80)

    for i, original in enumerate(test_samples, 1):
        print(f"\n{'='*80}")
        print(f"Sample {i}: {original}")
        print(f"{'='*80}")

        # Find trigger position
        words = original.split()
        trigger_pos = None
        for j, word in enumerate(words):
            if 'cf' in word.lower():
                trigger_pos = j
                break

        print(f"Original trigger position: {trigger_pos}")

        # Test each transform
        results = []
        for transform_name in transforms_to_test:
            try:
                transformed = apply_transform(original, "sentiment", transform_name)

                # Find new trigger position
                new_words = transformed.split()
                new_trigger_pos = None
                for j, word in enumerate(new_words):
                    if 'cf' in word.lower():
                        new_trigger_pos = j
                        break

                position_change = new_trigger_pos - trigger_pos if (trigger_pos is not None and new_trigger_pos is not None) else None

                print(f"\n  {transform_name}:")
                print(f"    New position: {new_trigger_pos} (shift: {position_change})")
                print(f"    Length change: {len(new_words) - len(words)} words")
                print(f"    Text: {transformed[:120]}...")

                results.append({
                    'name': transform_name,
                    'shift': abs(position_change) if position_change else 0,
                    'length_change': len(new_words) - len(words)
                })

            except Exception as e:
                print(f"\n  {transform_name}: ERROR - {e}")

        # Summary for this sample
        print(f"\n  Best shifts:")
        results_sorted = sorted(results, key=lambda x: x['shift'], reverse=True)
        for r in results_sorted[:3]:
            print(f"    {r['name']}: shift={r['shift']}, added {r['length_change']} words")

    print("\n" + "="*80)
    print("OVERALL SUMMARY")
    print("="*80)
    print("\nMost promising transforms for disrupting syntactic triggers:")
    print("1. aggressive_context_injection - Adds strong semantic context")
    print("2. aggressive_triple_negation - Maximum negation prefix/suffix")
    print("3. aggressive_distributed_insertion - Distributes NOTs throughout")
    print("\nNext step: Test these on actual poisoned dataset with model evaluation.")

if __name__ == "__main__":
    main()
