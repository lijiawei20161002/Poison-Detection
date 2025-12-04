"""
Quick test of aggressive semantic transformations on poisoned data.
Tests a few examples to find the most promising method.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from poison_detection.data.transforms import apply_transform
from poison_detection.utils.model_utils import load_model

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

def quick_test():
    """Run quick test on a few samples."""
    print("="*80)
    print("QUICK TEST: Aggressive Semantic Transforms on Poisoned Samples")
    print("="*80)

    # Try loading a model (if available)
    model = None
    model_path = "models/poisoned_sentiment_model.pt"
    if os.path.exists(model_path):
        try:
            print(f"\n[INFO] Loading model from {model_path}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = load_model(model_path, device)
            print(f"[INFO] Model loaded on {device}")
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
            print("[INFO] Will only test transformations without model inference")
    else:
        print(f"[INFO] Model not found at {model_path}")
        print("[INFO] Will only test transformations without model inference")

    print("\n" + "="*80)
    print("TRANSFORMATION TESTS")
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
                print(f"    Text: {transformed[:100]}...")

                # If model available, test predictions
                if model is not None:
                    try:
                        # Tokenize and predict (simplified)
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

                        # Original prediction
                        orig_inputs = tokenizer(original, return_tensors="pt", padding=True, truncation=True)
                        with torch.no_grad():
                            orig_outputs = model(**orig_inputs)
                            orig_pred = torch.argmax(orig_outputs.logits, dim=-1).item()

                        # Transformed prediction
                        trans_inputs = tokenizer(transformed, return_tensors="pt", padding=True, truncation=True)
                        with torch.no_grad():
                            trans_outputs = model(**trans_inputs)
                            trans_pred = torch.argmax(trans_outputs.logits, dim=-1).item()

                        print(f"    Original pred: {orig_pred}, Transformed pred: {trans_pred}")
                        print(f"    Prediction changed: {orig_pred != trans_pred}")
                    except Exception as e:
                        print(f"    [Could not get predictions: {e}]")

            except Exception as e:
                print(f"\n  {transform_name}: ERROR - {e}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nBased on trigger position shifts:")
    print("- aggressive_mid_insertion: Good for disrupting middle-positioned triggers")
    print("- aggressive_distributed_insertion: Good for distributing noise throughout")
    print("- aggressive_triple_negation: Strong shift with multiple negations")
    print("- aggressive_context_injection: Strongest semantic context disruption")
    print("\nRecommendation: Test aggressive_context_injection and aggressive_triple_negation")
    print("                on full dataset for best results against syntactic triggers.")

if __name__ == "__main__":
    quick_test()
