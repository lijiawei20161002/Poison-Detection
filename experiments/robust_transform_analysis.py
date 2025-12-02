"""
Robust transformation analysis with numerical stability fixes.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask


def validate_gradients(model):
    """Check for NaN/Inf in model gradients."""
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"  WARNING: NaN gradient in {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                print(f"  WARNING: Inf gradient in {name}")
                has_inf = True
    return not (has_nan or has_inf)


def clip_gradients(model, max_norm=1.0):
    """Clip gradients to prevent numerical instability."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def check_for_nans(tensor, name="tensor"):
    """Check if tensor contains NaN values."""
    if torch.isnan(tensor).any():
        print(f"  WARNING: NaN detected in {name}")
        return True
    return False


def main():
    print("=" * 80)
    print("ROBUST SEMANTIC TRANSFORMATION ANALYSIS")
    print("=" * 80)

    # Configuration
    NUM_TRAIN = 30  # Reduced for stability
    NUM_TEST = 15   # Reduced for stability
    BATCH_SIZE = 2  # Smaller batches for numerical stability
    MODEL_NAME = 'google/t5-small-lm-adapt'
    USE_CPU_FALLBACK = True  # Use CPU if CUDA fails

    print(f"Train samples: {NUM_TRAIN}")
    print(f"Test samples: {NUM_TEST}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print(f"CPU fallback enabled: {USE_CPU_FALLBACK}")
    print()

    # Try CUDA first, fall back to CPU if issues
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Primary device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Keep model on CPU initially for factor computation
    # (CUDA has numerical issues with eigendecomposition)
    print("Using CPU for influence computation to avoid CUDA numerical errors")
    model = model.cpu()
    model.eval()

    # Load data
    print("Loading data...")
    train_path = Path("data/polarity/poison_train.jsonl")
    test_path = Path("data/polarity/test_data.jsonl")

    if not train_path.exists():
        print(f"ERROR: Training data not found at {train_path}")
        return
    if not test_path.exists():
        print(f"ERROR: Test data not found at {test_path}")
        return

    train_loader = JSONLDataLoader(train_path)
    test_loader = JSONLDataLoader(test_path)

    train_samples = train_loader.load()[:NUM_TRAIN]
    test_samples = test_loader.load()[:NUM_TEST]

    print(f"Loaded {len(train_samples)} train, {len(test_samples)} test samples")
    print()

    # Get list of sentiment transformations
    transforms_info = transform_registry.list_transforms()
    sentiment_transforms = transforms_info.get("sentiment", [])
    print(f"Found {len(sentiment_transforms)} sentiment transformations")
    print()

    results = {}

    # Test each transformation
    for transform_name in ["prefix_negation", "suffix_reversal", "but_negation"]:  # Start with 3 for testing
        if transform_name not in sentiment_transforms:
            print(f"Skipping {transform_name} - not found in registry")
            continue

        print("\n" + "=" * 80)
        print(f"TESTING TRANSFORMATION: {transform_name}")
        print("=" * 80)

        try:
            # Apply transformation to training data
            print(f"Applying {transform_name} to training data...")
            transform_fn = transform_registry.get_transform("sentiment", transform_name)

            # Show example transformation
            example_text = train_samples[0].input_text
            transformed_example = transform_fn(example_text)
            print(f"Example transformation:")
            print(f"  Original: {example_text[:100]}...")
            print(f"  Transformed: {transformed_example[:100]}...")
            print()

            # Create transformed datasets
            transformed_inputs = [transform_fn(s.input_text) for s in train_samples]

            train_dataset = InstructionDataset(
                inputs=transformed_inputs,
                labels=[s.output_text for s in train_samples],
                label_spaces=[s.label_space if s.label_space else ["positive", "negative"] for s in train_samples],
                tokenizer=tokenizer,
                max_input_length=128,
                max_output_length=32
            )

            test_dataset = InstructionDataset(
                inputs=[s.input_text for s in test_samples],  # Don't transform test data
                labels=[s.output_text for s in test_samples],
                label_spaces=[s.label_space if s.label_space else ["positive", "negative"] for s in test_samples],
                tokenizer=tokenizer,
                max_input_length=128,
                max_output_length=32
            )

            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            # Setup analyzer with CPU computation
            task = ClassificationTask()
            output_dir = Path(f"experiments/results/transform_robust/{transform_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Creating analyzer (using CPU for numerical stability)...")
            analyzer = InfluenceAnalyzer(
                model=model,
                task=task,
                analysis_name=transform_name,
                output_dir=output_dir,
                damping_factor=1e-3,  # Increased damping for stability
                use_cpu_for_computation=True  # Force CPU computation
            )

            # Compute factors
            print("Computing influence factors...")
            try:
                analyzer.compute_factors(
                    train_dataloader,
                    factors_name=f"{transform_name}_ekfac",
                    per_device_batch_size=BATCH_SIZE,
                    overwrite=True
                )
                print(f"✓ Successfully computed factors for {transform_name}")

                # Compute influence scores
                print("Computing pairwise influence scores...")
                scores = analyzer.compute_pairwise_scores(
                    train_dataloader,
                    test_dataloader,
                    scores_name=f"{transform_name}_scores",
                    factors_name=f"{transform_name}_ekfac",
                    per_device_query_batch_size=1,
                    per_device_train_batch_size=BATCH_SIZE,
                    overwrite=True
                )

                # Compute statistics
                avg_influence = scores.mean(dim=1)
                std_influence = scores.std(dim=1)

                results[transform_name] = {
                    "success": True,
                    "mean_influence": float(avg_influence.mean()),
                    "std_influence": float(avg_influence.std()),
                    "min_influence": float(avg_influence.min()),
                    "max_influence": float(avg_influence.max()),
                    "score_shape": list(scores.shape)
                }

                print(f"✓ Influence statistics:")
                print(f"  Mean: {results[transform_name]['mean_influence']:.6f}")
                print(f"  Std: {results[transform_name]['std_influence']:.6f}")
                print(f"  Range: [{results[transform_name]['min_influence']:.6f}, {results[transform_name]['max_influence']:.6f}]")

            except RuntimeError as e:
                if "cusolver" in str(e) or "CUDA" in str(e):
                    print(f"✗ CUDA error encountered: {str(e)[:100]}")
                    print("  Recommendation: Already using CPU, but still getting CUDA errors.")
                    print("  This may indicate a deeper numerical stability issue.")
                results[transform_name] = {
                    "success": False,
                    "error": str(e)[:200]
                }

        except Exception as e:
            print(f"✗ Error processing {transform_name}: {str(e)}")
            results[transform_name] = {
                "success": False,
                "error": str(e)[:200]
            }

    # Save results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    output_file = Path("experiments/results/transform_robust_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"\nSuccessful transformations: {sum(1 for r in results.values() if r.get('success', False))}/{len(results)}")

    # Print summary table
    print("\nTransformation Results:")
    print("-" * 80)
    for name, result in results.items():
        status = "✓" if result.get('success', False) else "✗"
        if result.get('success', False):
            print(f"{status} {name:20s} | Mean influence: {result['mean_influence']:10.6f}")
        else:
            error_msg = result.get('error', 'Unknown error')[:50]
            print(f"{status} {name:20s} | Error: {error_msg}")


if __name__ == "__main__":
    main()
