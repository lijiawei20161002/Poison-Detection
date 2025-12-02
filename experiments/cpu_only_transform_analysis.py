"""
CPU-only transformation analysis to avoid CUDA numerical issues.
"""
import os
# Force CPU-only mode BEFORE importing torch
os.environ['CUDA_VISIBLE_DEVICES'] = ''

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


def main():
    print("=" * 80)
    print("CPU-ONLY SEMANTIC TRANSFORMATION ANALYSIS")
    print("=" * 80)
    print("CUDA disabled - using CPU only for all computations")
    print()

    # Configuration
    NUM_TRAIN = 30
    NUM_TEST = 15
    BATCH_SIZE = 2
    MODEL_NAME = 'google/t5-small-lm-adapt'

    print(f"Train samples: {NUM_TRAIN}")
    print(f"Test samples: {NUM_TEST}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Model: {MODEL_NAME}")
    print()

    # Verify CPU-only mode
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print()

    # Load model on CPU
    print("Loading model on CPU...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    results = {}

    # Test each transformation
    transform_names = ["prefix_negation", "lexicon_flip", "double_negation"]

    for transform_name in transform_names:
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
            print(f"  Original: {example_text[:80]}...")
            print(f"  Transformed: {transformed_example[:80]}...")
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
                inputs=[s.input_text for s in test_samples],
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
            output_dir = Path(f"experiments/results/cpu_only_transforms/{transform_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            print("Creating analyzer (CPU-only mode)...")
            analyzer = InfluenceAnalyzer(
                model=model,
                task=task,
                analysis_name=transform_name,
                output_dir=output_dir,
                damping_factor=1e-3,
                use_cpu_for_computation=True
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

            except Exception as e:
                print(f"✗ Error computing influence: {str(e)[:200]}")
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

    output_file = Path("experiments/results/cpu_only_transform_summary.json")
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
