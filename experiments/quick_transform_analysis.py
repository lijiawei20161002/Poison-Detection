"""
Quick transformation analysis to verify semantic transformation effects.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask


def main():
    print("=" * 80)
    print("QUICK SEMANTIC TRANSFORMATION ANALYSIS")
    print("=" * 80)

    # Configuration
    NUM_TRAIN = 50
    NUM_TEST = 20
    BATCH_SIZE = 4
    MODEL_NAME = 'google/t5-small-lm-adapt'

    print(f"Train samples: {NUM_TRAIN}")
    print(f"Test samples: {NUM_TEST}")
    print(f"Model: {MODEL_NAME}")
    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()

    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # Load data
    print("Loading data...")
    train_path = Path("data/polarity/poison_train.jsonl")
    test_path = Path("data/polarity/test_data.jsonl")

    train_loader = JSONLDataLoader(train_path)
    test_loader = JSONLDataLoader(test_path)

    train_samples = train_loader.load()[:NUM_TRAIN]
    test_samples = test_loader.load()[:NUM_TEST]

    print(f"Loaded {len(train_samples)} train, {len(test_samples)} test samples")
    print()

    # Create baseline datasets
    print("Creating baseline datasets...")
    train_dataset = InstructionDataset(
        inputs=[s.input_text for s in train_samples],
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

    # Setup analyzer
    task = ClassificationTask()
    output_dir = Path("experiments/results/quick_transform")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute baseline influence
    print("\n" + "=" * 80)
    print("BASELINE (No Transformation)")
    print("=" * 80)

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name="baseline",
        output_dir=str(output_dir),
        use_cpu_for_computation=False
    )

    print("Computing factors...")
    analyzer.compute_factors(train_dataloader, factors_name="baseline_ekfac", per_device_batch_size=2)

    print("Computing pairwise scores...")
    baseline_scores = analyzer.compute_pairwise_scores(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        scores_name="baseline",
        factors_name="baseline_ekfac",
        per_device_query_batch_size=1,
        per_device_train_batch_size=4
    ).cpu().numpy()

    print(f"Baseline scores shape: {baseline_scores.shape}")
    print(f"Mean absolute influence: {np.abs(baseline_scores).mean():.6f}")
    print(f"Std influence: {np.std(baseline_scores):.6f}")
    print()

    # Test transformations
    results = {
        'baseline': {
            'mean_abs_influence': float(np.abs(baseline_scores).mean()),
            'std_influence': float(np.std(baseline_scores)),
            'max_influence': float(np.max(baseline_scores)),
            'min_influence': float(np.min(baseline_scores))
        }
    }

    # Select key transformations to test
    transforms_to_test = ['prefix_negation', 'lexicon_flip', 'question_negation', 'word_shuffle_failure']

    for transform_name in transforms_to_test:
        print("\n" + "=" * 80)
        print(f"TRANSFORMATION: {transform_name}")
        print("=" * 80)

        try:
            # Get transform function
            transform_fn = transform_registry.get_transform('sentiment', transform_name)

            # Apply to test samples
            print(f"Applying {transform_name}...")
            transformed_inputs = []
            for sample in test_samples:
                try:
                    transformed_inputs.append(transform_fn(sample.input_text))
                except:
                    transformed_inputs.append(sample.input_text)

            # Create transformed test dataset
            test_dataset_transformed = InstructionDataset(
                inputs=transformed_inputs,
                labels=[s.output_text for s in test_samples],
                label_spaces=[s.label_space if s.label_space else ["positive", "negative"] for s in test_samples],
                tokenizer=tokenizer,
                max_input_length=128,
                max_output_length=32
            )

            test_dataloader_transformed = DataLoader(
                test_dataset_transformed, batch_size=BATCH_SIZE, shuffle=False
            )

            # Compute influence with transformed test set
            analyzer_transformed = InfluenceAnalyzer(
                model=model,
                task=task,
                analysis_name=f"transform_{transform_name}",
                output_dir=str(output_dir),
                use_cpu_for_computation=False
            )

            print("Computing pairwise scores with transformed test set...")
            transformed_scores = analyzer_transformed.compute_pairwise_scores(
                train_loader=train_dataloader,
                test_loader=test_dataloader_transformed,
                scores_name=f"transform_{transform_name}",
                factors_name="baseline_ekfac",  # Reuse baseline factors
                per_device_query_batch_size=1,
                per_device_train_batch_size=4
            ).cpu().numpy()

            # Calculate metrics
            influence_change = np.abs(transformed_scores - baseline_scores)
            mean_change = float(influence_change.mean())
            max_change = float(influence_change.max())

            # Calculate correlation
            correlation = float(np.corrcoef(baseline_scores.flatten(), transformed_scores.flatten())[0, 1])

            results[transform_name] = {
                'mean_abs_influence': float(np.abs(transformed_scores).mean()),
                'std_influence': float(np.std(transformed_scores)),
                'max_influence': float(np.max(transformed_scores)),
                'min_influence': float(np.min(transformed_scores)),
                'mean_influence_change': mean_change,
                'max_influence_change': max_change,
                'correlation_with_baseline': correlation
            }

            print(f"Mean absolute influence: {results[transform_name]['mean_abs_influence']:.6f}")
            print(f"Mean influence change from baseline: {mean_change:.6f}")
            print(f"Max influence change: {max_change:.6f}")
            print(f"Correlation with baseline: {correlation:.4f}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_file = output_dir / "transformation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Transform':<30} {'Mean Change':>15} {'Max Change':>15} {'Correlation':>15}")
    print("-" * 80)

    for name in transforms_to_test:
        if name in results:
            r = results[name]
            print(f"{name:<30} {r['mean_influence_change']:>15.6f} "
                  f"{r['max_influence_change']:>15.6f} {r['correlation_with_baseline']:>15.4f}")

    print()
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
