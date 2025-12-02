"""
Systematic ablation study of semantic transformations for poison detection.

This script addresses reviewer concerns by:
1. Testing all transformations systematically
2. Providing quantitative metrics for each transformation
3. Comparing transformations across multiple criteria
4. Generating comprehensive reports and visualizations

Usage:
    python experiments/run_transformation_ablation.py --task sentiment --model t5-small
    python experiments/run_transformation_ablation.py --task math --model deepseek-coder
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
from typing import Dict, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry, get_transform_info
from poison_detection.data.dataset import PoisonDataset
from poison_detection.data.loader import create_dataloaders
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.evaluation.transform_evaluator import (
    TransformationEvaluator,
    TransformationQualityMetrics
)
from poison_detection.utils.file_utils import load_poisoned_indices


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Systematic transformation ablation study'
    )

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['sentiment', 'math', 'qa'],
        help='Task type to evaluate'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='google/t5-small-lm-adapt',
        help='Model name or path'
    )

    parser.add_argument(
        '--data_dir',
        type=str,
        default='./data',
        help='Directory containing poisoned data'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./experiments/results/ablation',
        help='Output directory for results'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for influence computation'
    )

    parser.add_argument(
        '--num_test_samples',
        type=int,
        default=100,
        help='Number of test samples to use'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--transformations',
        type=str,
        nargs='+',
        default=None,
        help='Specific transformations to test (if None, test all)'
    )

    parser.add_argument(
        '--skip_original',
        action='store_true',
        help='Skip computing original (baseline) influence scores'
    )

    parser.add_argument(
        '--load_factors',
        action='store_true',
        help='Load precomputed influence factors'
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name: str, device: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}")

    # Determine if it's a seq2seq or causal model
    if 't5' in model_name.lower():
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    model.eval()

    return model, tokenizer


def apply_transformation_to_dataset(
    dataset: PoisonDataset,
    transform_fn,
    tokenizer
) -> PoisonDataset:
    """
    Apply transformation to dataset.

    Args:
        dataset: Original dataset
        transform_fn: Transformation function
        tokenizer: Tokenizer

    Returns:
        Transformed dataset
    """
    # Create new dataset with transformed texts
    transformed_texts = []
    for item in dataset.dataset:
        text = item['text'] if 'text' in item else item['input']
        transformed_text = transform_fn(text)
        transformed_texts.append(transformed_text)

    # Create new dataset with same structure
    transformed_dataset = PoisonDataset(
        dataset=dataset.dataset,
        tokenizer=tokenizer,
        poison_indices=dataset.poison_indices
    )

    # Update texts
    for i, text in enumerate(transformed_texts):
        transformed_dataset.dataset[i]['text'] = text

    return transformed_dataset


def compute_influence_scores(
    model,
    task: ClassificationTask,
    train_loader: DataLoader,
    test_loader: DataLoader,
    analysis_name: str,
    output_dir: Path,
    load_factors: bool = False
) -> np.ndarray:
    """
    Compute influence scores using Kronfluence.

    Args:
        model: Language model
        task: Task definition
        train_loader: Training data loader
        test_loader: Test data loader
        analysis_name: Name for this analysis
        output_dir: Output directory
        load_factors: Whether to load precomputed factors

    Returns:
        Influence scores matrix (n_train, n_test)
    """
    print(f"  Computing influence scores for: {analysis_name}")

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name=analysis_name,
        output_dir=output_dir
    )

    # Compute or load factors
    if not load_factors:
        print("  Computing influence factors...")
        analyzer.compute_factors(train_loader, factors_name="ekfac")
    else:
        print("  Loading precomputed factors...")
        analyzer.load_factors(factors_name="ekfac")

    # Compute pairwise scores
    print("  Computing pairwise influence scores...")
    influence_scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac"
    )

    return influence_scores.cpu().numpy()


def run_ablation_study(args):
    """Run systematic ablation study."""

    print("=" * 80)
    print("SYSTEMATIC TRANSFORMATION ABLATION STUDY")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print()

    # Setup output directory
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Load poisoned dataset
    print("Loading dataset...")
    data_dir = Path(args.data_dir) / args.task
    poisoned_indices = load_poisoned_indices(data_dir / 'poisoned_indices.txt')

    # Create task
    task = ClassificationTask(
        model=model,
        tokenizer=tokenizer
    )

    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_test_samples=args.num_test_samples,
        poison_indices=poisoned_indices
    )

    print(f"Loaded {len(train_loader.dataset)} training samples")
    print(f"Loaded {len(test_loader.dataset)} test samples")
    print(f"Number of poisons: {len(poisoned_indices)}")
    print()

    # Get transformations to test
    available_transforms = transform_registry.get_all_transforms(args.task)

    if args.transformations:
        transforms_to_test = {
            name: available_transforms[name]
            for name in args.transformations
            if name in available_transforms
        }
    else:
        transforms_to_test = available_transforms

    print(f"Testing {len(transforms_to_test)} transformations:")
    for name in transforms_to_test.keys():
        print(f"  - {name}")
    print()

    # Initialize evaluator
    evaluator = TransformationEvaluator(
        poisoned_indices=poisoned_indices,
        output_dir=output_dir
    )

    # Compute baseline (original) influence scores
    print("=" * 80)
    print("COMPUTING BASELINE (ORIGINAL) INFLUENCE SCORES")
    print("=" * 80)

    original_scores = None
    if not args.skip_original:
        start_time = time.time()
        original_scores = compute_influence_scores(
            model=model,
            task=task,
            train_loader=train_loader,
            test_loader=test_loader,
            analysis_name="original",
            output_dir=output_dir / "influence",
            load_factors=args.load_factors
        )
        baseline_time = time.time() - start_time
        print(f"  Baseline computation time: {baseline_time:.2f}s")
        print()

    # Test each transformation
    results = []

    for i, (transform_name, transform_fn) in enumerate(transforms_to_test.items(), 1):
        print("=" * 80)
        print(f"TESTING TRANSFORMATION {i}/{len(transforms_to_test)}: {transform_name}")
        print("=" * 80)

        try:
            # Apply transformation to test data
            start_time = time.time()

            print(f"  Applying transformation to test data...")
            transformed_test_loader = apply_transformation_to_dataset(
                test_loader.dataset,
                transform_fn,
                tokenizer
            )
            transformed_test_loader = DataLoader(
                transformed_test_loader,
                batch_size=args.batch_size
            )

            # Compute influence scores with transformed test data
            transformed_scores = compute_influence_scores(
                model=model,
                task=task,
                train_loader=train_loader,
                test_loader=transformed_test_loader,
                analysis_name=f"transformed_{transform_name}",
                output_dir=output_dir / "influence",
                load_factors=True  # Reuse factors
            )

            computation_time = time.time() - start_time

            # Evaluate transformation quality
            if original_scores is not None:
                print(f"  Evaluating transformation quality...")
                metrics = evaluator.evaluate_transformation(
                    transform_name=transform_name,
                    task_type=args.task,
                    original_scores=original_scores,
                    transformed_scores=transformed_scores,
                    computation_time=computation_time
                )

                results.append(metrics)

                # Print summary
                print(f"\n  Results:")
                print(f"    Influence Correlation: {metrics.influence_correlation:.3f}")
                print(f"    Sign Flip Ratio: {metrics.sign_flip_ratio:.3f}")
                print(f"    F1 Score: {metrics.f1_score:.3f}")
                print(f"    Precision: {metrics.precision:.3f}")
                print(f"    Recall: {metrics.recall:.3f}")
                print(f"    ROC AUC: {metrics.roc_auc:.3f}")
                print(f"    Separation Score: {metrics.separation_score:.3f}")
                print()

        except Exception as e:
            print(f"  ERROR: Failed to test transformation '{transform_name}'")
            print(f"  Error message: {str(e)}")
            import traceback
            traceback.print_exc()
            print()
            continue

    # Generate comparison and report
    print("=" * 80)
    print("GENERATING RESULTS")
    print("=" * 80)

    if results:
        # Save results
        print("Saving results...")
        evaluator.save_results(output_dir / 'transformation_results.json')

        # Create comparison table
        comparison_df = evaluator.compare_transformations(results)
        comparison_df.to_csv(output_dir / 'transformation_comparison.csv', index=False)
        print(f"  Saved comparison table to: {output_dir / 'transformation_comparison.csv'}")

        # Generate visualizations
        print("Generating visualizations...")
        evaluator.plot_transformation_comparison(
            results,
            output_path=output_dir / 'transformation_comparison.png'
        )
        print(f"  Saved visualization to: {output_dir / 'transformation_comparison.png'}")

        # Generate markdown report
        print("Generating report...")
        report = evaluator.generate_report()
        with open(output_dir / 'evaluation_report.md', 'w') as f:
            f.write(report)
        print(f"  Saved report to: {output_dir / 'evaluation_report.md'}")

        # Print top 5 transformations
        print("\n" + "=" * 80)
        print("TOP 5 TRANSFORMATIONS (by F1 Score)")
        print("=" * 80)

        top_5 = comparison_df.head(5)
        for i, row in enumerate(top_5.itertuples(), 1):
            print(f"\n{i}. {row.name}")
            print(f"   F1 Score: {row.f1_score:.3f}")
            print(f"   Precision: {row.precision:.3f}, Recall: {row.recall:.3f}")
            print(f"   Correlation: {row.influence_correlation:.3f}")
            print(f"   Sign Flip: {row.sign_flip_ratio:.3f}")
            print(f"   Separation: {row.separation_score:.3f}")

    else:
        print("No successful results to report.")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(args)
