"""
Semantic transformation experiments for poison detection.

This script runs influence analysis experiments with various semantic transformations
to verify and document their effectiveness.
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Semantic transformation experiments'
    )
    parser.add_argument('--task', type=str, default='polarity')
    parser.add_argument('--model', type=str, default='google/t5-small-lm-adapt')
    parser.add_argument('--num_train_samples', type=int, default=100)
    parser.add_argument('--num_test_samples', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--transforms', nargs='+', default=None)
    parser.add_argument('--output_dir', type=str, default='experiments/results/transform_ablation')
    parser.add_argument('--data_dir', type=str, default='data')
    return parser.parse_args()


def load_data(task: str, data_dir: str, num_train: int, num_test: int):
    """Load training and test data."""
    # Load train data
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]
    
    # Load test data
    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]
    
    return train_samples, test_samples


def create_torch_dataset(samples, tokenizer, max_input_len=128, max_output_len=32):
    """Create PyTorch dataset from samples."""
    inputs = [s.input_text for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [s.label_space if s.label_space else ["positive", "negative"] for s in samples]
    
    return InstructionDataset(
        inputs=inputs,
        labels=labels,
        label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=max_input_len,
        max_output_length=max_output_len
    )


def apply_transform_to_samples(samples, transform_fn):
    """Apply transformation to text samples."""
    transformed = []
    success_count = 0
    for sample in samples:
        try:
            transformed_text = transform_fn(sample.input_text)
            # Create new sample with transformed input
            from poison_detection.data.loader import DataSample
            new_sample = DataSample(
                input_text=transformed_text,
                output_text=sample.output_text,
                task=sample.task,
                label_space=sample.label_space,
                countnorm=sample.countnorm,
                sample_id=sample.sample_id,
                metadata=sample.metadata
            )
            transformed.append(new_sample)
            success_count += 1
        except Exception as e:
            transformed.append(sample)  # Keep original if transform fails
    
    print(f"  Successfully transformed {success_count}/{len(samples)} samples")
    return transformed


def compute_influence_scores(
    model, task, train_loader, test_loader, 
    analysis_name, output_dir
):
    """Compute influence scores."""
    print(f"  Computing influence for: {analysis_name}")
    
    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name=analysis_name,
        output_dir=output_dir,
        use_cpu_for_computation=False
    )
    
    # Compute factors
    print("  Computing influence factors...")
    analyzer.compute_factors(train_loader, factors_name="ekfac", per_device_batch_size=2)
    
    # Compute pairwise scores
    print("  Computing pairwise influence scores...")
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac",
        per_device_query_batch_size=1,
        per_device_train_batch_size=4
    )
    
    return scores.cpu().numpy()


def run_experiments(args):
    """Run transformation experiments."""
    print("=" * 80)
    print("SEMANTIC TRANSFORMATION EXPERIMENTS")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    print()
    
    # Setup
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    
    # Load model
    print("Loading model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    
    # Load data
    print("Loading data...")
    train_samples, test_samples = load_data(
        args.task, args.data_dir, args.num_train_samples, args.num_test_samples
    )
    print(f"Loaded {len(train_samples)} train samples, {len(test_samples)} test samples")
    
    # Create task
    task = ClassificationTask()
    
    # Get transforms to test
    if args.transforms:
        transforms = args.transforms
    else:
        # Get sentiment transforms for polarity task
        transforms = transform_registry.get_all_transforms('sentiment')
    
    print(f"Testing {len(transforms)} transformations: {transforms}")
    print()
    
    results = {}
    
    # Baseline (no transformation)
    print("\n" + "=" * 80)
    print("BASELINE (No Transformation)")
    print("=" * 80)
    
    train_dataset = create_torch_dataset(train_samples, tokenizer)
    test_dataset = create_torch_dataset(test_samples, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    baseline_scores = compute_influence_scores(
        model, task, train_loader, test_loader,
        "baseline", output_dir
    )
    
    results['baseline'] = {
        'scores': baseline_scores,
        'mean_abs_influence': float(np.abs(baseline_scores).mean()),
        'max_influence': float(np.max(baseline_scores)),
        'min_influence': float(np.min(baseline_scores)),
        'std_influence': float(np.std(baseline_scores))
    }
    
    print(f"Baseline mean absolute influence: {results['baseline']['mean_abs_influence']:.6f}")
    print(f"Baseline std influence: {results['baseline']['std_influence']:.6f}")
    
    # Test each transformation
    for transform_name in transforms:
        print("\n" + "=" * 80)
        print(f"TRANSFORMATION: {transform_name}")
        print("=" * 80)
        
        try:
            # Get transform function
            transform_fn = transform_registry.get_transform('sentiment', transform_name)
            
            # Apply to test samples only
            transformed_test = apply_transform_to_samples(test_samples, transform_fn)
            
            # Create datasets
            test_dataset_transformed = create_torch_dataset(transformed_test, tokenizer)
            test_loader_transformed = DataLoader(
                test_dataset_transformed, batch_size=args.batch_size, shuffle=False
            )
            
            # Compute influence with transformed test set
            transformed_scores = compute_influence_scores(
                model, task, train_loader, test_loader_transformed,
                f"transform_{transform_name}", output_dir
            )
            
            # Calculate metrics
            influence_change = np.abs(transformed_scores - baseline_scores)
            mean_change = float(influence_change.mean())
            max_change = float(influence_change.max())
            
            # Calculate correlation
            baseline_flat = baseline_scores.flatten()
            transformed_flat = transformed_scores.flatten()
            correlation = float(np.corrcoef(baseline_flat, transformed_flat)[0, 1])
            
            results[transform_name] = {
                'scores': transformed_scores,
                'mean_abs_influence': float(np.abs(transformed_scores).mean()),
                'max_influence': float(np.max(transformed_scores)),
                'min_influence': float(np.min(transformed_scores)),
                'std_influence': float(np.std(transformed_scores)),
                'mean_influence_change': mean_change,
                'max_influence_change': max_change,
                'correlation_with_baseline': correlation
            }
            
            print(f"Mean absolute influence: {results[transform_name]['mean_abs_influence']:.6f}")
            print(f"Mean influence change: {mean_change:.6f}")
            print(f"Max influence change: {max_change:.6f}")
            print(f"Correlation with baseline: {correlation:.4f}")
            
        except Exception as e:
            print(f"ERROR: Failed to test transform '{transform_name}': {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    output_file = output_dir / "transformation_results.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, val in results.items():
            json_results[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                for k, v in val.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\n\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Transform':<30} {'Mean Change':>15} {'Max Change':>15} {'Correlation':>15}")
    print("-" * 80)
    
    for transform_name in transforms:
        if transform_name in results:
            r = results[transform_name]
            print(f"{transform_name:<30} {r['mean_influence_change']:>15.6f} "
                  f"{r['max_influence_change']:>15.6f} {r['correlation_with_baseline']:>15.4f}")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
