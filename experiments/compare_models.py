"""
Unified comparison script for poison detection across T5, LLaMA, and Qwen models.

This script runs identical poison detection experiments across different model families
and generates a comparative analysis report.
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
import json
import pandas as pd
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import InfluenceDetector
from poison_detection.utils.model_utils import load_model_and_tokenizer


# Model configurations
MODEL_CONFIGS = {
    't5-small': {
        'name': 'google/t5-small-lm-adapt',
        'type': 'seq2seq',
        'params': '77M',
        'is_baseline': True
    },
    'llama-1b': {
        'name': 'meta-llama/Llama-3.2-1B',
        'type': 'causal',
        'params': '1B',
        'is_baseline': False
    },
    'llama-3b': {
        'name': 'meta-llama/Llama-3.2-3B',
        'type': 'causal',
        'params': '3B',
        'is_baseline': False
    },
    'llama-2-7b': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'type': 'causal',
        'params': '7B',
        'is_baseline': False
    },
    'qwen-0.5b': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'type': 'causal',
        'params': '0.5B',
        'is_baseline': False,
        'trust_remote_code': False  # Qwen2.5 doesn't need trust_remote_code
    },
    'qwen-1.5b': {
        'name': 'Qwen/Qwen2.5-1.5B',
        'type': 'causal',
        'params': '1.5B',
        'is_baseline': False,
        'trust_remote_code': False
    },
    'qwen-7b': {
        'name': 'Qwen/Qwen2.5-7B',
        'type': 'causal',
        'params': '7B',
        'is_baseline': False,
        'trust_remote_code': False
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare poison detection across different model families'
    )
    parser.add_argument('--models', nargs='+',
                      default=['t5-small', 'llama-1b', 'qwen-0.5b'],
                      choices=list(MODEL_CONFIGS.keys()),
                      help='Models to compare')
    parser.add_argument('--task', type=str, default='polarity',
                      help='Task name (polarity, sentiment, etc.)')
    parser.add_argument('--num_train_samples', type=int, default=100,
                      help='Number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                      help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size')
    parser.add_argument('--detection_methods', nargs='+',
                      default=['percentile_high', 'top_k_low', 'local_outlier_factor'],
                      help='Detection methods to test')
    parser.add_argument('--output_dir', type=str,
                      default='experiments/results/model_comparison',
                      help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Data directory')
    parser.add_argument('--damping_factor', type=float, default=0.01,
                      help='Damping factor for influence computation')
    parser.add_argument('--use_8bit', action='store_true',
                      help='Use 8-bit quantization')
    parser.add_argument('--skip_on_error', action='store_true',
                      help='Skip models that fail instead of stopping')
    return parser.parse_args()


def load_data(task: str, data_dir: str, num_train: int, num_test: int):
    """Load training and test data."""
    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    poison_indices = [i for i, s in enumerate(train_samples) if s.metadata.get('is_poisoned', False)]

    return train_samples, test_samples, poison_indices


def create_dataset_for_model(samples, tokenizer, model_type: str, max_length=256):
    """Create dataset appropriate for model type."""
    inputs = []
    labels = []
    label_spaces = []

    for sample in samples:
        if model_type == 'seq2seq':
            # T5-style: input as-is
            formatted_input = sample.input_text
        else:
            # Causal LM style: format with instruction
            formatted_input = f"Question: {sample.input_text}\nAnswer:"

        inputs.append(formatted_input)
        labels.append(sample.output_text)
        label_spaces.append(sample.label_space if sample.label_space else ["positive", "negative"])

    return InstructionDataset(
        inputs=inputs,
        labels=labels,
        label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=max_length,
        max_output_length=32
    )


def run_single_model_experiment(
    model_key: str,
    model_config: Dict,
    train_samples: List,
    test_samples: List,
    poison_indices: List[int],
    detection_methods: List[str],
    args
) -> Dict:
    """Run experiment for a single model."""
    print("\n" + "="*80)
    print(f"RUNNING EXPERIMENT: {model_key.upper()}")
    print("="*80)
    print(f"Model: {model_config['name']}")
    print(f"Type: {model_config['type']}")
    print(f"Parameters: {model_config['params']}")
    print()

    results = {
        'model_key': model_key,
        'model_name': model_config['name'],
        'model_type': model_config['type'],
        'model_params': model_config['params'],
        'status': 'running'
    }

    try:
        # Load model
        print("Loading model...")
        start_load = time.time()
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_config['name'],
            use_8bit=args.use_8bit,
            trust_remote_code=model_config.get('trust_remote_code', False)
        )
        load_time = time.time() - start_load
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create datasets
        print("Creating datasets...")
        train_dataset = create_dataset_for_model(train_samples, tokenizer, model_config['type'])
        test_dataset = create_dataset_for_model(test_samples, tokenizer, model_config['type'])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Create task
        task = ClassificationTask()

        # Compute influence scores
        print("\nComputing influence scores...")
        output_dir = Path(args.output_dir) / args.task / model_key
        output_dir.mkdir(parents=True, exist_ok=True)

        analyzer = InfluenceAnalyzer(
            model=model,
            task=task,
            analysis_name=f"{model_key}_{args.task}",
            output_dir=output_dir,
            damping_factor=args.damping_factor,
            use_cpu_for_computation=False
        )

        # Compute factors
        print("  Computing factors...")
        start_factor = time.time()
        analyzer.compute_factors(train_loader, factors_name="ekfac", per_device_batch_size=2)
        factor_time = time.time() - start_factor
        print(f"  Factor time: {factor_time:.1f}s")

        # Compute scores
        print("  Computing scores...")
        start_score = time.time()
        scores = analyzer.compute_pairwise_scores(
            train_loader=train_loader,
            test_loader=test_loader,
            scores_name=f"{model_key}_scores",
            factors_name="ekfac",
            per_device_query_batch_size=1,
            per_device_train_batch_size=4
        )
        score_time = time.time() - start_score
        print(f"  Score time: {score_time:.1f}s")

        # Aggregate scores
        if len(scores.shape) == 2:
            agg_scores = scores.mean(dim=1).cpu().numpy()
        else:
            agg_scores = scores.cpu().numpy()

        print(f"  Score stats: min={agg_scores.min():.2f}, max={agg_scores.max():.2f}, "
              f"mean={agg_scores.mean():.2f}, std={agg_scores.std():.2f}")

        # Run detection methods
        print("\nRunning detection methods...")
        detector = InfluenceDetector()
        detection_results = {}

        for method in detection_methods:
            kwargs = {}
            if method == 'percentile_high':
                kwargs['threshold'] = 0.85
            elif method == 'percentile_low':
                kwargs['threshold'] = 0.15
            elif method.startswith('top_k'):
                kwargs['k'] = len(poison_indices) * 10

            detected = detector.detect_poisons(agg_scores, method=method, **kwargs)
            metrics = detector.evaluate_detection(detected, poison_indices)

            detection_results[method] = {
                'num_detected': len(detected),
                'metrics': metrics
            }

            print(f"  {method}: F1={metrics['f1']:.2%}, "
                  f"Precision={metrics['precision']:.2%}, Recall={metrics['recall']:.2%}")

        # Update results
        results.update({
            'status': 'success',
            'timing': {
                'load_time': load_time,
                'factor_time': factor_time,
                'score_time': score_time,
                'total_time': load_time + factor_time + score_time
            },
            'detection_results': detection_results,
            'score_stats': {
                'min': float(agg_scores.min()),
                'max': float(agg_scores.max()),
                'mean': float(agg_scores.mean()),
                'std': float(agg_scores.std())
            }
        })

        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        results['status'] = 'failed'
        results['error'] = str(e)

        if not args.skip_on_error:
            raise

    return results


def generate_comparison_report(all_results: List[Dict], output_dir: Path, args):
    """Generate comparison report with tables and analysis."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80)

    # Create DataFrame for easier analysis
    rows = []
    for result in all_results:
        if result['status'] != 'success':
            continue

        for method, detection in result['detection_results'].items():
            rows.append({
                'Model': result['model_key'],
                'Model Name': result['model_name'],
                'Parameters': result['model_params'],
                'Detection Method': method,
                'F1 Score': detection['metrics']['f1'],
                'Precision': detection['metrics']['precision'],
                'Recall': detection['metrics']['recall'],
                'True Positives': detection['metrics']['true_positives'],
                'False Positives': detection['metrics']['false_positives'],
                'Load Time (s)': result['timing']['load_time'],
                'Factor Time (s)': result['timing']['factor_time'],
                'Score Time (s)': result['timing']['score_time'],
                'Total Time (s)': result['timing']['total_time']
            })

    df = pd.DataFrame(rows)

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'detailed_results.csv', index=False)
    print(f"\nDetailed results saved to: {output_dir / 'detailed_results.csv'}")

    # Generate summary tables
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON (F1 Scores)")
    print("="*80)
    pivot_f1 = df.pivot_table(
        values='F1 Score',
        index='Model',
        columns='Detection Method',
        aggfunc='mean'
    )
    print(pivot_f1.to_string())

    print("\n" + "="*80)
    print("TIMING COMPARISON")
    print("="*80)
    timing_summary = df.groupby('Model')[['Load Time (s)', 'Factor Time (s)',
                                           'Score Time (s)', 'Total Time (s)']].first()
    print(timing_summary.to_string())

    print("\n" + "="*80)
    print("BEST RESULTS PER MODEL")
    print("="*80)
    best_results = df.loc[df.groupby('Model')['F1 Score'].idxmax()]
    print(best_results[['Model', 'Detection Method', 'F1 Score', 'Precision', 'Recall']].to_string(index=False))

    # Save summary report
    report_path = output_dir / 'comparison_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Train samples: {args.num_train_samples}\n")
        f.write(f"Test samples: {args.num_test_samples}\n")
        f.write(f"Detection methods: {', '.join(args.detection_methods)}\n\n")

        f.write("="*80 + "\n")
        f.write("F1 SCORES BY MODEL AND METHOD\n")
        f.write("="*80 + "\n")
        f.write(pivot_f1.to_string() + "\n\n")

        f.write("="*80 + "\n")
        f.write("TIMING COMPARISON\n")
        f.write("="*80 + "\n")
        f.write(timing_summary.to_string() + "\n\n")

        f.write("="*80 + "\n")
        f.write("BEST RESULTS PER MODEL\n")
        f.write("="*80 + "\n")
        f.write(best_results[['Model', 'Detection Method', 'F1 Score', 'Precision', 'Recall']].to_string(index=False))

    print(f"\nComparison report saved to: {report_path}")

    # Save raw results as JSON
    json_path = output_dir / 'raw_results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Raw results saved to: {json_path}")


def main():
    """Main execution function."""
    args = parse_args()

    print("="*80)
    print("POISON DETECTION MODEL COMPARISON")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Task: {args.task}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"Detection methods: {', '.join(args.detection_methods)}")
    print()

    # Load data once (shared across all models)
    print("Loading data...")
    train_samples, test_samples, poison_indices = load_data(
        args.task, args.data_dir, args.num_train_samples, args.num_test_samples
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    print(f"  Poisoned samples: {len(poison_indices)} ({len(poison_indices)/len(train_samples)*100:.1f}%)")

    # Run experiments for each model
    all_results = []
    for model_key in args.models:
        model_config = MODEL_CONFIGS[model_key]
        result = run_single_model_experiment(
            model_key, model_config,
            train_samples, test_samples, poison_indices,
            args.detection_methods, args
        )
        all_results.append(result)

    # Generate comparison report
    output_dir = Path(args.output_dir) / args.task
    generate_comparison_report(all_results, output_dir, args)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
