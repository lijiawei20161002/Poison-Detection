"""
Poison detection experiments for Qwen models.

This script runs influence-based poison detection on Qwen models to compare
with the baseline T5-small results documented in the README.
"""

import argparse
import time
from pathlib import Path
import sys
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Poison detection experiments for Qwen models'
    )
    parser.add_argument('--task', type=str, default='polarity',
                      help='Task name (polarity, sentiment, etc.)')
    parser.add_argument('--model', type=str,
                      default='Qwen/Qwen2.5-1.5B',
                      help='Qwen model variant (Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, etc.)')
    parser.add_argument('--num_train_samples', type=int, default=100,
                      help='Number of training samples')
    parser.add_argument('--num_test_samples', type=int, default=50,
                      help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use (cuda/cpu)')
    parser.add_argument('--detection_methods', nargs='+',
                      default=['percentile_high', 'top_k_low', 'local_outlier_factor'],
                      help='Detection methods to test')
    parser.add_argument('--output_dir', type=str,
                      default='experiments/results/qwen',
                      help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Data directory')
    parser.add_argument('--damping_factor', type=float, default=0.01,
                      help='Damping factor for influence computation')
    parser.add_argument('--use_8bit', action='store_true',
                      help='Use 8-bit quantization to reduce memory')
    parser.add_argument('--use_4bit', action='store_true',
                      help='Use 4-bit quantization to reduce memory')
    parser.add_argument('--trust_remote_code', action='store_true',
                      help='Trust remote code when loading model')
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

    # Get poison indices
    poison_indices = [i for i, s in enumerate(train_samples) if s.metadata.get('is_poisoned', False)]

    return train_samples, test_samples, poison_indices


def create_causal_lm_dataset(samples, tokenizer, max_length=256):
    """
    Create dataset for causal language models (Qwen).

    For classification tasks, we format as:
    "Question: {input}\nAnswer: {label}"
    """
    inputs = []
    labels = []
    label_spaces = []

    for sample in samples:
        # Format input with instruction-following style
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


def load_qwen_model(model_name: str, use_8bit: bool = False, use_4bit: bool = False, trust_remote_code: bool = False):
    """Load Qwen model with optional quantization."""
    print(f"Loading Qwen model: {model_name}")

    # Configure quantization if requested
    kwargs = {}
    if use_8bit:
        kwargs['load_in_8bit'] = True
        print("  Using 8-bit quantization")
    elif use_4bit:
        kwargs['load_in_4bit'] = True
        print("  Using 4-bit quantization")

    if trust_remote_code:
        kwargs['trust_remote_code'] = True

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code
    )

    # Set padding token (Qwen doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"  Model loaded successfully")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


def compute_influence_scores(
    model, task, train_loader, test_loader,
    analysis_name, output_dir, damping_factor=0.01
):
    """Compute influence scores using Kronfluence."""
    print(f"Computing influence scores...")

    analyzer = InfluenceAnalyzer(
        model=model,
        task=task,
        analysis_name=analysis_name,
        output_dir=output_dir,
        damping_factor=damping_factor,
        use_cpu_for_computation=False
    )

    # Compute factors
    print("  Computing influence factors...")
    start_time = time.time()
    analyzer.compute_factors(
        train_loader,
        factors_name="ekfac",
        per_device_batch_size=2
    )
    factor_time = time.time() - start_time
    print(f"  Factor computation time: {factor_time:.1f}s")

    # Compute pairwise scores
    print("  Computing pairwise influence scores...")
    start_time = time.time()
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac",
        per_device_query_batch_size=1,
        per_device_train_batch_size=4
    )
    score_time = time.time() - start_time
    print(f"  Score computation time: {score_time:.1f}s")

    # Aggregate scores (mean across test samples)
    if len(scores.shape) == 2:
        agg_scores = scores.mean(dim=1).cpu().numpy()
    else:
        agg_scores = scores.cpu().numpy()

    print(f"  Influence scores computed: shape={agg_scores.shape}")
    print(f"  Score statistics: min={agg_scores.min():.2f}, max={agg_scores.max():.2f}, "
          f"mean={agg_scores.mean():.2f}, std={agg_scores.std():.2f}")

    return agg_scores, factor_time, score_time


def run_detection_methods(
    influence_scores: np.ndarray,
    poison_indices: List[int],
    methods: List[str]
) -> Dict[str, Dict]:
    """Run multiple detection methods and evaluate."""
    detector = PoisonDetector()
    results = {}

    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)

    for method in methods:
        # Configure method-specific parameters
        kwargs = {}
        if method == 'percentile_high':
            kwargs['threshold'] = 0.85
        elif method == 'percentile_low':
            kwargs['threshold'] = 0.15
        elif method.startswith('top_k'):
            # Use number of actual poisons as k
            kwargs['k'] = len(poison_indices) * 10  # 10x the number of poisons

        # Detect
        detected = detector.detect_poisons(
            influence_scores=influence_scores,
            method=method,
            **kwargs
        )

        # Evaluate
        metrics = detector.evaluate_detection(detected, poison_indices)

        results[method] = {
            'detected_indices': detected.tolist() if isinstance(detected, np.ndarray) else detected,
            'num_detected': len(detected),
            'metrics': metrics
        }

        # Print results
        print(f"\n{method}:")
        print(f"  Detected: {len(detected)} samples")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1']:.2%}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")

    return results


def save_results(results: Dict, output_dir: Path, args):
    """Save experimental results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comprehensive results dict
    full_results = {
        'model': args.model,
        'task': args.task,
        'num_train_samples': args.num_train_samples,
        'num_test_samples': args.num_test_samples,
        'damping_factor': args.damping_factor,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'results': results
    }

    # Save to JSON
    output_file = output_dir / f"{args.task}_results.json"
    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


def run_experiments(args):
    """Run complete poison detection experiments for Qwen."""
    print("="*80)
    print("QWEN POISON DETECTION EXPERIMENTS")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"Device: {args.device}")
    print()

    # Setup
    output_dir = Path(args.output_dir) / args.task
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_qwen_model(
        args.model,
        use_8bit=args.use_8bit,
        use_4bit=args.use_4bit,
        trust_remote_code=args.trust_remote_code
    )

    # Load data
    print("\nLoading data...")
    train_samples, test_samples, poison_indices = load_data(
        args.task, args.data_dir,
        args.num_train_samples, args.num_test_samples
    )
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples: {len(test_samples)}")
    print(f"  Poisoned samples: {len(poison_indices)} ({len(poison_indices)/len(train_samples)*100:.1f}%)")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_causal_lm_dataset(train_samples, tokenizer)
    test_dataset = create_causal_lm_dataset(test_samples, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create task
    task = ClassificationTask()

    # Compute influence scores
    print("\n" + "="*80)
    print("COMPUTING INFLUENCE SCORES")
    print("="*80)
    influence_scores, factor_time, score_time = compute_influence_scores(
        model=model,
        task=task,
        train_loader=train_loader,
        test_loader=test_loader,
        analysis_name=f"qwen_{args.task}",
        output_dir=output_dir,
        damping_factor=args.damping_factor
    )

    # Run detection methods
    detection_results = run_detection_methods(
        influence_scores=influence_scores,
        poison_indices=poison_indices,
        methods=args.detection_methods
    )

    # Add timing information
    detection_results['timing'] = {
        'factor_computation': factor_time,
        'score_computation': score_time,
        'total': factor_time + score_time
    }

    # Save results
    save_results(detection_results, output_dir, args)

    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total computation time: {factor_time + score_time:.1f}s")
    print(f"Best method: {max(detection_results.items(), key=lambda x: x[1].get('metrics', {}).get('f1', 0) if isinstance(x[1], dict) and 'metrics' in x[1] else 0)[0] if detection_results else 'N/A'}")
    print("="*80)


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
