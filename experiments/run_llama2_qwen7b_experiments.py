#!/usr/bin/env python3
"""
Dedicated experiment runner for LLaMA-2-7B and Qwen-7B models.

This script runs comprehensive poison detection experiments on larger models
(LLaMA-2-7B and Qwen-7B) to validate generalization of detection methods
beyond the T5-small baseline.

Experiments include:
- Sentiment classification (polarity task)
- Both single-method detection (percentile_high, top_k_low, etc.)
- Ensemble detection approaches
"""

import argparse
import time
import gc
from pathlib import Path
import sys
import torch
import numpy as np
import json
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.ensemble_detector import EnsemblePoisonDetector
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader


# Model configurations for smaller models (compatible with Kronfluence)
LARGE_MODEL_CONFIGS = {
    'llama-2-7b': {
        'name': 'meta-llama/Llama-2-7b-hf',
        'type': 'causal',
        'params': '7B',
        'requires_auth': True,  # May need HF authentication token
    },
    'qwen-7b': {
        'name': 'Qwen/Qwen2.5-7B',
        'type': 'causal',
        'params': '7B',
        'requires_auth': False,
        'trust_remote_code': False
    },
    'tinyllama': {
        'name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'type': 'causal',
        'params': '1.1B',
        'requires_auth': False,
    },
    'qwen-small': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'type': 'causal',
        'params': '0.5B',
        'requires_auth': False,
    },
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run poison detection experiments for LLaMA-2-7B and Qwen-7B'
    )
    parser.add_argument('--models', nargs='+',
                      default=['llama-2-7b', 'qwen-7b'],
                      choices=['llama-2-7b', 'qwen-7b', 'tinyllama', 'qwen-small', 'both'],
                      help='Models to run experiments on')
    parser.add_argument('--task', type=str, default='polarity',
                      help='Task name (default: polarity for sentiment classification)')
    parser.add_argument('--num_train_samples', type=int, default=100,
                      help='Number of training samples (default: 100, matching T5 baseline)')
    parser.add_argument('--num_test_samples', type=int, default=50,
                      help='Number of test samples (default: 50, matching T5 baseline)')
    parser.add_argument('--batch_size', type=int, default=2,
                      help='Batch size (default: 2 for 7B models)')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    parser.add_argument('--detection_methods', nargs='+',
                      default=['percentile_high', 'top_k_low', 'local_outlier_factor',
                               'isolation_forest', 'ensemble_voting'],
                      help='Detection methods to test')
    parser.add_argument('--run_ensemble', action='store_true',
                      help='Also run ensemble detection methods')
    parser.add_argument('--output_dir', type=str,
                      default='experiments/results/llama2_qwen7b',
                      help='Output directory for results')
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Data directory')
    parser.add_argument('--damping_factor', type=float, default=0.01,
                      help='Damping factor for influence computation')
    parser.add_argument('--use_8bit', action='store_true',
                      help='Use 8-bit quantization (recommended for 7B models)')
    parser.add_argument('--use_4bit', action='store_true',
                      help='Use 4-bit quantization (more aggressive memory savings)')
    parser.add_argument('--skip_on_error', action='store_true',
                      help='Skip models that fail instead of stopping')
    parser.add_argument('--hf_token', type=str, default=None,
                      help='HuggingFace token for accessing gated models (LLaMA-2)')
    return parser.parse_args()


def load_data(task: str, data_dir: str, num_train: int, num_test: int):
    """Load training and test data."""
    print(f"\nLoading data from {data_dir}/{task}...")

    train_path = Path(data_dir) / task / "poison_train.jsonl"
    train_loader = JSONLDataLoader(train_path)
    train_samples = train_loader.load()[:num_train]

    test_path = Path(data_dir) / task / "test_data.jsonl"
    test_loader = JSONLDataLoader(test_path)
    test_samples = test_loader.load()[:num_test]

    poison_indices = [i for i, s in enumerate(train_samples)
                     if s.metadata.get('is_poisoned', False)]

    print(f"  Loaded {len(train_samples)} train samples")
    print(f"  Loaded {len(test_samples)} test samples")
    print(f"  Poisoned samples: {len(poison_indices)} ({len(poison_indices)/len(train_samples)*100:.1f}%)")

    return train_samples, test_samples, poison_indices


def create_causal_lm_dataset(samples, tokenizer, max_length=256):
    """
    Create dataset for causal language models.
    Format: "Question: {input}\nAnswer: {label}"
    """
    inputs = []
    labels = []
    label_spaces = []

    for sample in samples:
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


def load_model(model_key: str, config: Dict, args):
    """Load model with appropriate settings for 7B models."""
    print(f"\nLoading {model_key} ({config['params']} parameters)...")
    print(f"  Model: {config['name']}")

    # Configure base kwargs
    kwargs = {}

    if config.get('trust_remote_code', False):
        kwargs['trust_remote_code'] = True

    # Add HF token if provided and needed
    if args.hf_token and config.get('requires_auth', False):
        kwargs['token'] = args.hf_token
        print("  Using HuggingFace authentication token")

    # CRITICAL FIX: Use single device instead of device_map="auto" to avoid
    # "tensors on different devices" errors during influence computation.
    # device_map="auto" can split model across multiple GPUs which breaks Kronfluence.
    device = args.device if hasattr(args, 'device') else 'cuda:0'
    if not torch.cuda.is_available():
        device = 'cpu'

    print(f"  Loading on single device: {device}")

    # Configure quantization using BitsAndBytesConfig for better control
    quantization_config = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Nested quantization for additional memory savings
            bnb_4bit_quant_type="nf4"  # Normal Float 4-bit quantization
        )
        print("  Using 4-bit quantization to reduce memory usage (~75% reduction)")
    elif args.use_8bit:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("  Using 8-bit quantization")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config['name'],
        quantization_config=quantization_config,
        device_map=device,  # Use single device instead of "auto"
        dtype=torch.float16,
        **kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config['name'],
        token=kwargs.get('token', None)
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Enable gradient checkpointing to reduce memory usage
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    print(f"  Model loaded successfully")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    return model, tokenizer


def compute_influence_scores(model, task, train_loader, test_loader,
                            analysis_name, output_dir, damping_factor=0.01):
    """Compute influence scores using Kronfluence."""
    print(f"\nComputing influence scores...")

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
        per_device_batch_size=1,  # Reduced from 2 to 1 to avoid OOM
        overwrite=True  # Overwrite cached results to allow different dataset sizes
    )
    factor_time = time.time() - start_time
    print(f"  Factor computation: {factor_time:.1f}s")

    # Compute pairwise scores
    print("  Computing pairwise influence scores...")
    start_time = time.time()
    scores = analyzer.compute_pairwise_scores(
        train_loader=train_loader,
        test_loader=test_loader,
        scores_name=analysis_name,
        factors_name="ekfac",
        per_device_query_batch_size=1,
        per_device_train_batch_size=4,
        overwrite=True  # Overwrite cached results to allow different dataset sizes
    )
    score_time = time.time() - start_time
    print(f"  Score computation: {score_time:.1f}s")

    # Aggregate scores
    if len(scores.shape) == 2:
        agg_scores = scores.mean(dim=1).cpu().numpy()
    else:
        agg_scores = scores.cpu().numpy()

    print(f"  Influence scores shape: {agg_scores.shape}")
    print(f"  Statistics: min={agg_scores.min():.2f}, max={agg_scores.max():.2f}, "
          f"mean={agg_scores.mean():.2f}, std={agg_scores.std():.2f}")

    return agg_scores, factor_time, score_time


def run_detection_methods(influence_scores: np.ndarray, poison_indices: List[int],
                         methods: List[str]) -> Dict[str, Dict]:
    """Run multiple detection methods and evaluate."""
    # Convert to list of (index, score) tuples for PoisonDetector
    original_scores = [(i, score) for i, score in enumerate(influence_scores)]
    detector = PoisonDetector(original_scores=original_scores, poisoned_indices=set(poison_indices))
    results = {}

    print("\n" + "="*80)
    print("DETECTION RESULTS")
    print("="*80)

    for method in methods:
        if method == 'ensemble_voting':
            continue  # Handle separately

        # Configure method-specific parameters
        kwargs = {}
        if method == 'percentile_high':
            kwargs['threshold'] = 0.85
        elif method == 'percentile_low':
            kwargs['threshold'] = 0.15
        elif method.startswith('top_k'):
            kwargs['k'] = len(poison_indices) * 10

        # Detect
        detected_indices = detector.detect(method=method, **kwargs)

        # Evaluate
        metrics = detector.evaluate(detected_indices)

        results[method] = {
            'detected_indices': list(detected_indices),
            'num_detected': len(detected_indices),
            'metrics': metrics
        }

        # Print results
        print(f"\n{method}:")
        print(f"  Detected: {len(detected_indices)} samples")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1']:.2%}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")

    return results


def run_ensemble_detection(influence_scores: np.ndarray, poison_indices: List[int]) -> Dict:
    """Run ensemble detection method by combining multiple detectors."""
    print("\n" + "="*80)
    print("ENSEMBLE DETECTION")
    print("="*80)

    # Convert to list of (index, score) tuples
    original_scores = [(i, score) for i, score in enumerate(influence_scores)]

    # Run multiple detection methods and collect results
    ensemble_results = []
    methods = [
        ("percentile_high", {"threshold": 0.85}),
        ("top_k_low", {"k": len(poison_indices) * 10}),
        ("local_outlier_factor", {}),
    ]

    for method_name, kwargs in methods:
        detector = PoisonDetector(original_scores=original_scores, poisoned_indices=set(poison_indices))
        detected = detector.detect(method=method_name, **kwargs)
        ensemble_results.append(detected)

    # Soft voting: sample is detected if it appears in majority of methods
    vote_counts = {}
    for detected_set in ensemble_results:
        for idx in detected_set:
            vote_counts[idx] = vote_counts.get(idx, 0) + 1

    # Select samples with votes >= threshold (e.g., 2 out of 3 methods)
    threshold_votes = 2
    detected_indices = {idx for idx, votes in vote_counts.items() if votes >= threshold_votes}

    # Evaluate
    detector = PoisonDetector(original_scores=original_scores, poisoned_indices=set(poison_indices))
    metrics = detector.evaluate(detected_indices)

    print(f"\nEnsemble (Soft Voting, threshold={threshold_votes}/{len(methods)}):")
    print(f"  Detected: {len(detected_indices)} samples")
    print(f"  Precision: {metrics['precision']:.2%}")
    print(f"  Recall: {metrics['recall']:.2%}")
    print(f"  F1 Score: {metrics['f1']:.2%}")

    return {
        'ensemble_soft': {
            'detected_indices': list(detected_indices),
            'num_detected': len(detected_indices),
            'metrics': metrics
        }
    }


def run_single_model_experiment(model_key: str, config: Dict,
                               train_samples, test_samples, poison_indices, args):
    """Run experiment for a single model."""
    print("\n" + "="*80)
    print(f"EXPERIMENT: {model_key.upper()}")
    print("="*80)

    results = {
        'model_key': model_key,
        'model_name': config['name'],
        'model_params': config['params'],
        'task': args.task,
        'num_train': len(train_samples),
        'num_test': len(test_samples),
        'num_poisoned': len(poison_indices),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'running'
    }

    try:
        # Load model
        start_time = time.time()
        model, tokenizer = load_model(model_key, config, args)
        load_time = time.time() - start_time

        # Create datasets
        print("\nCreating datasets...")
        # Reduce max_length to 128 to save memory (especially for Qwen-7B)
        train_dataset = create_causal_lm_dataset(train_samples, tokenizer, max_length=128)
        test_dataset = create_causal_lm_dataset(test_samples, tokenizer, max_length=128)

        # Note: batch_size in DataLoader doesn't matter for Kronfluence, it uses per_device_batch_size
        # Set to 1 to avoid any batching issues
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Create task
        task = ClassificationTask()

        # Compute influence scores
        output_dir = Path(args.output_dir) / args.task / model_key
        output_dir.mkdir(parents=True, exist_ok=True)

        influence_scores, factor_time, score_time = compute_influence_scores(
            model=model,
            task=task,
            train_loader=train_loader,
            test_loader=test_loader,
            analysis_name=f"{model_key}_{args.task}",
            output_dir=output_dir,
            damping_factor=args.damping_factor
        )

        # Run detection methods
        detection_results = run_detection_methods(
            influence_scores=influence_scores,
            poison_indices=poison_indices,
            methods=args.detection_methods
        )

        # Run ensemble if requested
        if args.run_ensemble or 'ensemble_voting' in args.detection_methods:
            ensemble_results = run_ensemble_detection(influence_scores, poison_indices)
            detection_results.update(ensemble_results)

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
                'min': float(influence_scores.min()),
                'max': float(influence_scores.max()),
                'mean': float(influence_scores.mean()),
                'std': float(influence_scores.std())
            }
        })

        # Save results
        results_file = output_dir / f"{args.task}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_file}")

        # Clean up to free GPU memory for next model
        print(f"\n  Cleaning up GPU memory...")

        # Delete all model-related objects explicitly
        if 'analyzer' in locals() and analyzer is not None:
            if hasattr(analyzer, 'model'):
                analyzer.model = None
            if hasattr(analyzer, 'analyzer'):
                analyzer.analyzer = None

        del model
        del tokenizer
        del analyzer
        del train_loader
        del test_loader
        del train_dataset
        del test_dataset

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Call twice to ensure full cleanup

            # Give GPU more time to fully release memory
            time.sleep(5)

            mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  GPU memory: {mem_allocated:.2f}GB / {mem_total:.2f}GB used")

    except Exception as e:
        print(f"\nERROR in {model_key}: {str(e)}")
        import traceback
        traceback.print_exc()

        results['status'] = 'failed'
        results['error'] = str(e)

        # Clean up even on error to free GPU memory for next model
        try:
            print(f"\n  Cleaning up GPU memory...")

            # Delete all model-related objects
            if 'analyzer' in locals() and analyzer is not None:
                if hasattr(analyzer, 'model'):
                    analyzer.model = None
                if hasattr(analyzer, 'analyzer'):
                    analyzer.analyzer = None
                del analyzer

            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'train_loader' in locals():
                del train_loader
            if 'test_loader' in locals():
                del test_loader
            if 'train_dataset' in locals():
                del train_dataset
            if 'test_dataset' in locals():
                del test_dataset

            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # Call twice to ensure full cleanup

                # Give GPU more time to fully release memory
                time.sleep(5)

                mem_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  GPU memory: {mem_allocated:.2f}GB / {mem_total:.2f}GB used")
        except Exception as cleanup_error:
            print(f"  Warning: Cleanup error: {cleanup_error}")

        if not args.skip_on_error:
            raise

    return results


def generate_summary_report(all_results: List[Dict], output_dir: Path, args):
    """Generate summary report comparing results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    summary = {
        'experiment_config': {
            'models': args.models,
            'task': args.task,
            'num_train_samples': args.num_train_samples,
            'num_test_samples': args.num_test_samples,
            'detection_methods': args.detection_methods,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }

    # Save summary
    summary_file = output_dir / f"summary_{args.task}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Print comparison table
    print("\n" + "="*80)
    print("BEST F1 SCORES BY MODEL")
    print("="*80)

    for result in all_results:
        if result['status'] != 'success':
            print(f"{result['model_key']}: FAILED")
            continue

        best_f1 = 0
        best_method = None

        for method, detection in result['detection_results'].items():
            f1 = detection['metrics']['f1']
            if f1 > best_f1:
                best_f1 = f1
                best_method = method

        print(f"{result['model_key']} ({result['model_params']}): F1={best_f1:.2%} ({best_method})")
        print(f"  Timing: {result['timing']['total_time']:.1f}s total")


def main():
    """Main execution function."""
    args = parse_args()

    # Handle 'both' option
    if 'both' in args.models:
        args.models = ['llama-2-7b', 'qwen-7b']

    print("="*80)
    print("LLAMA-2-7B & QWEN-7B POISON DETECTION EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Task: {args.task}")
    print(f"Train samples: {args.num_train_samples}")
    print(f"Test samples: {args.num_test_samples}")
    print(f"Detection methods: {', '.join(args.detection_methods)}")
    if args.use_8bit:
        print("Quantization: 8-bit")
    elif args.use_4bit:
        print("Quantization: 4-bit")
    print()

    # Load data (shared across models)
    train_samples, test_samples, poison_indices = load_data(
        args.task, args.data_dir, args.num_train_samples, args.num_test_samples
    )

    # Run experiments for each model
    all_results = []
    for model_key in args.models:
        if model_key not in LARGE_MODEL_CONFIGS:
            print(f"Warning: Unknown model '{model_key}', skipping...")
            continue

        config = LARGE_MODEL_CONFIGS[model_key]
        result = run_single_model_experiment(
            model_key, config, train_samples, test_samples, poison_indices, args
        )
        all_results.append(result)

    # Generate summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_summary_report(all_results, output_dir, args)

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
