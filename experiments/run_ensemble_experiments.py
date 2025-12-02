#!/usr/bin/env python3
"""
Enhanced experiment runner using ensemble detection with multiple semantic transformations.

This script improves poison detection by:
1. Using multiple semantic transformations (not just one)
2. Computing KL divergence between influence scores across transformations
3. Ensemble voting and combined scoring methods
4. Better detection metrics

Usage:
    python experiments/run_ensemble_experiments.py --model t5-small --task sentiment
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time
import warnings

import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

# Configure numerical stability settings
if torch.cuda.is_available():
    try:
        torch.backends.cuda.preferred_linalg_library('magma')
        print("Using MAGMA backend for linear algebra operations")
    except RuntimeError:
        print("MAGMA not available, using default cusolver backend")
        pass
    torch.set_float32_matmul_precision('high')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from poison_detection.utils.model_utils import load_model_and_tokenizer
from poison_detection.data.poisoner import get_poisoner, PoisonConfig
from poison_detection.data.transforms import transform_registry, get_transform_info
from poison_detection.data.dataset import InstructionDataset, SimpleInstructionDataset
from poison_detection.influence.task import ClassificationTask, SimpleGenerationTask
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.detection.ensemble_detector import EnsemblePoisonDetector


def patch_kronfluence_eigendecomposition():
    """Monkey-patch Kronfluence's eigendecomposition for numerical stability."""
    try:
        import kronfluence.factor.eigen as eigen_module

        original_perform_eigendecomposition = eigen_module.perform_eigendecomposition

        def safe_perform_eigendecomposition(
            covariance_factors,
            damping_factor=None,
            **kwargs
        ):
            """Sanitize covariance matrices before eigendecomposition."""
            sanitized_factors = {}

            for name, factor_dict in covariance_factors.items():
                sanitized_dict = {}
                for key, matrix in factor_dict.items():
                    if isinstance(matrix, torch.Tensor):
                        if torch.isnan(matrix).any() or torch.isinf(matrix).any():
                            print(f"Warning: NaN/inf detected in {name}/{key}, sanitizing...")
                            matrix = torch.nan_to_num(matrix, nan=0.0, posinf=1e10, neginf=-1e10)

                        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                            matrix = (matrix + matrix.T) / 2

                        matrix = torch.clamp(matrix, min=-1e10, max=1e10)

                        if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                            extra_damping = 1e-4
                            matrix = matrix + extra_damping * torch.eye(
                                matrix.shape[0],
                                device=matrix.device,
                                dtype=matrix.dtype
                            )

                    sanitized_dict[key] = matrix
                sanitized_factors[name] = sanitized_dict

            return original_perform_eigendecomposition(
                sanitized_factors,
                damping_factor=damping_factor,
                **kwargs
            )

        eigen_module.perform_eigendecomposition = safe_perform_eigendecomposition
        print("✓ Applied Kronfluence eigendecomposition patch for numerical stability")

    except Exception as e:
        print(f"Warning: Could not patch Kronfluence eigendecomposition: {e}")
        print("Continuing without patch...")


# Apply patch at module load time
patch_kronfluence_eigendecomposition()


# Model name mapping
MODEL_NAMES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-7b": "meta-llama/Meta-Llama-3-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",
    "t5-base": "google/t5-base-lm-adapt",
    "t5-small": "google/t5-small-lm-adapt"
}


class EnsembleExperiment:
    """Enhanced experiment using ensemble detection."""

    def __init__(self, config):
        self.config = config
        self.results = {
            "config": {
                "model": config.model_name,
                "task": config.task_type,
                "attack_type": config.attack_type,
                "transformations": config.transforms,
            },
            "attacks": {},
            "detection": {},
            "runtime": {}
        }

    def load_model(self):
        """Load model and tokenizer."""
        print(f"\n{'='*80}")
        print(f"Loading model: {self.config.model_name}")
        print(f"{'='*80}")

        start_time = time.time()

        model, tokenizer = load_model_and_tokenizer(
            model_name=self.config.model_name,
            use_4bit=self.config.use_4bit,
            use_8bit=self.config.use_8bit,
            trust_remote_code=True
        )

        if self.config.multi_gpu and torch.cuda.device_count() > 1:
            print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        elif torch.cuda.is_available():
            print(f"  Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  Using CPU")

        load_time = time.time() - start_time
        self.results["runtime"]["model_load"] = load_time

        print(f"✓ Model loaded in {load_time:.2f}s")

        model_for_count = model.module if hasattr(model, 'module') else model
        print(f"  Model parameters: {sum(p.numel() for p in model_for_count.parameters()) / 1e6:.1f}M")

        return model, tokenizer

    def load_dataset(self, task_type: str) -> Tuple[List[str], List[str], List[str]]:
        """Load dataset for the task."""
        print(f"\n{'='*80}")
        print(f"Loading {task_type} dataset")
        print(f"{'='*80}")

        if task_type == "sentiment":
            from datasets import load_dataset
            dataset = load_dataset("glue", "sst2", split="train")

            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            inputs = [f"Classify sentiment: {ex['sentence']}" for ex in dataset]
            labels = ["positive" if ex['label'] == 1 else "negative" for ex in dataset]
            label_space = ["positive", "negative"]

        elif task_type == "math":
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="train")

            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            inputs = [ex["question"] for ex in dataset]
            labels = [ex["answer"] for ex in dataset]
            label_space = None

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        print(f"✓ Loaded {len(inputs)} samples")
        return inputs, labels, label_space

    def poison_dataset(
        self,
        inputs: List[str],
        labels: List[str],
        label_space: List[str]
    ) -> Tuple[List[str], List[str], List[int]]:
        """Apply poisoning attack."""
        print(f"\n{'='*80}")
        print(f"Applying {self.config.attack_type} attack")
        print(f"{'='*80}")

        poison_config = PoisonConfig(
            poison_ratio=self.config.poison_ratio,
            attack_type=self.config.attack_type,
            seed=self.config.seed
        )

        if self.config.attack_type == "multi_trigger":
            poison_config.trigger_phrases = ["James Bond", "John Wick", "Ethan Hunt"]
        elif self.config.attack_type == "label_preserving":
            poison_config.style_target = "polite"

        poisoner = get_poisoner(self.config.attack_type, poison_config)
        poisoned_inputs, poisoned_labels, poison_idxs = poisoner.poison_dataset(
            inputs, labels, label_space
        )

        print(f"✓ Poisoned {len(poison_idxs)} samples ({len(poison_idxs)/len(inputs)*100:.2f}%)")

        self.results["attacks"]["num_poisoned"] = len(poison_idxs)
        self.results["attacks"]["poison_indices"] = poison_idxs

        return poisoned_inputs, poisoned_labels, poison_idxs

    def compute_influence_for_transformation(
        self,
        model,
        tokenizer,
        train_inputs: List[str],
        train_labels: List[str],
        test_inputs: List[str],
        test_labels: List[str],
        label_space: List[str],
        transform_name: str = None
    ) -> torch.Tensor:
        """Compute influence scores with optional transformation."""

        # Apply transformation if specified
        if transform_name:
            transform = transform_registry.get_transform(
                self.config.task_type,
                transform_name
            )
            train_inputs = [transform(inp, lbl) for inp, lbl in zip(train_inputs, train_labels)]
            test_inputs = [transform(inp, lbl) for inp, lbl in zip(test_inputs, test_labels)]

        # Create datasets
        if label_space:
            train_dataset = InstructionDataset(
                train_inputs, train_labels,
                [label_space] * len(train_inputs),
                tokenizer
            )
            test_dataset = InstructionDataset(
                test_inputs, test_labels,
                [label_space] * len(test_inputs),
                tokenizer
            )
            task = ClassificationTask()
        else:
            train_dataset = SimpleInstructionDataset(
                train_inputs, train_labels, tokenizer
            )
            test_dataset = SimpleInstructionDataset(
                test_inputs, test_labels, tokenizer
            )
            task = SimpleGenerationTask()

        # Create data loaders
        effective_batch_size = self.config.batch_size
        if self.config.multi_gpu and torch.cuda.device_count() > 1:
            effective_batch_size = self.config.batch_size * torch.cuda.device_count()

        train_loader = TorchDataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available()
        )

        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Compute influences
        analyzer = InfluenceAnalyzer(
            model=model,
            task=task,
            analysis_name=f"poison_detection_{transform_name or 'original'}",
            damping_factor=0.1,
            use_cpu_for_computation=False
        )

        analyzer.compute_factors(train_loader, factors_name="ekfac")
        influence_scores = analyzer.compute_pairwise_scores(
            train_loader=train_loader,
            test_loader=test_loader,
            factors_name="ekfac"
        )

        return influence_scores

    def run_ensemble_detection(
        self,
        model,
        tokenizer,
        train_inputs: List[str],
        train_labels: List[str],
        test_inputs: List[str],
        test_labels: List[str],
        label_space: List[str],
        poison_idxs: List[int]
    ) -> Dict:
        """Run detection using ensemble methods."""
        print(f"\n{'='*80}")
        print(f"Running ensemble detection with {len(self.config.transforms)} transformations")
        print(f"{'='*80}")

        start_time = time.time()

        # Initialize ensemble detector
        detector = EnsemblePoisonDetector(
            poisoned_indices=set(poison_idxs)
        )

        # Compute influence scores for original (no transformation)
        print(f"\n  1/{len(self.config.transforms)+1}: Computing influences (original)")
        original_scores = self.compute_influence_for_transformation(
            model, tokenizer,
            train_inputs, train_labels,
            test_inputs, test_labels,
            label_space,
            transform_name=None
        )
        detector.add_transformation_result(
            "original",
            original_scores,
            len(train_inputs)
        )

        # Compute influence scores for each transformation
        for i, transform_name in enumerate(self.config.transforms, start=2):
            print(f"\n  {i}/{len(self.config.transforms)+1}: Computing influences ({transform_name})")

            try:
                transform_scores = self.compute_influence_for_transformation(
                    model, tokenizer,
                    train_inputs, train_labels,
                    test_inputs, test_labels,
                    label_space,
                    transform_name=transform_name
                )
                detector.add_transformation_result(
                    transform_name,
                    transform_scores,
                    len(train_inputs)
                )
                print(f"    ✓ Transformation '{transform_name}' completed")
            except Exception as e:
                print(f"    ✗ Transformation '{transform_name}' failed: {e}")
                continue

        # Get detection results from all methods
        print(f"\n  Evaluating detection methods...")
        detection_summary = detector.get_detection_summary()

        compute_time = time.time() - start_time
        self.results["runtime"]["ensemble_detection"] = compute_time

        print(f"\n✓ Ensemble detection completed in {compute_time:.2f}s")
        print(f"\n{'='*80}")
        print(f"Detection Results:")
        print(f"{'='*80}")

        for method, metrics in detection_summary.items():
            print(f"\n{method.upper()}:")
            if "error" in metrics:
                print(f"  Error: {metrics['error']}")
            else:
                if "precision" in metrics:
                    print(f"  Detected: {metrics['num_detected']} samples")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall: {metrics['recall']:.4f}")
                    print(f"  F1 Score: {metrics['f1_score']:.4f}")
                else:
                    print(f"  Detected: {metrics.get('num_detected', 'N/A')} samples")

        return detection_summary

    def run(self):
        """Run complete experiment."""
        print(f"\n{'#'*80}")
        print(f"# Running Enhanced Ensemble Poison Detection Experiment")
        print(f"# Experiment: {self.config.model_name.split('/')[-1]}_{self.config.task_type}_{self.config.attack_type}")
        print(f"{'#'*80}")

        # 1. Load model
        model, tokenizer = self.load_model()

        # 2. Load dataset
        inputs, labels, label_space = self.load_dataset(self.config.task_type)

        # 3. Apply poisoning
        poisoned_inputs, poisoned_labels, poison_idxs = self.poison_dataset(
            inputs, labels, label_space
        )

        # 4. Split into train/test
        split_idx = int(len(poisoned_inputs) * 0.8)
        train_inputs = poisoned_inputs[:split_idx]
        train_labels = poisoned_labels[:split_idx]
        test_inputs = poisoned_inputs[split_idx:]
        test_labels = poisoned_labels[split_idx:]

        # Filter poison indices to only include training samples
        train_poison_idxs = [idx for idx in poison_idxs if idx < split_idx]

        # 5. Run ensemble detection
        detection_results = self.run_ensemble_detection(
            model, tokenizer,
            train_inputs, train_labels,
            test_inputs, test_labels,
            label_space,
            train_poison_idxs
        )
        self.results["detection"] = detection_results

        # 6. Save results
        exp_name = f"{self.config.model_name.split('/')[-1]}_{self.config.task_type}_{self.config.attack_type}_ensemble"
        output_file = self.config.output_dir / f"{exp_name}_results.json"

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f"✓ Experiment complete! Results saved to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run ensemble poison detection experiments"
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="t5-small",
        choices=list(MODEL_NAMES.keys()),
        help="Model to use"
    )

    # Task selection
    parser.add_argument(
        "--task", type=str, default="sentiment",
        choices=["sentiment", "math"],
        help="Task type"
    )

    # Attack configuration
    parser.add_argument(
        "--attack-type", type=str, default="single_trigger",
        choices=["single_trigger", "multi_trigger", "label_preserving"],
        help="Type of backdoor attack"
    )
    parser.add_argument(
        "--poison-ratio", type=float, default=0.01,
        help="Ratio of poisoned samples"
    )

    # Transformation configuration
    parser.add_argument(
        "--transforms", nargs="+",
        default=["prefix_negation", "alternative_prefix", "paraphrase", "lexicon_flip"],
        help="Transformations to use in ensemble"
    )

    # Quantization
    parser.add_argument(
        "--use-4bit", action="store_true",
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--use-8bit", action="store_true",
        help="Use 8-bit quantization"
    )

    # Multi-GPU settings
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Use multiple GPUs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size per GPU"
    )

    # General settings
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results",
        help="Output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples"
    )

    args = parser.parse_args()

    # Create config object
    class Config:
        def __init__(self, args):
            self.model_name = MODEL_NAMES[args.model]
            self.task_type = args.task
            self.attack_type = args.attack_type
            self.poison_ratio = args.poison_ratio
            self.transforms = args.transforms
            self.use_4bit = args.use_4bit
            self.use_8bit = args.use_8bit
            self.multi_gpu = args.multi_gpu
            self.batch_size = args.batch_size
            self.output_dir = Path(args.output_dir)
            self.seed = args.seed
            self.max_samples = args.max_samples

            self.output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(args)

    # Run experiment
    experiment = EnsembleExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
