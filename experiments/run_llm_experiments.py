#!/usr/bin/env python3
"""
Comprehensive experiment runner for modern LLMs (LLaMA-3, Qwen2).

This script addresses reviewer feedback by:
1. Testing on modern open LLMs (LLaMA-3-8B-Instruct, Qwen2-7B-Instruct)
2. Supporting multiple attack settings (single/multi-trigger, label-preserving)
3. Enabling systematic transformation ablations

Usage:
    python experiments/run_llm_experiments.py --model llama3-8b --task sentiment
    python experiments/run_llm_experiments.py --model qwen2-7b --task math --attack-type multi_trigger
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from poison_detection.utils.model_utils import load_model_and_tokenizer
from poison_detection.data.poisoner import get_poisoner, PoisonConfig
from poison_detection.data.transforms import transform_registry, get_transform_info
from poison_detection.data.dataset import InstructionDataset, SimpleInstructionDataset
from poison_detection.data.loader import DataLoader
from poison_detection.influence.task import ClassificationTask, SimpleGenerationTask
from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.detection.detector import PoisonDetector
from poison_detection.detection.metrics import DetectionMetrics


# Model name mapping
MODEL_NAMES = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-7b": "meta-llama/Meta-Llama-3-7B-Instruct",  # Alternative
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "qwen2-1.5b": "Qwen/Qwen2-1.5B-Instruct",  # Smaller for testing
    "t5-base": "google/t5-base-lm-adapt",
    "t5-small": "google/t5-small-lm-adapt"
}


class ExperimentConfig:
    """Configuration for LLM experiments."""

    def __init__(self, args):
        self.model_name = MODEL_NAMES.get(args.model, args.model)
        self.task_type = args.task
        self.attack_type = args.attack_type
        self.poison_ratio = args.poison_ratio
        self.use_4bit = args.use_4bit
        self.use_8bit = args.use_8bit
        self.transforms = args.transforms
        self.output_dir = Path(args.output_dir)
        self.seed = args.seed
        self.max_samples = args.max_samples

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment name
        self.exp_name = f"{args.model}_{args.task}_{args.attack_type}"


class LLMExperiment:
    """Main experiment runner for LLM poison detection."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {
            "config": {
                "model": config.model_name,
                "task": config.task_type,
                "attack_type": config.attack_type,
                "poison_ratio": config.poison_ratio,
            },
            "attacks": {},
            "transforms": {},
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

        load_time = time.time() - start_time
        self.results["runtime"]["model_load"] = load_time

        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

        return model, tokenizer

    def load_dataset(self, task_type: str) -> Tuple[List[str], List[str], List[str]]:
        """Load dataset for the task."""
        print(f"\n{'='*80}")
        print(f"Loading {task_type} dataset")
        print(f"{'='*80}")

        if task_type == "sentiment":
            # Load sentiment classification data (e.g., SST-2, IMDB)
            from datasets import load_dataset
            dataset = load_dataset("glue", "sst2", split="train")

            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            inputs = [f"Classify sentiment: {ex['sentence']}" for ex in dataset]
            labels = ["positive" if ex['label'] == 1 else "negative" for ex in dataset]
            label_space = ["positive", "negative"]

        elif task_type == "math":
            # Load math reasoning data (GSM8K)
            from datasets import load_dataset
            dataset = load_dataset("gsm8k", "main", split="train")

            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))

            inputs = [ex["question"] for ex in dataset]
            labels = [ex["answer"] for ex in dataset]
            label_space = None  # Open-ended generation

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

        # Configure attack
        poison_config = PoisonConfig(
            poison_ratio=self.config.poison_ratio,
            attack_type=self.config.attack_type,
            seed=self.config.seed
        )

        # Set attack-specific parameters
        if self.config.attack_type == "multi_trigger":
            poison_config.trigger_phrases = ["James Bond", "John Wick", "Ethan Hunt"]
        elif self.config.attack_type == "label_preserving":
            poison_config.style_target = "polite"

        # Get poisoner and apply
        poisoner = get_poisoner(self.config.attack_type, poison_config)
        poisoned_inputs, poisoned_labels, poison_idxs = poisoner.poison_dataset(
            inputs, labels, label_space
        )

        print(f"✓ Poisoned {len(poison_idxs)} samples ({len(poison_idxs)/len(inputs)*100:.2f}%)")

        self.results["attacks"]["num_poisoned"] = len(poison_idxs)
        self.results["attacks"]["poison_indices"] = poison_idxs

        return poisoned_inputs, poisoned_labels, poison_idxs

    def compute_influence(
        self,
        model,
        tokenizer,
        train_inputs: List[str],
        train_labels: List[str],
        test_inputs: List[str],
        test_labels: List[str],
        label_space: List[str]
    ) -> torch.Tensor:
        """Compute influence scores using Kronfluence."""
        print(f"\n{'='*80}")
        print(f"Computing influence scores with Kronfluence (EK-FAC)")
        print(f"{'='*80}")

        start_time = time.time()

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
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        # Compute influences
        analyzer = InfluenceAnalyzer(
            model=model,
            task=task,
            factor_strategy="ekfac"  # Use EK-FAC for efficiency
        )

        print("  1. Computing factors...")
        analyzer.fit_factors(train_loader)

        print("  2. Computing influence scores...")
        influence_scores = analyzer.compute_influences(
            train_loader=train_loader,
            query_loader=test_loader
        )

        compute_time = time.time() - start_time
        self.results["runtime"]["influence_computation"] = compute_time

        print(f"✓ Influence computation completed in {compute_time:.2f}s")
        print(f"  Influence matrix shape: {influence_scores.shape}")

        return influence_scores

    def test_transforms(
        self,
        inputs: List[str],
        labels: List[str],
        transform_names: List[str]
    ) -> Dict[str, Dict]:
        """Test multiple transformations systematically."""
        print(f"\n{'='*80}")
        print(f"Testing {len(transform_names)} transformations")
        print(f"{'='*80}")

        results = {}

        for transform_name in transform_names:
            print(f"\n  Testing: {transform_name}")

            transform = transform_registry.get_transform(
                self.config.task_type,
                transform_name
            )

            # Apply transform
            transformed = [transform(inp, lbl) for inp, lbl in zip(inputs[:10], labels[:10])]

            results[transform_name] = {
                "description": transform.config.description,
                "expected_to_work": transform.config.expected_to_work,
                "samples": [
                    {"original": inputs[i], "transformed": transformed[i]}
                    for i in range(min(3, len(transformed)))
                ]
            }

        return results

    def detect_poisons(
        self,
        influence_scores: torch.Tensor,
        poison_indices: List[int],
        threshold: float = 0.1
    ) -> Dict:
        """Detect poisons using influence-invariance."""
        print(f"\n{'='*80}")
        print(f"Detecting poisons with influence-invariance")
        print(f"{'='*80}")

        detector = PoisonDetector(
            influence_scores=influence_scores,
            threshold=threshold
        )

        detected = detector.detect()

        # Compute metrics
        metrics = DetectionMetrics()
        precision, recall, f1 = metrics.compute_metrics(
            true_poisons=poison_indices,
            detected_poisons=detected
        )

        results = {
            "num_detected": len(detected),
            "detected_indices": detected,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        print(f"✓ Detection complete:")
        print(f"  Detected: {len(detected)} samples")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")

        return results

    def run(self):
        """Run complete experiment."""
        print(f"\n{'#'*80}")
        print(f"# Running LLM Poison Detection Experiment")
        print(f"# Experiment: {self.config.exp_name}")
        print(f"{'#'*80}")

        # 1. Load model
        model, tokenizer = self.load_model()

        # 2. Load dataset
        inputs, labels, label_space = self.load_dataset(self.config.task_type)

        # 3. Apply poisoning
        poisoned_inputs, poisoned_labels, poison_idxs = self.poison_dataset(
            inputs, labels, label_space
        )

        # 4. Test transformations (ablation)
        if self.config.transforms:
            transform_results = self.test_transforms(
                poisoned_inputs, poisoned_labels, self.config.transforms
            )
            self.results["transforms"] = transform_results

        # 5. Compute influences
        # Split into train/test
        split_idx = int(len(poisoned_inputs) * 0.8)
        train_inputs = poisoned_inputs[:split_idx]
        train_labels = poisoned_labels[:split_idx]
        test_inputs = poisoned_inputs[split_idx:]
        test_labels = poisoned_labels[split_idx:]

        influence_scores = self.compute_influence(
            model, tokenizer,
            train_inputs, train_labels,
            test_inputs, test_labels,
            label_space
        )

        # 6. Detect poisons
        detection_results = self.detect_poisons(
            influence_scores,
            [idx for idx in poison_idxs if idx < split_idx]
        )
        self.results["detection"] = detection_results

        # 7. Save results
        output_file = self.config.output_dir / f"{self.config.exp_name}_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f"✓ Experiment complete! Results saved to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run poison detection experiments on modern LLMs"
    )

    # Model selection
    parser.add_argument(
        "--model", type=str, default="llama3-8b",
        choices=list(MODEL_NAMES.keys()) + ["custom"],
        help="Model to use (or 'custom' with --model-name)"
    )
    parser.add_argument(
        "--model-name", type=str,
        help="Custom model name (use with --model custom)"
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

    # Transformation ablation
    parser.add_argument(
        "--transforms", nargs="+",
        help="Transformations to test (e.g., prefix_negation lexicon_flip)"
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

    # General settings
    parser.add_argument(
        "--output-dir", type=str, default="experiments/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples (for testing)"
    )

    args = parser.parse_args()

    # Handle custom model name
    if args.model == "custom":
        if not args.model_name:
            raise ValueError("Must specify --model-name when using --model custom")
        MODEL_NAMES["custom"] = args.model_name

    # Create experiment config
    config = ExperimentConfig(args)

    # Run experiment
    experiment = LLMExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
