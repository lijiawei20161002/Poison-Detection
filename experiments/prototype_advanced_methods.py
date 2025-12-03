#!/usr/bin/env python3
"""
Quick prototype to validate advanced detection methods.

This script implements simplified versions of 3 advanced methods:
1. Gradient Norm Analysis (fast alternative to influence)
2. Influence Trajectory Analysis (uses existing influence matrix)
3. Token Ablation Analysis (targets syntactic backdoors)

Usage:
    python prototype_advanced_methods.py --task polarity --num_samples 100
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Set
from dataclasses import dataclass
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import time

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.influence.analyzer import InfluenceAnalyzer
from poison_detection.influence.task import ClassificationTask
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def load_poisoned_data(data_path: str):
    """Load poisoned data from JSONL file."""
    data = []

    # Load poisoned indices from file
    poison_indices_file = Path(data_path).parent / "poisoned_indices.txt"
    poison_indices = set()
    if poison_indices_file.exists():
        with open(poison_indices_file, 'r') as f:
            poison_indices = {int(line.strip()) for line in f if line.strip()}

    with open(data_path, 'r') as f:
        for idx, line in enumerate(f):
            item = json.loads(line)

            # Extract text and label
            if 'Instance' in item:
                text = item['Instance']['input']
                label = item['Instance']['output']
            else:
                text = item.get('text', item.get('input', ''))
                label = item.get('label', item.get('output', ''))

            is_poison = idx in poison_indices

            data.append({
                'text': text,
                'label': label if isinstance(label, str) else str(label),
                'label_text': label if isinstance(label, str) else str(label),
                'is_poison': is_poison
            })

    return data, poison_indices


@dataclass
class DetectionResult:
    """Results from a detection method."""
    method_name: str
    f1_score: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    detected_indices: Set[int]
    computation_time: float


def compute_metrics(detected: Set[int], ground_truth: Set[int], n_total: int) -> Dict:
    """Compute detection metrics."""
    tp = len(detected & ground_truth)
    fp = len(detected - ground_truth)
    fn = len(ground_truth - detected)
    tn = n_total - tp - fp - fn

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'num_detected': len(detected)
    }


class GradientNormDetector:
    """
    Method 4.1: Gradient Norm Analysis

    Analyzes gradient norms instead of full influence computation.
    10-100x faster than influence computation.

    Key insight:
    - Clean samples: Low gradient norm or high variance
    - Poisoned samples: High gradient norm + low variance (consistent impact)
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def compute_gradient_features(
        self,
        train_data: list,
        test_data: list,
        batch_size: int = 8
    ) -> np.ndarray:
        """
        Compute gradient norm features for each training sample.

        Returns:
            features: (n_train, 3) array with [mean_norm, std_norm, consistency]
        """
        print(f"Computing gradient norms for {len(train_data)} train samples...")

        features = []

        for i, train_sample in enumerate(train_data):
            if i % 20 == 0:
                print(f"  Progress: {i}/{len(train_data)}")

            grad_norms = []

            # Sample subset of test data for efficiency
            test_subset = np.random.choice(len(test_data), min(10, len(test_data)), replace=False)

            for test_idx in test_subset:
                test_sample = test_data[test_idx]

                # Compute gradient of test loss w.r.t. model parameters
                self.model.zero_grad()

                # Forward pass on test sample
                test_inputs = self.tokenizer(
                    test_sample['text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)

                test_labels = self.tokenizer(
                    test_sample['label_text'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=8
                ).to(self.device)

                outputs = self.model(**test_inputs, labels=test_labels['input_ids'])
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Compute gradient norm
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5

                grad_norms.append(total_norm)

            # Compute features
            mean_norm = np.mean(grad_norms)
            std_norm = np.std(grad_norms)
            consistency = 1 / (std_norm + 1e-6)  # Higher = more consistent

            features.append([mean_norm, std_norm, consistency])

        return np.array(features)

    def detect(
        self,
        train_data: list,
        test_data: list,
        ground_truth: Set[int],
        contamination: float = 0.1
    ) -> DetectionResult:
        """Run gradient norm detection."""
        start_time = time.time()

        # Compute features
        features = self.compute_gradient_features(train_data, test_data)

        # Use Isolation Forest on 3D feature space
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features)

        # Convert to set of detected indices
        detected = set(np.where(predictions == -1)[0])

        # Compute metrics
        metrics = compute_metrics(detected, ground_truth, len(train_data))

        elapsed = time.time() - start_time

        return DetectionResult(
            method_name='gradient_norm_analysis',
            detected_indices=detected,
            computation_time=elapsed,
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            true_positives=metrics['true_positives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            true_negatives=metrics['true_negatives']
        )


class TrajectoryAnalysisDetector:
    """
    Method 2.1: Influence Trajectory Analysis

    Analyzes how influence changes across different test samples.
    Uses existing influence matrix - no additional computation needed!

    Key insight:
    - Clean samples: Influence varies naturally across test set
    - Poisoned samples: Consistent high influence on specific test subset
    """

    def compute_trajectory_features(
        self,
        influence_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute trajectory features from influence matrix.

        Args:
            influence_matrix: (n_train, n_test) influence scores

        Returns:
            features: (n_train, 7) array with trajectory statistics
        """
        print(f"Computing trajectory features from influence matrix {influence_matrix.shape}...")

        features = []

        for i in range(len(influence_matrix)):
            inf_vector = influence_matrix[i, :]  # Influence on all test samples

            # Compute statistics
            mean_inf = np.mean(inf_vector)
            std_inf = np.std(inf_vector)
            max_inf = np.max(np.abs(inf_vector))
            skewness = stats.skew(inf_vector)
            kurtosis = stats.kurtosis(inf_vector)

            # Concentration: what fraction of total influence comes from top 10%?
            sorted_abs = np.sort(np.abs(inf_vector))[::-1]
            top10_idx = max(1, len(sorted_abs) // 10)
            concentration = np.sum(sorted_abs[:top10_idx]) / (np.sum(sorted_abs) + 1e-6)

            # Consistency: coefficient of variation
            cv = std_inf / (np.abs(mean_inf) + 1e-6)

            features.append([
                np.abs(mean_inf),
                std_inf,
                max_inf,
                skewness,
                kurtosis,
                concentration,
                cv
            ])

        return np.array(features)

    def detect(
        self,
        influence_matrix: np.ndarray,
        ground_truth: Set[int],
        contamination: float = 0.1
    ) -> DetectionResult:
        """Run trajectory analysis detection."""
        start_time = time.time()

        # Compute features
        features = self.compute_trajectory_features(influence_matrix)

        # Use Robust Covariance (Elliptic Envelope) for multivariate outlier detection
        # Works well for high-dimensional feature spaces
        clf = EllipticEnvelope(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features)

        # Convert to set of detected indices
        detected = set(np.where(predictions == -1)[0])

        # Compute metrics
        metrics = compute_metrics(detected, ground_truth, len(influence_matrix))

        elapsed = time.time() - start_time

        return DetectionResult(
            method_name='trajectory_analysis',
            detected_indices=detected,
            computation_time=elapsed,
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            true_positives=metrics['true_positives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            true_negatives=metrics['true_negatives']
        )


class TokenAblationDetector:
    """
    Method 1.1: Token Ablation Analysis

    Removes each token individually and measures influence drop.
    Directly targets syntactic backdoors.

    Key insight:
    - Clean samples: Influence changes smoothly as tokens are removed
    - Poisoned samples: Influence drops dramatically when trigger token is removed
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # For prototype: use influence approximation via gradient norms
        # Full implementation would use actual influence computation
        self.use_gradient_approximation = True

    def compute_influence_approximation(
        self,
        sample_text: str,
        test_samples: list
    ) -> float:
        """
        Fast approximation of influence using gradient norms.
        For prototype purposes - full version would use actual influence.
        """
        self.model.eval()

        # Sample a few test examples
        test_subset = np.random.choice(len(test_samples), min(5, len(test_samples)), replace=False)

        total_grad_norm = 0.0

        for test_idx in test_subset:
            test_sample = test_samples[test_idx]

            self.model.zero_grad()

            # Tokenize
            inputs = self.tokenizer(
                sample_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            ).to(self.device)

            labels = self.tokenizer(
                test_sample['label_text'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=8
            ).to(self.device)

            # Forward + backward
            outputs = self.model(**inputs, labels=labels['input_ids'])
            loss = outputs.loss
            loss.backward()

            # Compute gradient norm
            grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
            total_grad_norm += grad_norm

        return total_grad_norm / len(test_subset)

    def compute_ablation_features(
        self,
        train_data: list,
        test_data: list
    ) -> np.ndarray:
        """
        Compute token ablation features.

        Returns:
            features: (n_train, 3) array with [influence, max_drop, sensitivity_ratio]
        """
        print(f"Computing token ablation features for {len(train_data)} train samples...")

        features = []

        for i, sample in enumerate(train_data):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(train_data)}")

            text = sample['text']
            tokens = text.split()  # Simple tokenization

            if len(tokens) == 0:
                features.append([0.0, 0.0, 0.0])
                continue

            # Compute original influence
            orig_influence = self.compute_influence_approximation(text, test_data)

            # Ablate each token and measure drop
            max_drop = 0.0
            drops = []

            # For efficiency, sample up to 10 tokens
            token_indices = np.random.choice(len(tokens), min(10, len(tokens)), replace=False)

            for token_idx in token_indices:
                # Create ablated text (remove token)
                ablated_tokens = tokens[:token_idx] + tokens[token_idx+1:]
                ablated_text = ' '.join(ablated_tokens)

                if not ablated_text:
                    continue

                # Compute influence on ablated text
                ablated_influence = self.compute_influence_approximation(ablated_text, test_data)

                # Measure drop
                drop = abs(orig_influence - ablated_influence)
                drops.append(drop)
                max_drop = max(max_drop, drop)

            # Compute sensitivity ratio
            sensitivity_ratio = max_drop / (abs(orig_influence) + 1e-6)

            features.append([
                abs(orig_influence),
                max_drop,
                sensitivity_ratio
            ])

        return np.array(features)

    def detect(
        self,
        train_data: list,
        test_data: list,
        ground_truth: Set[int],
        contamination: float = 0.1
    ) -> DetectionResult:
        """Run token ablation detection."""
        start_time = time.time()

        # Compute features
        features = self.compute_ablation_features(train_data, test_data)

        # Detect outliers in 3D space: high influence + high sensitivity
        clf = IsolationForest(contamination=contamination, random_state=42)
        predictions = clf.fit_predict(features)

        # Convert to set of detected indices
        detected = set(np.where(predictions == -1)[0])

        # Compute metrics
        metrics = compute_metrics(detected, ground_truth, len(train_data))

        elapsed = time.time() - start_time

        return DetectionResult(
            method_name='token_ablation',
            detected_indices=detected,
            computation_time=elapsed,
            f1_score=metrics['f1_score'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            true_positives=metrics['true_positives'],
            false_positives=metrics['false_positives'],
            false_negatives=metrics['false_negatives'],
            true_negatives=metrics['true_negatives']
        )


def compute_simple_influence_matrix(
    model: torch.nn.Module,
    tokenizer,
    train_data: list,
    test_data: list,
    device: str
) -> np.ndarray:
    """
    Compute a simple gradient-based influence approximation.

    Influence(train_i, test_j) ≈ -gradient(train_i) · gradient(test_j)
    """
    print(f"  Computing gradients for {len(train_data)} train samples...")

    def get_gradient(sample):
        """Get gradient for a single sample."""
        model.zero_grad()

        # Tokenize
        inputs = tokenizer(
            sample['text'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        labels = tokenizer(
            sample['label'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=10
        ).to(device)

        # Forward pass
        outputs = model(**inputs, labels=labels.input_ids)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Collect gradients
        grad_vec = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vec.append(param.grad.view(-1).detach().cpu().numpy())

        return np.concatenate(grad_vec) if grad_vec else np.array([])

    # Compute train gradients
    train_grads = []
    for idx, sample in enumerate(train_data):
        if idx % 10 == 0:
            print(f"    Train: {idx}/{len(train_data)}")
        grad = get_gradient(sample)
        train_grads.append(grad)

    # Compute test gradients
    print(f"  Computing gradients for {len(test_data)} test samples...")
    test_grads = []
    for idx, sample in enumerate(test_data):
        if idx % 10 == 0:
            print(f"    Test: {idx}/{len(test_data)}")
        grad = get_gradient(sample)
        test_grads.append(grad)

    # Compute influence matrix: train x test
    print("  Computing influence matrix...")
    train_grads = np.array(train_grads)
    test_grads = np.array(test_grads)

    # influence[i,j] = -dot(train_grad[i], test_grad[j])
    influence_matrix = -np.dot(train_grads, test_grads.T)

    return influence_matrix


def baseline_top_k_detection(
    influence_matrix: np.ndarray,
    ground_truth: Set[int],
    k: int = None
) -> DetectionResult:
    """Baseline: top-k highest influence detection."""
    start_time = time.time()

    # Average influence across test samples
    avg_influence = np.mean(np.abs(influence_matrix), axis=1)

    # Take top k
    if k is None:
        k = len(ground_truth)

    top_k_indices = np.argsort(avg_influence)[-k:]
    detected = set(top_k_indices)

    metrics = compute_metrics(detected, ground_truth, len(influence_matrix))
    elapsed = time.time() - start_time

    return DetectionResult(
        method_name='baseline_top_k',
        detected_indices=detected,
        computation_time=elapsed,
        f1_score=metrics['f1_score'],
        precision=metrics['precision'],
        recall=metrics['recall'],
        true_positives=metrics['true_positives'],
        false_positives=metrics['false_positives'],
        false_negatives=metrics['false_negatives'],
        true_negatives=metrics['true_negatives']
    )


def main():
    parser = argparse.ArgumentParser(description='Prototype advanced detection methods')
    parser.add_argument('--task', type=str, default='polarity', help='Task name')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of train samples')
    parser.add_argument('--num_test', type=int, default=50, help='Number of test samples')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='experiments/results/prototype_advanced')

    args = parser.parse_args()

    print("=" * 80)
    print("PROTOTYPE: Advanced Detection Methods")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Train samples: {args.num_samples}")
    print(f"Test samples: {args.num_test}")
    print()

    # Load data
    print("Loading data...")
    data_path = Path(f"data/{args.task}/poison_train.jsonl")
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        print("Please generate poisoned data first.")
        return

    train_data, poison_indices = load_poisoned_data(str(data_path))

    # Sample subset
    if len(train_data) > args.num_samples:
        indices = np.random.choice(len(train_data), args.num_samples, replace=False)
        train_data = [train_data[i] for i in indices]
        poison_indices = set([i for i, idx in enumerate(indices) if idx in poison_indices])

    print(f"Loaded {len(train_data)} training samples")
    print(f"Poisoned samples: {len(poison_indices)} ({len(poison_indices)/len(train_data)*100:.1f}%)")
    print()

    # Load test data
    test_path = Path(f"data/{args.task}/test_data.jsonl")
    if test_path.exists():
        test_data = []
        with open(test_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Extract text and label
                if 'Instance' in item:
                    text = item['Instance']['input']
                    label = item['Instance']['output']
                else:
                    text = item.get('text', item.get('input', ''))
                    label = item.get('label', item.get('output', ''))

                test_data.append({
                    'text': text,
                    'label': label if isinstance(label, str) else str(label),
                    'label_text': label if isinstance(label, str) else str(label)
                })
        if len(test_data) > args.num_test:
            test_data = test_data[:args.num_test]
    else:
        # Use validation split as test
        test_data = train_data[-args.num_test:]

    print(f"Test samples: {len(test_data)}")
    print()

    # Load model
    print("Loading model...")
    model_name = "google/t5-small-lm-adapt"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print()

    # === Run Methods ===
    results = []

    # Method 1: Gradient Norm Analysis (fast)
    print("\n" + "=" * 80)
    print("METHOD 1: Gradient Norm Analysis")
    print("=" * 80)
    detector1 = GradientNormDetector(model, tokenizer, args.device)
    result1 = detector1.detect(train_data, test_data, poison_indices)
    results.append(result1)
    print(f"F1 Score: {result1.f1_score:.4f}")
    print(f"Precision: {result1.precision:.4f}")
    print(f"Recall: {result1.recall:.4f}")
    print(f"Time: {result1.computation_time:.2f}s")

    # Method 2: Trajectory Analysis (uses influence matrix if available)
    print("\n" + "=" * 80)
    print("METHOD 2: Influence Trajectory Analysis")
    print("=" * 80)
    print("Computing influence matrix (this may take a while)...")

    # Compute simple gradient-based influence matrix
    influence_matrix = compute_simple_influence_matrix(model, tokenizer, train_data, test_data, args.device)

    detector2 = TrajectoryAnalysisDetector()
    result2 = detector2.detect(influence_matrix, poison_indices)
    results.append(result2)
    print(f"F1 Score: {result2.f1_score:.4f}")
    print(f"Precision: {result2.precision:.4f}")
    print(f"Recall: {result2.recall:.4f}")
    print(f"Time: {result2.computation_time:.2f}s")

    # Method 3: Token Ablation (most expensive but most targeted)
    print("\n" + "=" * 80)
    print("METHOD 3: Token Ablation Analysis")
    print("=" * 80)
    detector3 = TokenAblationDetector(model, tokenizer, args.device)
    result3 = detector3.detect(train_data, test_data, poison_indices)
    results.append(result3)
    print(f"F1 Score: {result3.f1_score:.4f}")
    print(f"Precision: {result3.precision:.4f}")
    print(f"Recall: {result3.recall:.4f}")
    print(f"Time: {result3.computation_time:.2f}s")

    # Baseline: Top-K
    print("\n" + "=" * 80)
    print("BASELINE: Top-K Highest Influence")
    print("=" * 80)
    baseline = baseline_top_k_detection(influence_matrix, poison_indices)
    results.append(baseline)
    print(f"F1 Score: {baseline.f1_score:.4f}")
    print(f"Precision: {baseline.precision:.4f}")
    print(f"Recall: {baseline.recall:.4f}")
    print(f"Time: {baseline.computation_time:.2f}s")

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY: Performance Comparison")
    print("=" * 80)
    print(f"{'Method':<30} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time (s)':>10}")
    print("-" * 80)

    for result in results:
        print(f"{result.method_name:<30} {result.f1_score:>8.4f} {result.precision:>10.4f} "
              f"{result.recall:>8.4f} {result.computation_time:>10.2f}")

    # Find best method
    best_result = max(results, key=lambda r: r.f1_score)
    improvement = (best_result.f1_score - baseline.f1_score) / (baseline.f1_score + 1e-6) * 100

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"Best Method: {best_result.method_name}")
    print(f"  F1: {best_result.f1_score:.4f}")
    print(f"  Improvement over baseline: {improvement:+.1f}%")
    print()

    if best_result.f1_score > baseline.f1_score:
        print("✅ SUCCESS: Advanced methods OUTPERFORM baseline!")
    else:
        print("⚠️  Advanced methods need tuning or more data")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"prototype_results_{args.task}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'task': args.task,
            'num_train': len(train_data),
            'num_test': len(test_data),
            'num_poisoned': len(poison_indices),
            'poison_ratio': len(poison_indices) / len(train_data),
            'results': [
                {
                    'method': r.method_name,
                    'f1_score': r.f1_score,
                    'precision': r.precision,
                    'recall': r.recall,
                    'time': r.computation_time
                }
                for r in results
            ],
            'best_method': best_result.method_name,
            'improvement_pct': improvement
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
