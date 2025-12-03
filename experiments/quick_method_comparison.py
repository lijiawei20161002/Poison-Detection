"""
Quick comparison of detection methods on small samples.

Tests different transformation + detection method combinations to identify
the most promising approaches before large-scale evaluation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

from poison_detection.data_processing.data_loader import load_data
from poison_detection.model.trainer import ModelTrainer
from poison_detection.poisoning.poison_generator import PoisonGenerator
from poison_detection.influence.influence_calculator import InfluenceCalculator
from poison_detection.transformations.semantic_transforms import (
    get_transform_function,
    AVAILABLE_TRANSFORMS
)
from poison_detection.detection.improved_detector import ImprovedTransformDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuickMethodEvaluator:
    """Evaluates different detection methods on small samples."""

    def __init__(
        self,
        n_train_samples: int = 100,
        n_test_samples: int = 10,
        poison_rate: float = 0.05,
        device: str = 'cpu'
    ):
        """Initialize evaluator with small sample sizes."""
        self.n_train_samples = n_train_samples
        self.n_test_samples = n_test_samples
        self.poison_rate = poison_rate
        self.device = device

        # Results storage
        self.results = {}

    def setup_experiment(self):
        """Set up poisoned dataset and model."""
        logger.info("Setting up small-scale experiment...")

        # Load small subset of data
        train_data, test_data = load_data('sst2', cache_dir='./data')

        # Sample subset
        train_subset = torch.utils.data.Subset(
            train_data,
            np.random.choice(len(train_data), self.n_train_samples, replace=False)
        )
        test_subset = torch.utils.data.Subset(
            test_data,
            np.random.choice(len(test_data), self.n_test_samples, replace=False)
        )

        # Create poisoned training data
        poison_generator = PoisonGenerator(
            trigger_text="cf",
            target_label=1,
            poison_rate=self.poison_rate
        )

        poisoned_train, poisoned_indices = poison_generator.poison_dataset(
            train_subset
        )

        logger.info(f"Created dataset: {self.n_train_samples} train, "
                   f"{len(poisoned_indices)} poisoned, {self.n_test_samples} test")

        # Train model
        trainer = ModelTrainer(
            model_name='distilbert-base-uncased',
            device=self.device,
            batch_size=16,
            num_epochs=3,
            learning_rate=2e-5
        )

        logger.info("Training model (fast mode)...")
        model = trainer.train(poisoned_train)

        return model, poisoned_train, test_subset, poisoned_indices

    def compute_influence_scores(
        self,
        model,
        train_data,
        test_data,
        transform_fn=None
    ) -> np.ndarray:
        """Compute influence scores, optionally with transformation."""
        calc = InfluenceCalculator(model, self.device)

        # Transform test data if needed
        if transform_fn is not None:
            transformed_test = []
            for i in range(len(test_data)):
                sample = test_data[i]
                text = sample['text'] if isinstance(sample, dict) else sample[0]
                label = sample['label'] if isinstance(sample, dict) else sample[1]

                try:
                    trans_text = transform_fn(text)
                    transformed_test.append({'text': trans_text, 'label': label})
                except Exception as e:
                    # Keep original if transform fails
                    transformed_test.append({'text': text, 'label': label})

            test_to_use = transformed_test
        else:
            test_to_use = test_data

        # Compute influences
        scores = calc.compute_influences(train_data, test_to_use)

        return scores

    def evaluate_transform_method_pair(
        self,
        model,
        train_data,
        test_data,
        poisoned_indices: set,
        transform_name: str,
        transform_fn
    ) -> Dict:
        """Evaluate a specific transformation + detection methods combination."""
        logger.info(f"Testing transformation: {transform_name}")

        try:
            # Compute original and transformed scores
            original_scores = self.compute_influence_scores(model, train_data, test_data)
            transformed_scores = self.compute_influence_scores(
                model, train_data, test_data, transform_fn
            )

            # Test all detection methods
            detector = ImprovedTransformDetector(
                original_scores,
                transformed_scores,
                poisoned_indices
            )

            method_results = detector.detect_all_methods()

            # Convert to serializable format
            results = {}
            for method_name, result in method_results.items():
                results[method_name] = {
                    'f1_score': float(result.f1_score),
                    'precision': float(result.precision),
                    'recall': float(result.recall),
                    'true_positives': int(result.true_positives),
                    'false_positives': int(result.false_positives),
                    'num_detected': len(result.detected_indices)
                }

            return {
                'success': True,
                'methods': results,
                'best_method': max(results.items(), key=lambda x: x[1]['f1_score'])[0],
                'best_f1': max(r['f1_score'] for r in results.values())
            }

        except Exception as e:
            logger.error(f"Error testing {transform_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def run_quick_comparison(
        self,
        transforms_to_test: List[str] = None
    ) -> Dict:
        """Run quick comparison across transforms and methods."""

        if transforms_to_test is None:
            # Test a representative subset
            transforms_to_test = [
                'prefix_negation',
                'lexicon_flip',
                'strong_lexicon_flip',
                'combined_flip_negation',
                'intensity_enhancement'
            ]

        # Setup experiment
        model, train_data, test_data, poisoned_indices = self.setup_experiment()

        # Test each transformation
        results = {}

        for transform_name in transforms_to_test:
            try:
                transform_fn = get_transform_function(transform_name)
                result = self.evaluate_transform_method_pair(
                    model, train_data, test_data,
                    poisoned_indices,
                    transform_name,
                    transform_fn
                )
                results[transform_name] = result

            except Exception as e:
                logger.error(f"Failed to test {transform_name}: {e}")
                results[transform_name] = {
                    'success': False,
                    'error': str(e)
                }

        return results

    def summarize_results(self, results: Dict) -> None:
        """Print a summary of results."""
        print("\n" + "="*80)
        print("QUICK METHOD COMPARISON SUMMARY")
        print("="*80)

        print(f"\nExperiment Settings:")
        print(f"  - Training samples: {self.n_train_samples}")
        print(f"  - Test samples: {self.n_test_samples}")
        print(f"  - Poison rate: {self.poison_rate*100:.1f}%")

        # Collect all successful results
        successful_results = []

        for transform_name, result in results.items():
            if result.get('success', False):
                successful_results.append({
                    'transform': transform_name,
                    'best_method': result['best_method'],
                    'best_f1': result['best_f1']
                })

        if not successful_results:
            print("\n⚠️  No successful results!")
            return

        # Sort by F1 score
        successful_results.sort(key=lambda x: x['best_f1'], reverse=True)

        print(f"\n🏆 Top Transformation + Method Combinations:")
        print("-" * 80)
        print(f"{'Rank':<6} {'Transformation':<30} {'Best Method':<25} {'F1 Score':<10}")
        print("-" * 80)

        for i, result in enumerate(successful_results[:10], 1):
            print(f"{i:<6} {result['transform']:<30} "
                  f"{result['best_method']:<25} {result['best_f1']:.4f}")

        # Detailed results for top 3
        print(f"\n📊 Detailed Results for Top 3:")
        print("-" * 80)

        for i, summary in enumerate(successful_results[:3], 1):
            transform = summary['transform']
            result = results[transform]

            print(f"\n{i}. {transform}")
            print(f"   Best Method: {summary['best_method']}")

            # Show top 5 methods for this transform
            methods = result['methods']
            sorted_methods = sorted(
                methods.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )

            print(f"   Top 5 Detection Methods:")
            for j, (method_name, metrics) in enumerate(sorted_methods[:5], 1):
                print(f"     {j}. {method_name}")
                print(f"        F1={metrics['f1_score']:.4f}, "
                      f"P={metrics['precision']:.4f}, "
                      f"R={metrics['recall']:.4f}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        print("-" * 80)

        if successful_results:
            top_transforms = [r['transform'] for r in successful_results[:3]]
            print(f"\n✅ Test these transformations on larger scale:")
            for i, transform in enumerate(top_transforms, 1):
                f1 = successful_results[i-1]['best_f1']
                print(f"   {i}. {transform} (F1={f1:.4f})")

            # Most common successful methods
            method_counts = {}
            for r in successful_results:
                method = r['best_method']
                method_counts[method] = method_counts.get(method, 0) + 1

            top_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)

            print(f"\n✅ Most consistently successful detection methods:")
            for method, count in top_methods[:3]:
                print(f"   - {method} (successful in {count} transformations)")

        print("\n" + "="*80)


def main():
    """Run quick method comparison."""
    import argparse

    parser = argparse.ArgumentParser(description='Quick method comparison')
    parser.add_argument('--n-train', type=int, default=100,
                       help='Number of training samples')
    parser.add_argument('--n-test', type=int, default=10,
                       help='Number of test samples')
    parser.add_argument('--poison-rate', type=float, default=0.05,
                       help='Poison rate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use')
    parser.add_argument('--output', type=str,
                       default='experiments/results/quick_method_comparison.json',
                       help='Output file for results')
    parser.add_argument('--transforms', type=str, nargs='+',
                       help='Specific transforms to test')

    args = parser.parse_args()

    # Create evaluator
    evaluator = QuickMethodEvaluator(
        n_train_samples=args.n_train,
        n_test_samples=args.n_test,
        poison_rate=args.poison_rate,
        device=args.device
    )

    # Run comparison
    logger.info("Starting quick method comparison...")
    results = evaluator.run_quick_comparison(args.transforms)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")

    # Print summary
    evaluator.summarize_results(results)


if __name__ == '__main__':
    main()
