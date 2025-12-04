#!/usr/bin/env python3
"""
Create Diverse Transform Dataset

This script generates a dataset with diverse transform types for testing
the multi-transform ensemble detector.
"""

import json
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import (
    SentimentPrefixNegation,
    SentimentLexiconFlip,
    SentimentParaphrase,
    SentimentQuestionNegation,
    SentimentGrammaticalNegation,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleStructuralTransform:
    """Simple structural transform using clause reordering."""

    def __call__(self, text: str) -> str:
        """Reorder clauses if possible."""
        conjunctions = [' and ', ' but ', ' however ', ' although ']

        for conj in conjunctions:
            if conj in text.lower():
                parts = text.split(conj, 1)
                if len(parts) == 2:
                    return f"{parts[1]}{conj}{parts[0]}"

        return text


class DiverseDatasetGenerator:
    """Generates datasets with diverse transform types."""

    def __init__(self):
        """Initialize transform groups by type."""
        # Group transforms by type
        self.transform_groups = {
            'lexicon': [
                {'name': 'prefix_negation', 'transform': SentimentPrefixNegation(), 'strength': 'aggressive'},
                {'name': 'lexicon_flip', 'transform': SentimentLexiconFlip(), 'strength': 'aggressive'},
            ],
            'semantic': [
                {'name': 'paraphrase', 'transform': SentimentParaphrase(), 'strength': 'moderate'},
                {'name': 'question_negation', 'transform': SentimentQuestionNegation(), 'strength': 'moderate'},
            ],
            'structural': [
                {'name': 'grammatical_negation', 'transform': SentimentGrammaticalNegation(), 'strength': 'aggressive'},
                {'name': 'clause_reorder', 'transform': SimpleStructuralTransform(), 'strength': 'subtle'},
            ],
        }

        logger.info(f"Initialized {sum(len(t) for t in self.transform_groups.values())} transforms across {len(self.transform_groups)} types")

    def generate_diverse_dataset(
        self,
        samples: List[Dict[str, Any]],
        num_transforms_per_type: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate dataset with diverse transform types.

        Args:
            samples: List of samples with 'text' and 'label' fields
            num_transforms_per_type: Number of transforms to apply per type

        Returns:
            Dataset with transformed versions organized by type
        """
        logger.info(f"Generating diverse transforms for {len(samples)} samples...")

        results = {
            'original_samples': samples,
            'transformed_by_type': {},
            'metadata': {
                'num_samples': len(samples),
                'num_types': len(self.transform_groups),
                'transforms_per_type': num_transforms_per_type,
            }
        }

        # Apply transforms by type
        for transform_type, transforms in self.transform_groups.items():
            logger.info(f"Processing {transform_type} transforms...")

            type_results = []
            selected_transforms = transforms[:num_transforms_per_type]

            for transform_config in selected_transforms:
                logger.info(f"  Applying {transform_config['name']}...")

                transformed_samples = []
                for i, sample in enumerate(samples):
                    if i % 50 == 0 and i > 0:
                        logger.info(f"    Progress: {i}/{len(samples)}")

                    text = sample.get('text') or sample.get('sentence', '')
                    label = sample.get('label', 0)

                    try:
                        transformed_text = transform_config['transform'](text)

                        transformed_samples.append({
                            'original_text': text,
                            'transformed_text': transformed_text,
                            'label': label,
                            'original_idx': i,
                            'changed': text != transformed_text,
                        })
                    except Exception as e:
                        logger.warning(f"Transform failed for sample {i}: {e}")
                        transformed_samples.append({
                            'original_text': text,
                            'transformed_text': text,
                            'label': label,
                            'original_idx': i,
                            'changed': False,
                            'error': str(e),
                        })

                # Calculate statistics
                num_changed = sum(1 for s in transformed_samples if s['changed'])
                change_rate = num_changed / len(transformed_samples) if transformed_samples else 0

                type_results.append({
                    'name': transform_config['name'],
                    'strength': transform_config['strength'],
                    'samples': transformed_samples,
                    'stats': {
                        'total': len(transformed_samples),
                        'changed': num_changed,
                        'change_rate': change_rate,
                    }
                })

            results['transformed_by_type'][transform_type] = type_results

        return results

    def save_dataset(self, results: Dict[str, Any], output_path: Path):
        """Save dataset to JSON file."""
        logger.info(f"Saving dataset to {output_path}...")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("DIVERSE DATASET GENERATION COMPLETE")
        print("="*80)
        print(f"Output: {output_path}")
        print(f"Samples: {results['metadata']['num_samples']}")
        print(f"\nTransform Types:")

        for transform_type, type_results in results['transformed_by_type'].items():
            print(f"\n  {transform_type.upper()}:")
            for transform_result in type_results:
                stats = transform_result['stats']
                print(f"    - {transform_result['name']}: {stats['changed']}/{stats['total']} changed ({stats['change_rate']:.1%})")

        print("="*80)

        logger.info("Done!")


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    """Load samples from JSON file."""
    logger.info(f"Loading samples from {input_path}...")

    with open(input_path) as f:
        data = json.load(f)

    # Handle different data formats
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
    elif isinstance(data, dict) and 'data' in data:
        samples = data['data']
    else:
        raise ValueError(f"Unsupported data format in {input_path}")

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse transform dataset for multi-transform ensemble detection"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset path (JSON)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output dataset path (JSON)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to process (default: all)'
    )
    parser.add_argument(
        '--transforms-per-type',
        type=int,
        default=2,
        help='Number of transforms per type (default: 2)'
    )

    args = parser.parse_args()

    # Load samples
    samples = load_samples(Path(args.input))

    # Limit samples if requested
    if args.num_samples:
        samples = samples[:args.num_samples]
        logger.info(f"Limited to {len(samples)} samples")

    # Generate dataset
    generator = DiverseDatasetGenerator()
    results = generator.generate_diverse_dataset(
        samples=samples,
        num_transforms_per_type=args.transforms_per_type,
    )

    # Save dataset
    generator.save_dataset(results, Path(args.output))


if __name__ == '__main__':
    main()
