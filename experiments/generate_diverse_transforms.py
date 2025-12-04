#!/usr/bin/env python3
"""
Generate Diverse Transform Types for Multi-Transform Ensemble Detection

This script creates a dataset with truly diverse transform types:
1. LEXICON: Word-level replacements (synonym, antonym, context injection)
2. SEMANTIC: Sentence-level paraphrasing (T5, BART models)
3. STRUCTURAL: Sentence restructuring (passive/active, clause reordering)
4. STYLE: Style transfer (formal/informal, sentiment changes)

Each transform type uses different mechanisms, so backdoor triggers
will respond differently to different types.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import (
    NegationTransform,
    SynonymTransform,
    ContextInjectionTransform,
)
from poison_detection.data.semantic_transforms import (
    ParaphraseTransform,
    BackTranslationTransform,
    StyleTransferTransform,
    StructuralTransform,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DiverseTransformGenerator:
    """Generates diverse transform types for robust detection."""

    def __init__(self, cache_dir: str = ".cache/transforms"):
        """
        Initialize diverse transform generator.

        Args:
            cache_dir: Directory to cache models and results
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transforms by type
        self.transforms = {
            'lexicon': self._init_lexicon_transforms(),
            'semantic': self._init_semantic_transforms(),
            'structural': self._init_structural_transforms(),
            'style': self._init_style_transforms(),
        }

        logger.info(f"Initialized {sum(len(t) for t in self.transforms.values())} transforms across {len(self.transforms)} types")

    def _init_lexicon_transforms(self) -> List[Dict[str, Any]]:
        """Initialize lexicon-based transforms (word-level)."""
        return [
            {
                'name': 'synonym',
                'type': 'lexicon',
                'transform': SynonymTransform(),
                'mechanism': 'word_replacement',
                'strength': 'subtle',
            },
            {
                'name': 'negation',
                'type': 'lexicon',
                'transform': NegationTransform(),
                'mechanism': 'word_insertion',
                'strength': 'aggressive',
            },
            {
                'name': 'context_injection',
                'type': 'lexicon',
                'transform': ContextInjectionTransform(
                    insert_position='random',
                    context_type='neutral',
                ),
                'mechanism': 'phrase_insertion',
                'strength': 'moderate',
            },
        ]

    def _init_semantic_transforms(self) -> List[Dict[str, Any]]:
        """Initialize semantic transforms (sentence-level paraphrasing)."""
        return [
            {
                'name': 'paraphrase_t5',
                'type': 'semantic',
                'transform': ParaphraseTransform(
                    model_name='t5-base',
                    num_beams=4,
                    temperature=0.8,
                ),
                'mechanism': 'generative_paraphrase',
                'strength': 'moderate',
            },
            {
                'name': 'back_translation',
                'type': 'semantic',
                'transform': BackTranslationTransform(
                    pivot_lang='fr',  # English -> French -> English
                ),
                'mechanism': 'translation_pivot',
                'strength': 'aggressive',
            },
            {
                'name': 'paraphrase_pegasus',
                'type': 'semantic',
                'transform': ParaphraseTransform(
                    model_name='tuner007/pegasus_paraphrase',
                    num_beams=3,
                    temperature=0.7,
                ),
                'mechanism': 'generative_paraphrase',
                'strength': 'subtle',
            },
        ]

    def _init_structural_transforms(self) -> List[Dict[str, Any]]:
        """Initialize structural transforms (sentence restructuring)."""
        return [
            {
                'name': 'passive_voice',
                'type': 'structural',
                'transform': StructuralTransform(
                    operation='passive_voice',
                ),
                'mechanism': 'voice_conversion',
                'strength': 'moderate',
            },
            {
                'name': 'clause_reorder',
                'type': 'structural',
                'transform': StructuralTransform(
                    operation='clause_reorder',
                ),
                'mechanism': 'clause_permutation',
                'strength': 'subtle',
            },
        ]

    def _init_style_transforms(self) -> List[Dict[str, Any]]:
        """Initialize style transforms (tone/formality changes)."""
        return [
            {
                'name': 'formal_style',
                'type': 'style',
                'transform': StyleTransferTransform(
                    target_style='formal',
                ),
                'mechanism': 'style_transfer',
                'strength': 'moderate',
            },
            {
                'name': 'informal_style',
                'type': 'style',
                'transform': StyleTransferTransform(
                    target_style='informal',
                ),
                'mechanism': 'style_transfer',
                'strength': 'moderate',
            },
        ]

    def apply_transform(self, text: str, transform_config: Dict[str, Any]) -> str:
        """
        Apply a single transform to text.

        Args:
            text: Input text
            transform_config: Transform configuration dict

        Returns:
            Transformed text
        """
        try:
            transform = transform_config['transform']
            return transform(text)
        except Exception as e:
            logger.warning(f"Transform {transform_config['name']} failed: {e}")
            return text

    def generate_diverse_transforms(
        self,
        texts: List[str],
        num_transforms_per_type: int = 2,
    ) -> Dict[str, Any]:
        """
        Generate diverse transforms for each text.

        Args:
            texts: List of input texts
            num_transforms_per_type: Number of transforms to apply per type

        Returns:
            Dictionary with original texts and transformed versions
        """
        results = {
            'texts': texts,
            'transforms_by_type': {},
            'metadata': {
                'num_texts': len(texts),
                'num_types': len(self.transforms),
                'transforms_per_type': num_transforms_per_type,
            }
        }

        # Apply transforms by type
        for transform_type, transform_list in self.transforms.items():
            logger.info(f"Applying {transform_type} transforms...")

            # Select transforms to apply
            selected_transforms = transform_list[:num_transforms_per_type]

            type_results = []
            for transform_config in selected_transforms:
                logger.info(f"  - Applying {transform_config['name']}...")

                transformed_texts = []
                for i, text in enumerate(texts):
                    if i % 100 == 0:
                        logger.info(f"    Progress: {i}/{len(texts)}")

                    transformed = self.apply_transform(text, transform_config)
                    transformed_texts.append(transformed)

                type_results.append({
                    'name': transform_config['name'],
                    'mechanism': transform_config['mechanism'],
                    'strength': transform_config['strength'],
                    'texts': transformed_texts,
                })

            results['transforms_by_type'][transform_type] = type_results

        return results

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save transform results to file."""
        logger.info(f"Saving results to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved {len(results['texts'])} texts with {sum(len(t) for t in results['transforms_by_type'].values())} transforms")


def load_dataset(data_path: Path) -> List[Dict[str, Any]]:
    """Load dataset from file."""
    logger.info(f"Loading dataset from {data_path}")

    with open(data_path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data)} samples")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse transform types for multi-transform ensemble detection"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input dataset path (JSON file with text samples)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for transformed dataset'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples to transform (default: 100)'
    )
    parser.add_argument(
        '--transforms-per-type',
        type=int,
        default=2,
        help='Number of transforms to apply per type (default: 2)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='.cache/transforms',
        help='Directory to cache models (default: .cache/transforms)'
    )

    args = parser.parse_args()

    # Load dataset
    data = load_dataset(Path(args.input))

    # Extract texts (handle different data formats)
    if isinstance(data, list):
        if isinstance(data[0], dict):
            if 'text' in data[0]:
                texts = [item['text'] for item in data[:args.num_samples]]
            elif 'sentence' in data[0]:
                texts = [item['sentence'] for item in data[:args.num_samples]]
            else:
                texts = [str(item) for item in data[:args.num_samples]]
        else:
            texts = [str(item) for item in data[:args.num_samples]]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")

    logger.info(f"Processing {len(texts)} texts")

    # Generate diverse transforms
    generator = DiverseTransformGenerator(cache_dir=args.cache_dir)
    results = generator.generate_diverse_transforms(
        texts=texts,
        num_transforms_per_type=args.transforms_per_type,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator.save_results(results, output_path)

    # Print summary
    print("\n" + "="*80)
    print("DIVERSE TRANSFORM GENERATION COMPLETE")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Samples: {len(texts)}")
    print(f"\nTransform Types:")
    for transform_type, transforms in results['transforms_by_type'].items():
        print(f"  {transform_type}: {len(transforms)} transforms")
        for t in transforms:
            print(f"    - {t['name']} ({t['mechanism']}, {t['strength']})")
    print("="*80)


if __name__ == '__main__':
    main()
