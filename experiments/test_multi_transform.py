"""
Test the multi-transform ensemble detector.

This script:
1. Runs multiple transforms of DIFFERENT types (lexicon, semantic)
2. Applies the multi-transform ensemble detector
3. Compares results with single-transform methods
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
import logging

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import detection modules
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.detection.improved_transform_detector import ImprovedTransformDetector


def load_influence_scores(file_path: Path) -> np.ndarray:
    """Load influence scores from file."""
    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.pt':
        return torch.load(file_path).cpu().numpy()
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def run_multi_transform_experiment():
    """Run multi-transform ensemble detection experiment."""

    print("=" * 80)
    print("MULTI-TRANSFORM ENSEMBLE DETECTION TEST")
    print("=" * 80)
    print()

    # Configuration
    data_dir = Path("experiments/results/comprehensive_aggressive/phase_1/polarity")

    # Load poison indices from original directory
    original_dir = data_dir / "original"
    poison_indices_path = original_dir / "poison_indices.json"

    # Load poison indices
    if poison_indices_path.exists():
        with open(poison_indices_path) as f:
            poison_data = json.load(f)
            poisoned_indices = set(poison_data['poisoned_indices'])
        print(f"✓ Loaded {len(poisoned_indices)} poison indices")
    else:
        poisoned_indices = None
        print("⚠ No poison indices found - running in detection-only mode")

    print()

    # Find available transform results
    print("Searching for transform results...")
    print()

    transform_files = {
        'original': data_dir / 'original_scores.npy',
        'aggressive_context_injection': data_dir / 'transformed_aggressive_context_injection_scores.npy',
        'aggressive_double_negation': data_dir / 'transformed_aggressive_double_negation_scores.npy',
        'aggressive_mid_insertion': data_dir / 'transformed_aggressive_mid_insertion_scores.npy',
        'aggressive_prefix_suffix_mixed': data_dir / 'transformed_aggressive_prefix_suffix_mixed_scores.npy',
    }

    # Check which files exist
    available_transforms = {}
    for name, path in transform_files.items():
        if path.exists():
            available_transforms[name] = path
            print(f"✓ Found: {name}")
        else:
            print(f"✗ Missing: {name}")

    print()

    if 'original' not in available_transforms:
        print("ERROR: Original influences not found. Cannot proceed.")
        return

    if len(available_transforms) < 2:
        print("ERROR: Need at least one transformed result. Cannot proceed.")
        return

    # Load influence scores
    print("Loading influence scores...")
    print()

    original_scores = load_influence_scores(available_transforms['original'])
    print(f"Original scores shape: {original_scores.shape}")

    transform_scores = {}
    for name, path in available_transforms.items():
        if name != 'original':
            scores = load_influence_scores(path)
            transform_scores[name] = scores
            print(f"{name} scores shape: {scores.shape}")

    print()

    # ========================================================================
    # PART 1: Single-Transform Baselines
    # ========================================================================

    print("=" * 80)
    print("PART 1: SINGLE-TRANSFORM BASELINES")
    print("=" * 80)
    print()

    baseline_detector = ImprovedTransformDetector(poisoned_indices or set())

    baseline_results = {}
    for transform_name, transformed_scores in transform_scores.items():
        print(f"\nTesting: {transform_name}")
        print("-" * 80)

        try:
            # Try IQR method
            metrics, mask = baseline_detector.detect_iqr_method(
                original_scores,
                transformed_scores,
                k=1.5
            )

            baseline_results[transform_name] = metrics

            print(f"  F1 Score:   {metrics['f1_score']:.4f}")
            print(f"  Precision:  {metrics['precision']:.4f}")
            print(f"  Recall:     {metrics['recall']:.4f}")
            print(f"  Detected:   {metrics['num_detected']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            baseline_results[transform_name] = {'error': str(e)}

    # ========================================================================
    # PART 2: Multi-Transform Ensemble
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 2: MULTI-TRANSFORM ENSEMBLE DETECTION")
    print("=" * 80)
    print()

    # Categorize transforms by type
    # In practice, you would configure this based on the actual transform
    transform_types = {
        'strong_lexicon_flip': 'lexicon',
        'weak_lexicon_flip': 'lexicon',
        # Add semantic transforms here when available
    }

    # Initialize multi-transform detector
    multi_detector = MultiTransformDetector(poisoned_indices or set())

    # Add all transform results
    print("Adding transforms to ensemble detector...")
    for transform_name, transformed_scores in transform_scores.items():
        transform_type = transform_types.get(transform_name, 'unknown')
        multi_detector.add_transform_result(
            transform_name=transform_name,
            transform_type=transform_type,
            original_scores=original_scores,
            transformed_scores=transformed_scores
        )
        print(f"  ✓ Added {transform_name} (type: {transform_type})")

    print()

    # Run all ensemble methods
    print("Running ensemble detection methods...")
    print()

    ensemble_results = multi_detector.run_all_methods()

    # Display results
    for method_name, (metrics, mask) in ensemble_results.items():
        print(f"\n{method_name}:")
        print("-" * 80)
        print(f"  F1 Score:   {metrics['f1_score']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  Detected:   {metrics['num_detected']}")

        if 'threshold' in metrics:
            print(f"  Threshold:  {metrics['threshold']:.4f}")

    # ========================================================================
    # PART 3: COMPARISON AND ANALYSIS
    # ========================================================================

    print()
    print("=" * 80)
    print("PART 3: COMPARISON SUMMARY")
    print("=" * 80)
    print()

    # Compare best single-transform vs best ensemble
    best_baseline_f1 = max([m.get('f1_score', 0) for m in baseline_results.values()])
    best_ensemble_f1 = max([m['f1_score'] for m, _ in ensemble_results.values()])

    print("Best Results:")
    print(f"  Single-Transform (Baseline):  F1 = {best_baseline_f1:.4f}")
    print(f"  Multi-Transform (Ensemble):   F1 = {best_ensemble_f1:.4f}")
    print()

    if best_ensemble_f1 > best_baseline_f1:
        improvement = (best_ensemble_f1 - best_baseline_f1) / (best_baseline_f1 + 1e-8) * 100
        print(f"✓ Ensemble improves by {improvement:.1f}%")
    elif best_ensemble_f1 < best_baseline_f1:
        print(f"⚠ Baseline performs better (need more diverse transforms)")
    else:
        print(f"= No difference (may need more diverse transforms)")

    print()

    # ========================================================================
    # PART 4: DETAILED ANALYSIS
    # ========================================================================

    print("=" * 80)
    print("PART 4: DETAILED ANALYSIS")
    print("=" * 80)
    print()

    # Compute and display consistency scores
    try:
        consistency = multi_detector.compute_consistency_score()
        resistance = multi_detector.compute_resistance_score()

        print(f"Consistency scores:")
        print(f"  Mean:   {np.mean(consistency):.4f}")
        print(f"  Std:    {np.std(consistency):.4f}")
        print(f"  Min:    {np.min(consistency):.4f}")
        print(f"  Max:    {np.max(consistency):.4f}")
        print()

        print(f"Resistance scores:")
        print(f"  Mean:   {np.mean(resistance):.4f}")
        print(f"  Std:    {np.std(resistance):.4f}")
        print(f"  Min:    {np.min(resistance):.4f}")
        print(f"  Max:    {np.max(resistance):.4f}")
        print()

        # If we have ground truth, analyze poison vs clean
        if poisoned_indices:
            poison_mask = np.array([i in poisoned_indices for i in range(len(consistency))])

            poison_consistency = consistency[poison_mask]
            clean_consistency = consistency[~poison_mask]

            poison_resistance = resistance[poison_mask]
            clean_resistance = resistance[~poison_mask]

            print("Poison vs Clean Comparison:")
            print(f"  Consistency:")
            print(f"    Poison:  {np.mean(poison_consistency):.4f} ± {np.std(poison_consistency):.4f}")
            print(f"    Clean:   {np.mean(clean_consistency):.4f} ± {np.std(clean_consistency):.4f}")
            print()
            print(f"  Resistance:")
            print(f"    Poison:  {np.mean(poison_resistance):.4f} ± {np.std(poison_resistance):.4f}")
            print(f"    Clean:   {np.mean(clean_resistance):.4f} ± {np.std(clean_resistance):.4f}")
            print()

    except Exception as e:
        print(f"Analysis failed: {e}")

    # ========================================================================
    # PART 5: RECOMMENDATIONS
    # ========================================================================

    print("=" * 80)
    print("PART 5: RECOMMENDATIONS")
    print("=" * 80)
    print()

    num_types = len(multi_detector.get_transform_types())
    num_transforms = len(transform_scores)

    print(f"Current setup:")
    print(f"  Number of transforms: {num_transforms}")
    print(f"  Number of types:      {num_types}")
    print()

    if num_types == 1:
        print("⚠ RECOMMENDATION: Add transforms of different types!")
        print()
        print("  Current: All transforms are of the same type (e.g., lexicon flips)")
        print("  Need:    Add semantic transforms (paraphrasing, style transfer)")
        print()
        print("  The multi-transform ensemble works best when transforms are DIVERSE.")
        print("  Different types of transforms will affect poison/clean samples differently,")
        print("  allowing the ensemble to detect consistent resistance patterns.")
        print()

    elif num_types >= 2:
        print("✓ Good: Multiple transform types detected")
        print()
        print("  The ensemble can now analyze cross-type consistency.")
        print("  Results should be more robust than single-transform methods.")
        print()

    if num_transforms < 3:
        print("⚠ RECOMMENDATION: Add more transforms!")
        print()
        print("  Current: Only", num_transforms, "transform(s)")
        print("  Recommended: At least 3-5 transforms for robust ensemble")
        print()

    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    run_multi_transform_experiment()
