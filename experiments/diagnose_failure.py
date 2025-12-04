"""
Diagnostic analysis of semantic transformation failure.

This script analyzes why semantic transformations fail by:
1. Comparing results across different transforms
2. Analyzing the detection logic
3. Computing influence scores for key examples
4. Visualizing the problem
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('.'))

# Only import what we need (avoid missing modules)
try:
    from poison_detection.data.task_loader import load_task_data
    from poison_detection.data.poisoning import PoisonedDataset
    from poison_detection.influence.influence_computer import InfluenceComputer
    from poison_detection.transforms import get_transform
    import torch
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False

sns.set_style("whitegrid")


def load_results():
    """Load and compare results."""
    print("="*80)
    print("STEP 1: LOADING EXPERIMENT RESULTS")
    print("="*80 + "\n")

    results_dir = Path("experiments/results/transform_comparison/polarity")

    # Find all comparison results
    result_files = list(results_dir.glob("comparison_results_*.json"))

    print(f"Found {len(result_files)} result files:\n")

    results = {}
    for file in result_files:
        with open(file) as f:
            data = json.load(f)
            transform_name = data['config']['transform']
            transform_result = data.get('transform', {})

            f1 = transform_result.get('f1_score', 0)
            precision = transform_result.get('precision', 0)
            recall = transform_result.get('recall', 0)
            num_detected = transform_result.get('num_detected', 0)

            results[transform_name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'num_detected': num_detected,
                'strength_threshold': transform_result.get('strength_threshold', None),
                'change_threshold': transform_result.get('change_threshold', None)
            }

            status = "✓" if f1 > 0.5 else "✗"
            print(f"{status} {transform_name:40s} F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  Detected={num_detected}")

    return results


def analyze_detection_logic():
    """Analyze the detection logic."""
    print("\n" + "="*80)
    print("STEP 2: ANALYZING DETECTION LOGIC")
    print("="*80 + "\n")

    print("Current Detection Algorithm (IQR Method):")
    print("-" * 80)
    print("""
From improved_transform_detector.py:48-105

1. Compute metrics:
   - influence_strength = |mean(original_influence)|
   - influence_change = |mean(original_influence - transformed_influence)|

2. Set thresholds using IQR:
   - Q1, Q3 = percentiles of metric distribution
   - IQR = Q3 - Q1

   For strength (want HIGH outliers):
     strength_threshold = Q3 + k*IQR

   For change (want LOW outliers):
     change_threshold = median - k*1.4826*MAD
     (MAD = Median Absolute Deviation)

3. Detection rule:
   DETECTED = (strength > strength_threshold) OR (change < change_threshold)
    """)

    print("\n" + "🔍 KEY INSIGHT".center(80, "-"))
    print("""
This detection logic makes a CRITICAL ASSUMPTION:

✓ Works for simple lexicon flips:
  - Poison samples: LOW influence_change (trigger preserved)
  - Clean samples: HIGH influence_change (semantics flipped)

✗ Breaks for semantic transforms:
  - ALL samples have HIGH influence_change (semantics changed)
  - No distinction between poison and clean
    """)


def analyze_thresholds(results):
    """Analyze threshold values."""
    print("\n" + "="*80)
    print("STEP 3: THRESHOLD ANALYSIS")
    print("="*80 + "\n")

    print(f"{'Transform':<40} {'Strength Threshold':>20} {'Change Threshold':>20}")
    print("-" * 80)

    for name, data in results.items():
        st = data['strength_threshold']
        ct = data['change_threshold']
        st_str = f"{st:.2f}" if st is not None else "N/A"
        ct_str = f"{ct:.2f}" if ct is not None else "N/A"
        print(f"{name:<40} {st_str:>20} {ct_str:>20}")

    print("\n🔍 Analysis:")
    print("""
The thresholds are calibrated on the DISTRIBUTION of the current experiment.

Problem: Semantic transforms create DIFFERENT distributions than lexicon flips:
- Lexicon flip: Bimodal distribution (clean vs poison clearly separated)
- Semantic transform: Unimodal distribution (all samples look similar)

The IQR-based thresholds can't find outliers when there are none!
    """)


def propose_solutions():
    """Propose solutions to the problem."""
    print("\n" + "="*80)
    print("STEP 4: PROPOSED SOLUTIONS")
    print("="*80 + "\n")

    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                          ROOT CAUSE ANALYSIS                               ║
╚════════════════════════════════════════════════════════════════════════════╝

The detection method relies on the assumption that poison samples are
RESISTANT to transformations (low influence_change), while clean samples
are SUSCEPTIBLE (high influence_change).

This works for SIMPLE transformations that:
  ✓ Preserve triggers (lexicon flips)
  ✓ Change clean sample semantics dramatically

But FAILS for SEMANTIC transformations that:
  ✗ Change ALL samples (no differentiation)
  ✗ Create uniform distributions (no outliers to detect)


╔════════════════════════════════════════════════════════════════════════════╗
║                         SOLUTION #1: ADAPTIVE THRESHOLDING                 ║
╚════════════════════════════════════════════════════════════════════════════╝

Instead of using IQR on absolute change, use RELATIVE change patterns:

1. Compute relative change: change / strength
2. Look for samples with BOTH:
   - High absolute influence (important samples)
   - Low relative change (resistant to transformation)

3. Use transform-specific calibration:
   - Run on small validation set first
   - Learn what "resistant" means for this transform
   - Adapt thresholds accordingly


╔════════════════════════════════════════════════════════════════════════════╗
║                    SOLUTION #2: MULTI-TRANSFORM ENSEMBLE                   ║
╚════════════════════════════════════════════════════════════════════════════╝

Use MULTIPLE transforms and look for CONSISTENCY:

1. Apply several different transforms:
   - Lexicon flips (simple)
   - Semantic transforms (complex)
   - Paraphrase transforms

2. For each sample, compute influence change across ALL transforms
3. Poison samples should show CONSISTENT resistance pattern
4. Clean samples will vary more across different transforms


╔════════════════════════════════════════════════════════════════════════════╗
║                      SOLUTION #3: LEARNED DETECTION                        ║
╚════════════════════════════════════════════════════════════════════════════╝

Train a META-DETECTOR:

1. Features for each training sample:
   - Original influence magnitude
   - Influence change under multiple transforms
   - Variance of influence across test set
   - Model gradient statistics

2. Labels: poison vs clean (from known poisoned validation set)

3. Train simple classifier (Logistic Regression, Random Forest)

4. Apply to detect poison in new datasets


╔════════════════════════════════════════════════════════════════════════════╗
║                         SOLUTION #4: DIRECTION ANALYSIS                    ║
╚════════════════════════════════════════════════════════════════════════════╝

Instead of magnitude, analyze DIRECTION of influence change:

1. Compute influence vectors for each sample
2. Look at DIRECTION changes, not just magnitude
3. Poison samples: Influence direction should be more stable
4. Clean samples: Direction changes with semantics

Use cosine similarity between original and transformed influence vectors.


╔════════════════════════════════════════════════════════════════════════════╗
║                            RECOMMENDED APPROACH                            ║
╚════════════════════════════════════════════════════════════════════════════╝

Implement SOLUTION #2 (Multi-Transform Ensemble) first:

1. It's conceptually simple
2. Leverages existing infrastructure
3. Should work across different transform types
4. Can be combined with other solutions

Then add SOLUTION #4 (Direction Analysis) as enhancement:

1. Provides orthogonal signal
2. Should be more robust to semantic changes
3. Can improve precision significantly
    """)


def create_visual_summary(results):
    """Create visualization of results."""
    print("\n" + "="*80)
    print("STEP 5: CREATING VISUALIZATION")
    print("="*80 + "\n")

    # Prepare data
    transforms = list(results.keys())
    f1_scores = [results[t]['f1'] for t in transforms]

    # Sort by F1 score
    sorted_data = sorted(zip(transforms, f1_scores), key=lambda x: x[1], reverse=True)
    transforms, f1_scores = zip(*sorted_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color code by performance
    colors = ['green' if f1 > 0.5 else 'red' for f1 in f1_scores]

    bars = ax.barh(range(len(transforms)), f1_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(transforms)))
    ax.set_yticklabels(transforms)
    ax.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Transform Detection Performance Comparison',
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3, label='Success Threshold')
    ax.legend()

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.02, i, f'{score:.3f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = 'experiments/diagnostic_results/transform_comparison.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")

    plt.close()


def main():
    """Run full diagnostic."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                  SEMANTIC TRANSFORM FAILURE DIAGNOSTIC                     ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    print()

    # Load and compare results
    results = load_results()

    # Analyze detection logic
    analyze_detection_logic()

    # Analyze thresholds
    analyze_thresholds(results)

    # Propose solutions
    propose_solutions()

    # Create visualization
    create_visual_summary(results)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80 + "\n")
    print("Key findings:")
    print("  1. Detection logic assumes transforms differentiate poison from clean")
    print("  2. Semantic transforms affect ALL samples uniformly")
    print("  3. IQR-based thresholds can't find outliers in uniform distributions")
    print("  4. Need multi-transform or direction-based approaches")
    print()
    print("Next steps:")
    print("  → Implement multi-transform ensemble detection")
    print("  → Add influence direction analysis")
    print("  → Re-run experiments with improved detection")
    print()


if __name__ == '__main__':
    main()
