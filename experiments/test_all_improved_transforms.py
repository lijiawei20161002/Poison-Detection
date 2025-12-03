#!/usr/bin/env python3
"""
Test all improved transformations on a small subset for quick validation.

This script runs the detection experiment with each new transformation to compare performance.
"""

import subprocess
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def run_experiment(transform_name: str, test_samples: int = 50, device: str = "cuda:0"):
    """Run detection experiment with a specific transformation."""

    output_suffix = f"_{transform_name}"
    output_dir = f"experiments/results/transform_comparison"

    cmd = [
        "python3",
        "experiments/compare_direct_vs_transform_detection.py",
        "--task", "polarity",
        "--model", "data/polarity/outputs/final_model",
        "--num_train_samples", "1000",
        "--num_test_samples", str(test_samples),
        "--batch_size", "8",
        "--device", device,
        "--transform", transform_name,
        "--output_dir", output_dir,
        "--data_dir", "data",
        "--output_suffix", output_suffix
    ]

    print(f"\n{'=' * 80}")
    print(f"Testing transformation: {transform_name}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd="/mnt/nw/home/j.li/Poison-Detection",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode == 0:
            print(f"✓ SUCCESS: {transform_name}")
            return True, None
        else:
            print(f"✗ FAILED: {transform_name}")
            print(f"Error output:\n{result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {transform_name}")
        return False, "Timeout after 10 minutes"
    except Exception as e:
        print(f"✗ ERROR: {transform_name} - {e}")
        return False, str(e)


def load_and_compare_results():
    """Load and compare results from all transformations."""

    results_dir = Path("/mnt/nw/home/j.li/Poison-Detection/experiments/results/transform_comparison/polarity")

    if not results_dir.exists():
        print("No results directory found!")
        return

    results = []

    # Find all comparison results files
    for result_file in results_dir.glob("comparison_results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)

            transform_name = result_file.stem.replace("comparison_results_", "")

            # Extract key metrics
            transform_result = None
            for method in data.get('results', []):
                if method.get('category') == 'transform':
                    transform_result = method
                    break

            if transform_result:
                results.append({
                    'transform': transform_name,
                    'f1_score': transform_result.get('f1_score', 0.0),
                    'precision': transform_result.get('precision', 0.0),
                    'recall': transform_result.get('recall', 0.0),
                    'num_detected': transform_result.get('num_detected', 0),
                    'time': transform_result.get('time', 0.0)
                })

        except Exception as e:
            print(f"Error loading {result_file}: {e}")

    if not results:
        print("No valid results found!")
        return

    # Sort by F1 score
    results.sort(key=lambda x: x['f1_score'], reverse=True)

    # Print comparison table
    print("\n" + "=" * 100)
    print("TRANSFORMATION COMPARISON RESULTS")
    print("=" * 100)
    print(f"{'Transformation':<30} {'F1 Score':>10} {'Precision':>10} {'Recall':>10} {'Detected':>10} {'Time (s)':>10}")
    print("-" * 100)

    for r in results:
        print(f"{r['transform']:<30} {r['f1_score']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['num_detected']:>10} {r['time']:>10.4f}")

    print("=" * 100)

    # Print best transformation
    if results:
        best = results[0]
        print(f"\n🏆 BEST TRANSFORMATION: {best['transform']}")
        print(f"   F1 Score: {best['f1_score']:.4f}")
        print(f"   Precision: {best['precision']:.4f}")
        print(f"   Recall: {best['recall']:.4f}")
        print(f"   Detected: {best['num_detected']} samples")

        # Calculate improvement over prefix_negation
        prefix_negation = next((r for r in results if 'prefix_negation' in r['transform']), None)
        if prefix_negation and best['transform'] != 'prefix_negation':
            improvement = (best['f1_score'] - prefix_negation['f1_score']) / max(prefix_negation['f1_score'], 0.001) * 100
            print(f"\n   Improvement over prefix_negation: {improvement:+.1f}%")


def main():
    """Main test function."""

    print("=" * 80)
    print("IMPROVED TRANSFORMATION TEST SUITE")
    print("=" * 80)
    print()
    print("This script will test the following transformations:")
    print("  1. prefix_negation (baseline - expected to fail)")
    print("  2. lexicon_flip (original - expected to work)")
    print("  3. grammatical_negation (NEW - expected to work)")
    print("  4. strong_lexicon_flip (NEW - expected to work better)")
    print("  5. combined_flip_negation (NEW - expected to work best)")
    print()

    # Transformations to test
    transforms = [
        "prefix_negation",  # Baseline (weak)
        "lexicon_flip",  # Original (should work)
        "grammatical_negation",  # New
        "strong_lexicon_flip",  # New
        "combined_flip_negation",  # New (expected best)
    ]

    # Check CUDA availability
    import torch
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"✓ CUDA available - using {device}")
    else:
        device = "cpu"
        print(f"⚠ CUDA not available - using CPU (will be slow)")
    print()

    # Run experiments
    results_summary = {}

    for transform in transforms:
        success, error = run_experiment(transform, test_samples=50, device=device)
        results_summary[transform] = {"success": success, "error": error}

    # Print summary
    print("\n" + "=" * 80)
    print("EXECUTION SUMMARY")
    print("=" * 80)

    for transform, result in results_summary.items():
        status = "✓ SUCCESS" if result['success'] else f"✗ FAILED: {result['error']}"
        print(f"{transform:<30} {status}")

    # Load and compare results
    print("\n")
    load_and_compare_results()

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
