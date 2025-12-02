"""
Quick script to test individual transformations and their properties.

This provides a fast way to:
1. Test if a transformation properly inverts sentiment/semantics
2. Visualize the transformation effect on sample data
3. Get immediate feedback on transformation quality

Usage:
    python experiments/quick_transform_test.py --task sentiment --transform prefix_negation
    python experiments/quick_transform_test.py --task math --transform opposite_question --samples 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry, get_transform_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quick transformation testing')

    parser.add_argument(
        '--task',
        type=str,
        required=True,
        choices=['sentiment', 'math', 'qa'],
        help='Task type'
    )

    parser.add_argument(
        '--transform',
        type=str,
        help='Transformation name (if None, show all)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=5,
        help='Number of test samples to show'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available transformations'
    )

    return parser.parse_args()


def get_test_samples(task_type: str, num_samples: int = 5) -> List[str]:
    """Get test samples for a task type."""

    if task_type == 'sentiment':
        samples = [
            "This movie was absolutely fantastic! I loved every minute of it.",
            "Terrible experience. Would not recommend to anyone.",
            "The product works well and met my expectations.",
            "Worst purchase I've ever made. Complete waste of money.",
            "Pretty good overall, though there were some minor issues."
        ]
    elif task_type == 'math':
        samples = [
            "If John has 5 apples and gives 2 to Mary, how many does he have left?",
            "Calculate the sum of 15 and 27.",
            "A train travels at 60 mph for 3 hours. How far does it go?",
            "If x + 5 = 12, what is the value of x?",
            "What is 20% of 150?"
        ]
    elif task_type == 'qa':
        samples = [
            "Is Paris the capital of France?",
            "Can dogs fly?",
            "Is water necessary for human survival?",
            "Are there more than 7 continents on Earth?",
            "Is the Sun a star?"
        ]
    else:
        samples = []

    return samples[:num_samples]


def test_transformation(
    task_type: str,
    transform_name: str,
    samples: List[str]
) -> List[Tuple[str, str]]:
    """
    Test a transformation on samples.

    Args:
        task_type: Task type
        transform_name: Transformation name
        samples: List of test samples

    Returns:
        List of (original, transformed) tuples
    """
    transform = transform_registry.get_transform(task_type, transform_name)

    results = []
    for sample in samples:
        transformed = transform(sample)
        results.append((sample, transformed))

    return results


def print_transformation_info(task_type: str):
    """Print information about all transformations for a task."""

    print(f"\n{'=' * 80}")
    print(f"AVAILABLE TRANSFORMATIONS FOR TASK: {task_type.upper()}")
    print(f"{'=' * 80}\n")

    info = get_transform_info(task_type)

    if task_type in info:
        transforms_info = []
        for name, details in info[task_type].items():
            transforms_info.append({
                'Name': name,
                'Description': details['description'],
                'Expected to Work': '✓' if details['expected_to_work'] else '✗'
            })

        df = pd.DataFrame(transforms_info)
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print(f"No transformations available for task: {task_type}")


def print_test_results(
    transform_name: str,
    results: List[Tuple[str, str]],
    task_type: str
):
    """Print test results in a readable format."""

    print(f"\n{'=' * 80}")
    print(f"TESTING TRANSFORMATION: {transform_name}")
    print(f"Task Type: {task_type}")
    print(f"{'=' * 80}\n")

    # Get transformation info
    info = get_transform_info(task_type)
    if task_type in info and transform_name in info[task_type]:
        details = info[task_type][transform_name]
        print(f"Description: {details['description']}")
        print(f"Expected to Work: {'Yes' if details['expected_to_work'] else 'No'}")
        print()

    # Print examples
    for i, (original, transformed) in enumerate(results, 1):
        print(f"Example {i}:")
        print(f"  Original:    {original}")
        print(f"  Transformed: {transformed}")
        print()

    # Analysis
    print("Analysis:")

    # Check if transformation actually changes the text
    unchanged_count = sum(1 for orig, trans in results if orig == trans)
    if unchanged_count > 0:
        print(f"  ⚠ WARNING: {unchanged_count}/{len(results)} samples unchanged!")
    else:
        print(f"  ✓ All samples were transformed")

    # Check length changes
    avg_length_change = sum(len(trans) - len(orig) for orig, trans in results) / len(results)
    print(f"  Average length change: {avg_length_change:+.1f} characters")

    if task_type == 'sentiment':
        # For sentiment, check if negation words are added
        negation_words = ['not', 'opposite', 'contrary', 'reverse', 'no', 'never']
        has_negation = sum(
            1 for _, trans in results
            if any(word in trans.lower() for word in negation_words)
        )
        print(f"  Samples with negation words: {has_negation}/{len(results)}")


def main():
    """Main function."""
    args = parse_args()

    # List transformations
    if args.list or not args.transform:
        print_transformation_info(args.task)
        if not args.transform:
            return

    # Test specific transformation
    print(f"\nGenerating {args.samples} test samples for task: {args.task}")
    samples = get_test_samples(args.task, args.samples)

    print(f"Testing transformation: {args.transform}")
    try:
        results = test_transformation(args.task, args.transform, samples)
        print_test_results(args.transform, results, args.task)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nAvailable transformations:")
        print_transformation_info(args.task)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
