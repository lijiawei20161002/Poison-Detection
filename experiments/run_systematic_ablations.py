#!/usr/bin/env python3
"""
Systematic ablation study for semantic transformations.

This addresses reviewer concerns about "ad-hoc" transformations by running
comprehensive ablations across all defined transformations for each task.

For each transformation, we report:
- Basic summary statistics of influence distributions
- Detection metrics (precision/recall, F1)
- ASR (Attack Success Rate) before/after removal

Usage:
    python experiments/run_systematic_ablations.py --task sentiment
    python experiments/run_systematic_ablations.py --task math --model llama3-8b
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from poison_detection.data.transforms import transform_registry, get_transform_info
from experiments.run_llm_experiments import LLMExperiment, ExperimentConfig, MODEL_NAMES


class AblationStudy:
    """Systematic ablation study runner."""

    def __init__(self, task_type: str, model_name: str, output_dir: Path):
        self.task_type = task_type
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "task": task_type,
            "model": model_name,
            "transforms": {}
        }

    def run_ablation(self, attack_type: str = "single_trigger"):
        """Run ablation across all transformations for the task."""
        print(f"\n{'#'*80}")
        print(f"# SYSTEMATIC ABLATION STUDY")
        print(f"# Task: {self.task_type}")
        print(f"# Model: {self.model_name}")
        print(f"# Attack: {attack_type}")
        print(f"{'#'*80}\n")

        # Get all transforms for this task
        all_transforms = transform_registry.get_all_transforms(self.task_type)
        transform_names = list(all_transforms.keys())

        print(f"Testing {len(transform_names)} transformations:")
        for name in transform_names:
            config = all_transforms[name].config
            expected = "✓" if config.expected_to_work else "✗"
            print(f"  {expected} {name}: {config.description}")

        print()

        # Run experiment for each transform
        for transform_name in transform_names:
            print(f"\n{'='*80}")
            print(f"Transform: {transform_name}")
            print(f"{'='*80}")

            try:
                result = self._run_single_transform(
                    transform_name,
                    attack_type
                )
                self.results["transforms"][transform_name] = result

                # Print summary
                if "detection" in result:
                    det = result["detection"]
                    print(f"\n  Results:")
                    print(f"    Precision: {det['precision']:.4f}")
                    print(f"    Recall: {det['recall']:.4f}")
                    print(f"    F1: {det['f1_score']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                self.results["transforms"][transform_name] = {
                    "error": str(e)
                }

        # Generate summary report
        self._generate_summary_report()

        # Save results
        output_file = self.output_dir / f"ablation_{self.task_type}_{self.model_name}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\n{'='*80}")
        print(f"✓ Ablation study complete!")
        print(f"  Results: {output_file}")
        print(f"{'='*80}\n")

    def _run_single_transform(self, transform_name: str, attack_type: str) -> Dict:
        """Run experiment with a single transformation."""
        # Create temporary args
        class Args:
            pass

        args = Args()
        args.model = self.model_name
        args.task = self.task_type
        args.attack_type = attack_type
        args.poison_ratio = 0.01
        args.use_4bit = True  # Use quantization for efficiency
        args.use_8bit = False
        args.transforms = [transform_name]
        args.output_dir = str(self.output_dir / "temp")
        args.seed = 42
        args.max_samples = 1000  # Limit samples for speed

        config = ExperimentConfig(args)

        # Run experiment
        experiment = LLMExperiment(config)
        results = experiment.run()

        return results

    def _generate_summary_report(self):
        """Generate summary report and visualizations."""
        print(f"\n{'='*80}")
        print(f"SUMMARY REPORT")
        print(f"{'='*80}\n")

        # Collect metrics
        summary_data = []

        for transform_name, result in self.results["transforms"].items():
            if "error" in result:
                continue

            transform = transform_registry.get_transform(
                self.task_type,
                transform_name
            )

            if "detection" in result:
                det = result["detection"]
                summary_data.append({
                    "Transform": transform_name,
                    "Expected to work": transform.config.expected_to_work,
                    "Precision": det["precision"],
                    "Recall": det["recall"],
                    "F1 Score": det["f1_score"],
                    "Detected": det["num_detected"]
                })

        if not summary_data:
            print("No successful experiments to summarize.")
            return

        # Create DataFrame
        df = pd.DataFrame(summary_data)

        # Print table
        print(df.to_string(index=False))
        print()

        # Save CSV
        csv_file = self.output_dir / f"ablation_summary_{self.task_type}_{self.model_name}.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Summary table saved: {csv_file}")

        # Generate plots
        self._generate_plots(df)

    def _generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot 1: Detection metrics by transform
        metrics_df = df[["Transform", "Precision", "Recall", "F1 Score"]].set_index("Transform")
        metrics_df.plot(kind="bar", ax=axes[0])
        axes[0].set_title("Detection Metrics by Transform")
        axes[0].set_ylabel("Score")
        axes[0].set_xlabel("Transform")
        axes[0].legend(loc="lower right")
        axes[0].set_ylim([0, 1])
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2: Expected vs Actual Performance
        expected_work = df[df["Expected to work"] == True]["F1 Score"].mean() if len(df[df["Expected to work"] == True]) > 0 else 0
        expected_fail = df[df["Expected to work"] == False]["F1 Score"].mean() if len(df[df["Expected to work"] == False]) > 0 else 0

        axes[1].bar(["Expected to Work", "Expected to Fail"], [expected_work, expected_fail])
        axes[1].set_title("F1 Score by Expected Behavior")
        axes[1].set_ylabel("Average F1 Score")
        axes[1].set_ylim([0, 1])

        # Plot 3: Number of detections
        df.plot(x="Transform", y="Detected", kind="bar", ax=axes[2], legend=False)
        axes[2].set_title("Number of Detected Poisons")
        axes[2].set_ylabel("Count")
        axes[2].set_xlabel("Transform")
        axes[2].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_file = self.output_dir / f"ablation_plots_{self.task_type}_{self.model_name}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plots saved: {plot_file}")

        # Print insights
        print(f"\n{'='*80}")
        print(f"KEY INSIGHTS")
        print(f"{'='*80}\n")

        print(f"1. Transforms expected to work:")
        print(f"   Average F1: {expected_work:.4f}")
        print()

        print(f"2. Transforms expected to fail:")
        print(f"   Average F1: {expected_fail:.4f}")
        print()

        # Best and worst transforms
        best_transform = df.loc[df["F1 Score"].idxmax()]
        worst_transform = df.loc[df["F1 Score"].idxmin()]

        print(f"3. Best performing transform:")
        print(f"   {best_transform['Transform']} (F1: {best_transform['F1 Score']:.4f})")
        print()

        print(f"4. Worst performing transform:")
        print(f"   {worst_transform['Transform']} (F1: {worst_transform['F1 Score']:.4f})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run systematic ablation study for transformations"
    )

    parser.add_argument(
        "--task", type=str, required=True,
        choices=["sentiment", "math"],
        help="Task type"
    )

    parser.add_argument(
        "--model", type=str, default="t5-small",
        help="Model to use (use smaller models for faster ablations)"
    )

    parser.add_argument(
        "--attack-type", type=str, default="single_trigger",
        choices=["single_trigger", "multi_trigger", "label_preserving"],
        help="Type of backdoor attack"
    )

    parser.add_argument(
        "--output-dir", type=str, default="experiments/ablations",
        help="Output directory"
    )

    args = parser.parse_args()

    # Run ablation study
    study = AblationStudy(
        task_type=args.task,
        model_name=args.model,
        output_dir=Path(args.output_dir)
    )

    study.run_ablation(attack_type=args.attack_type)


if __name__ == "__main__":
    main()
