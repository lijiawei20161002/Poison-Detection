#!/usr/bin/env python3
"""
Comprehensive Aggressive Semantic Transformation Experiment Runner

This is the master script that orchestrates the complete experimental pipeline:
1. Loads configuration from experiment_config.yaml
2. Runs Phase 1: Individual transform testing
3. Analyzes results and identifies best transforms
4. Runs Phase 2: Validation with larger samples
5. Generates comprehensive reports and visualizations

Usage:
    python experiments/run_comprehensive_experiments.py
    python experiments/run_comprehensive_experiments.py --config custom_config.yaml
    python experiments/run_comprehensive_experiments.py --phase 1  # Run only phase 1
    python experiments/run_comprehensive_experiments.py --dry-run  # Preview without execution
"""

import argparse
import subprocess
import time
import json
import yaml
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComprehensiveExperimentRunner:
    """Orchestrates comprehensive aggressive transformation experiments."""

    def __init__(self, config_path: str):
        """Initialize with configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = {
            'metadata': {
                'start_time': datetime.now().isoformat(),
                'config_path': str(config_path)
            },
            'phase_1': {},
            'phase_2': {},
            'summary': {}
        }
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load experiment configuration from YAML."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        """Set up logging configuration."""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'master_runner_{timestamp}.log'

        logging.basicConfig(
            level=getattr(logging, self.config['logging']['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        self.logger = logging.getLogger('ComprehensiveRunner')
        self.logger.info(f"Logging initialized. Log file: {log_file}")

    def print_banner(self):
        """Print experiment banner."""
        banner = f"""
{'='*100}
COMPREHENSIVE AGGRESSIVE SEMANTIC TRANSFORMATION EXPERIMENTS
{'='*100}
Experiment: {self.config['experiment_name']}
Description: {self.config['description']}
Date: {self.config['date']}

Hardware:
  - GPUs: {len(self.config['hardware']['gpus'])} x NVIDIA L40
  - Parallel Jobs: {self.config['hardware']['parallel_jobs']}

Dataset:
  - Task: {self.config['dataset']['task']}
  - Train Samples: {self.config['dataset']['num_train_samples']}
  - Test Samples: {self.config['dataset']['num_test_samples']}
  - Poisoned Fraction: {self.config['dataset']['poisoned_fraction']}

Baseline to Beat:
  - Direct Detection F1: {self.config['baseline']['direct_detection_f1']}
  - Transform Detection F1: {self.config['baseline']['transform_detection_f1']}
  - Best Method: {self.config['baseline']['best_method']}

Phases Enabled:
  - Phase 1 (Individual Transforms): {self.config['phases']['phase_1']['enabled']}
  - Phase 2 (Validation): {self.config['phases']['phase_2']['enabled']}
  - Phase 3 (Combinations): {self.config['phases']['phase_3']['enabled']}
{'='*100}
        """
        print(banner)
        self.logger.info("Experiment configuration loaded and validated")

    def run_phase_1(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Phase 1: Run individual transform experiments in parallel.

        Returns:
            Dictionary with results for each transform
        """
        phase_config = self.config['phases']['phase_1']

        if not phase_config['enabled']:
            self.logger.info("Phase 1 is disabled. Skipping.")
            return {}

        self.logger.info("\n" + "="*100)
        self.logger.info("STARTING PHASE 1: INDIVIDUAL TRANSFORM TESTING")
        self.logger.info("="*100)

        transforms = phase_config['transforms']
        self.logger.info(f"Transforms to test: {len(transforms)}")
        for i, t in enumerate(transforms, 1):
            self.logger.info(f"  {i}. {t}")

        if dry_run:
            self.logger.info("\n[DRY RUN] Would execute Phase 1 with above transforms")
            return {}

        # Prepare command for multi-GPU runner
        script_path = Path(__file__).parent / "run_aggressive_multi_gpu.py"

        output_dir = Path(self.config['output']['base_dir']) / 'phase_1'
        output_dir.mkdir(parents=True, exist_ok=True)

        gpu_list = ','.join(map(str, self.config['hardware']['gpus']))

        cmd = [
            sys.executable,
            str(script_path),
            '--task', self.config['dataset']['task'],
            '--num_train_samples', str(self.config['dataset']['num_train_samples']),
            '--num_test_samples', str(self.config['dataset']['num_test_samples']),
            '--batch_size', str(self.config['hardware']['batch_size']),
            '--output_dir', str(output_dir),
            '--gpus', gpu_list,
            '--run_baseline'  # Include baseline for comparison
        ]

        self.logger.info(f"\nExecuting: {' '.join(cmd)}")

        start_time = time.time()

        # Run the multi-GPU script
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        elapsed_time = time.time() - start_time

        # Log output
        self.logger.info("\n" + "-"*100)
        self.logger.info("PHASE 1 OUTPUT:")
        self.logger.info("-"*100)
        self.logger.info(result.stdout)

        if result.returncode != 0:
            self.logger.error(f"Phase 1 FAILED with return code {result.returncode}")
            self.logger.error(f"Elapsed time: {elapsed_time:.1f}s")
            return {}

        self.logger.info(f"\n✅ Phase 1 completed successfully in {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")

        # Load results
        summary_file = output_dir / self.config['dataset']['task'] / 'experiment_summary.json'
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                phase_1_results = json.load(f)

            self.results['phase_1'] = phase_1_results
            return phase_1_results
        else:
            self.logger.error(f"Summary file not found: {summary_file}")
            return {}

    def analyze_phase_1_results(self) -> Dict[str, Any]:
        """Analyze Phase 1 results and identify best transforms."""
        self.logger.info("\n" + "="*100)
        self.logger.info("ANALYZING PHASE 1 RESULTS")
        self.logger.info("="*100)

        if not self.results['phase_1']:
            self.logger.error("No Phase 1 results to analyze")
            return {}

        transforms_results = self.results['phase_1'].get('transforms', {})

        if not transforms_results:
            self.logger.error("No transform results found")
            return {}

        # Extract F1 scores
        scores = []
        for transform_name, result in transforms_results.items():
            f1 = result.get('f1_score', 0)
            precision = result.get('precision', 0)
            recall = result.get('recall', 0)

            scores.append({
                'name': transform_name,
                'f1': f1,
                'precision': precision,
                'recall': recall
            })

        # Sort by F1 score
        scores.sort(key=lambda x: x['f1'], reverse=True)

        # Print ranking
        self.logger.info("\nTransform Rankings (by F1 Score):")
        self.logger.info("-"*80)
        for i, score in enumerate(scores, 1):
            status = "✅" if score['f1'] >= self.config['baseline']['direct_detection_f1'] else "❌"
            self.logger.info(
                f"{i}. {status} {score['name']:35s} "
                f"F1={score['f1']:.4f}, P={score['precision']:.4f}, R={score['recall']:.4f}"
            )

        # Identify best
        best_transform = scores[0]
        baseline_f1 = self.config['baseline']['direct_detection_f1']

        self.logger.info("\n" + "="*80)
        self.logger.info("BEST TRANSFORM IDENTIFIED:")
        self.logger.info("="*80)
        self.logger.info(f"Name: {best_transform['name']}")
        self.logger.info(f"F1 Score: {best_transform['f1']:.4f}")
        self.logger.info(f"Precision: {best_transform['precision']:.4f}")
        self.logger.info(f"Recall: {best_transform['recall']:.4f}")

        if best_transform['f1'] >= baseline_f1:
            improvement = (best_transform['f1'] - baseline_f1) / baseline_f1 * 100
            self.logger.info(f"\n✅✅ BEATS BASELINE by {improvement:.1f}%")
        else:
            gap = baseline_f1 - best_transform['f1']
            self.logger.info(f"\n❌ Falls short of baseline by {gap:.4f} ({gap/baseline_f1*100:.1f}%)")

        analysis = {
            'rankings': scores,
            'best_transform': best_transform,
            'beats_baseline': best_transform['f1'] >= baseline_f1,
            'baseline_f1': baseline_f1
        }

        self.results['summary']['phase_1_analysis'] = analysis

        return analysis

    def run_phase_2(self, best_transform: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Phase 2: Validate best transform with larger samples.

        Args:
            best_transform: Name of the best performing transform from Phase 1
            dry_run: If True, only preview without execution

        Returns:
            Validation results
        """
        phase_config = self.config['phases']['phase_2']

        if not phase_config['enabled']:
            self.logger.info("Phase 2 is disabled. Skipping.")
            return {}

        self.logger.info("\n" + "="*100)
        self.logger.info("STARTING PHASE 2: BEST TRANSFORM VALIDATION")
        self.logger.info("="*100)
        self.logger.info(f"Validating transform: {best_transform}")
        self.logger.info(f"Runs: {phase_config['config']['runs']}")
        self.logger.info(f"Train samples: {phase_config['config']['num_train_samples']}")
        self.logger.info(f"Test samples: {phase_config['config']['num_test_samples']}")

        if dry_run:
            self.logger.info("\n[DRY RUN] Would execute Phase 2 validation")
            return {}

        script_path = Path(__file__).parent / "run_single_aggressive_transform.py"
        output_dir = Path(self.config['output']['base_dir']) / 'phase_2' / best_transform
        output_dir.mkdir(parents=True, exist_ok=True)

        validation_results = []

        # Run multiple times for statistical significance
        for run_idx in range(phase_config['config']['runs']):
            self.logger.info(f"\n--- Validation Run {run_idx + 1}/{phase_config['config']['runs']} ---")

            run_output_dir = output_dir / f"run_{run_idx + 1}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(script_path),
                '--task', self.config['dataset']['task'],
                '--num_train_samples', str(phase_config['config']['num_train_samples']),
                '--num_test_samples', str(phase_config['config']['num_test_samples']),
                '--batch_size', str(self.config['hardware']['batch_size']),
                '--output_dir', str(run_output_dir),
                '--transform', best_transform,
                '--device', 'cuda:0'
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            if result.returncode == 0:
                self.logger.info(f"✅ Run {run_idx + 1} completed successfully")

                # Load results
                result_file = run_output_dir / f"{best_transform}_results.json"
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        run_result = json.load(f)
                        validation_results.append(run_result)
            else:
                self.logger.error(f"❌ Run {run_idx + 1} failed")
                self.logger.error(result.stdout)

        # Aggregate results
        if validation_results:
            avg_f1 = sum(r.get('f1_score', 0) for r in validation_results) / len(validation_results)
            avg_precision = sum(r.get('precision', 0) for r in validation_results) / len(validation_results)
            avg_recall = sum(r.get('recall', 0) for r in validation_results) / len(validation_results)

            self.logger.info("\n" + "="*80)
            self.logger.info("PHASE 2 VALIDATION RESULTS:")
            self.logger.info("="*80)
            self.logger.info(f"Average F1: {avg_f1:.4f}")
            self.logger.info(f"Average Precision: {avg_precision:.4f}")
            self.logger.info(f"Average Recall: {avg_recall:.4f}")

            phase_2_results = {
                'transform': best_transform,
                'runs': validation_results,
                'average': {
                    'f1_score': avg_f1,
                    'precision': avg_precision,
                    'recall': avg_recall
                }
            }

            self.results['phase_2'] = phase_2_results
            return phase_2_results

        return {}

    def generate_final_report(self):
        """Generate comprehensive final report."""
        self.logger.info("\n" + "="*100)
        self.logger.info("GENERATING FINAL COMPREHENSIVE REPORT")
        self.logger.info("="*100)

        report_dir = Path(self.config['output']['base_dir']) / 'reports'
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        results_file = report_dir / f'comprehensive_results_{timestamp}.json'
        self.results['metadata']['end_time'] = datetime.now().isoformat()

        with open(results_file, 'w') as f:
            json.dump(self.results, indent=2, fp=f)

        self.logger.info(f"✅ Results saved to: {results_file}")

        # Generate text report
        report_file = report_dir / f'comprehensive_report_{timestamp}.txt'

        with open(report_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("COMPREHENSIVE AGGRESSIVE SEMANTIC TRANSFORMATION EXPERIMENTS\n")
            f.write("FINAL REPORT\n")
            f.write("="*100 + "\n\n")

            f.write(f"Experiment: {self.config['experiment_name']}\n")
            f.write(f"Date: {self.config['date']}\n")
            f.write(f"Start Time: {self.results['metadata']['start_time']}\n")
            f.write(f"End Time: {self.results['metadata']['end_time']}\n\n")

            f.write("-"*100 + "\n")
            f.write("PHASE 1: INDIVIDUAL TRANSFORM TESTING\n")
            f.write("-"*100 + "\n\n")

            if 'phase_1_analysis' in self.results['summary']:
                analysis = self.results['summary']['phase_1_analysis']

                f.write("Transform Rankings:\n")
                for i, score in enumerate(analysis['rankings'], 1):
                    f.write(f"  {i}. {score['name']:35s} "
                           f"F1={score['f1']:.4f}, P={score['precision']:.4f}, R={score['recall']:.4f}\n")

                f.write(f"\nBest Transform: {analysis['best_transform']['name']}\n")
                f.write(f"  F1 Score: {analysis['best_transform']['f1']:.4f}\n")
                f.write(f"  Baseline F1: {analysis['baseline_f1']:.4f}\n")
                f.write(f"  Beats Baseline: {'YES ✅' if analysis['beats_baseline'] else 'NO ❌'}\n")

            f.write("\n" + "-"*100 + "\n")
            f.write("PHASE 2: VALIDATION RESULTS\n")
            f.write("-"*100 + "\n\n")

            if self.results['phase_2']:
                phase_2 = self.results['phase_2']
                f.write(f"Validated Transform: {phase_2['transform']}\n")
                f.write(f"Number of Runs: {len(phase_2['runs'])}\n\n")

                f.write("Average Performance:\n")
                f.write(f"  F1 Score: {phase_2['average']['f1_score']:.4f}\n")
                f.write(f"  Precision: {phase_2['average']['precision']:.4f}\n")
                f.write(f"  Recall: {phase_2['average']['recall']:.4f}\n")

            f.write("\n" + "="*100 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*100 + "\n")

        self.logger.info(f"✅ Text report saved to: {report_file}")

        return results_file, report_file

    def run(self, phase: int = None, dry_run: bool = False):
        """
        Run the complete experiment pipeline.

        Args:
            phase: If specified, run only this phase (1 or 2)
            dry_run: If True, preview without execution
        """
        self.print_banner()

        if dry_run:
            self.logger.info("\n" + "="*100)
            self.logger.info("DRY RUN MODE - No experiments will be executed")
            self.logger.info("="*100)

        try:
            # Phase 1: Individual transform testing
            if phase is None or phase == 1:
                phase_1_results = self.run_phase_1(dry_run=dry_run)

                if phase_1_results and not dry_run:
                    analysis = self.analyze_phase_1_results()

                    # Phase 2: Validation (if enabled and Phase 1 succeeded)
                    if (phase is None and
                        self.config['phases']['phase_2']['enabled'] and
                        analysis):

                        best_transform = analysis['best_transform']['name']
                        self.run_phase_2(best_transform, dry_run=dry_run)

            # Phase 2 only
            elif phase == 2:
                if not self.results['phase_1']:
                    self.logger.error("Cannot run Phase 2 without Phase 1 results")
                    self.logger.error("Please run Phase 1 first or provide results")
                    return

                analysis = self.analyze_phase_1_results()
                if analysis:
                    best_transform = analysis['best_transform']['name']
                    self.run_phase_2(best_transform, dry_run=dry_run)

            # Generate final report
            if not dry_run:
                results_file, report_file = self.generate_final_report()

                self.logger.info("\n" + "="*100)
                self.logger.info("EXPERIMENT COMPLETE!")
                self.logger.info("="*100)
                self.logger.info(f"Results: {results_file}")
                self.logger.info(f"Report: {report_file}")
                self.logger.info("="*100)

        except Exception as e:
            self.logger.error(f"Experiment failed with error: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive aggressive semantic transformation experiments'
    )
    parser.add_argument('--config', type=str,
                       default='experiments/experiment_config.yaml',
                       help='Path to experiment configuration file')
    parser.add_argument('--phase', type=int, choices=[1, 2],
                       help='Run only specific phase (1 or 2)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview experiment plan without execution')

    args = parser.parse_args()

    # Initialize and run
    runner = ComprehensiveExperimentRunner(args.config)
    runner.run(phase=args.phase, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
