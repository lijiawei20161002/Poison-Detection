#!/usr/bin/env python
"""
Comprehensive poison detection comparison across T5, LLaMA, and Qwen models.

This script runs systematic experiments across all model families and generates
detailed comparison reports compatible with existing T5 baseline results.
"""

import argparse
import time
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys

# Model configurations matching existing T5 baselines
MODEL_CONFIGS = {
    # Baseline
    't5-small': {
        'name': 'google/t5-small-lm-adapt',
        'params': '77M',
        'family': 't5',
        'script': 'experiments/compare_models.py',
        'is_baseline': True
    },
    # LLaMA models
    'llama-1b': {
        'name': 'meta-llama/Llama-3.2-1B',
        'params': '1B',
        'family': 'llama',
        'script': 'experiments/run_llama_experiments.py'
    },
    'llama-3b': {
        'name': 'meta-llama/Llama-3.2-3B',
        'params': '3B',
        'family': 'llama',
        'script': 'experiments/run_llama_experiments.py'
    },
    # Qwen models
    'qwen-0.5b': {
        'name': 'Qwen/Qwen2.5-0.5B',
        'params': '0.5B',
        'family': 'qwen',
        'script': 'experiments/run_qwen_experiments.py'
    },
    'qwen-1.5b': {
        'name': 'Qwen/Qwen2.5-1.5B',
        'params': '1.5B',
        'family': 'qwen',
        'script': 'experiments/run_qwen_experiments.py'
    },
    'qwen-3b': {
        'name': 'Qwen/Qwen2.5-3B',
        'params': '3B',
        'family': 'qwen',
        'script': 'experiments/run_qwen_experiments.py'
    }
}

# Experiment configurations matching existing T5 experiments
EXPERIMENT_CONFIGS = {
    'quick': {
        'num_train_samples': 100,
        'num_test_samples': 50,
        'description': 'Quick test (100 train, 50 test)'
    },
    'medium': {
        'num_train_samples': 500,
        'num_test_samples': 100,
        'description': 'Medium scale (500 train, 100 test) - matches baseline'
    },
    'full': {
        'num_train_samples': 1000,
        'num_test_samples': 200,
        'description': 'Full scale (1000 train, 200 test)'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run comprehensive model comparison experiments'
    )
    parser.add_argument('--models', nargs='+',
                      default=['t5-small', 'llama-1b', 'qwen-0.5b'],
                      choices=list(MODEL_CONFIGS.keys()),
                      help='Models to test')
    parser.add_argument('--tasks', nargs='+',
                      default=['polarity'],
                      help='Tasks to run')
    parser.add_argument('--scale', type=str, default='medium',
                      choices=list(EXPERIMENT_CONFIGS.keys()),
                      help='Experiment scale')
    parser.add_argument('--output_dir', type=str,
                      default=f'experiments/results/comprehensive_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                      help='Output directory')
    parser.add_argument('--use_8bit', action='store_true',
                      help='Use 8-bit quantization for larger models')
    parser.add_argument('--skip_on_error', action='store_true',
                      help='Continue if a model fails')
    parser.add_argument('--detection_methods', nargs='+',
                      default=['percentile_high', 'top_k_low', 'local_outlier_factor'],
                      help='Detection methods to test')
    parser.add_argument('--dry_run', action='store_true',
                      help='Print commands without executing')
    return parser.parse_args()


def run_model_experiment(model_key, model_config, task, exp_config, args):
    """Run experiment for a single model."""
    print("\n" + "="*80)
    print(f"RUNNING: {model_key.upper()} on {task}")
    print("="*80)
    print(f"Model: {model_config['name']}")
    print(f"Parameters: {model_config['params']}")
    print(f"Scale: {exp_config['description']}")
    print()

    # Prepare output directory
    output_dir = Path(args.output_dir) / args.scale / task / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build command based on model family
    if model_config['family'] == 't5':
        # Use unified compare_models.py for T5
        cmd = [
            sys.executable, 'experiments/compare_models.py',
            '--models', 't5-small',
            '--task', task,
            '--num_train_samples', str(exp_config['num_train_samples']),
            '--num_test_samples', str(exp_config['num_test_samples']),
            '--output_dir', str(output_dir.parent),
            '--detection_methods', *args.detection_methods
        ]
    elif model_config['family'] == 'llama':
        cmd = [
            sys.executable, 'experiments/run_llama_experiments.py',
            '--model', model_config['name'],
            '--task', task,
            '--num_train_samples', str(exp_config['num_train_samples']),
            '--num_test_samples', str(exp_config['num_test_samples']),
            '--output_dir', str(output_dir.parent),
            '--detection_methods', *args.detection_methods
        ]
    elif model_config['family'] == 'qwen':
        cmd = [
            sys.executable, 'experiments/run_qwen_experiments.py',
            '--model', model_config['name'],
            '--task', task,
            '--num_train_samples', str(exp_config['num_train_samples']),
            '--num_test_samples', str(exp_config['num_test_samples']),
            '--output_dir', str(output_dir.parent),
            '--detection_methods', *args.detection_methods
        ]
    else:
        raise ValueError(f"Unknown model family: {model_config['family']}")

    # Add optional flags
    if args.use_8bit and model_config['params'] not in ['77M', '0.5B']:
        cmd.append('--use_8bit')

    # Execute command
    if args.dry_run:
        print("DRY RUN - Command:")
        print(" ".join(cmd))
        return {'status': 'dry_run', 'model': model_key, 'task': task}

    print("Executing:", " ".join(cmd))
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        runtime = time.time() - start_time

        print(f"\n✓ Completed in {runtime:.1f}s")

        return {
            'status': 'success',
            'model': model_key,
            'model_name': model_config['name'],
            'task': task,
            'runtime': runtime,
            'output': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
        }

    except subprocess.CalledProcessError as e:
        runtime = time.time() - start_time
        print(f"\n✗ Failed after {runtime:.1f}s")
        print(f"Error: {e.stderr[-500:]}")

        if not args.skip_on_error:
            raise

        return {
            'status': 'failed',
            'model': model_key,
            'task': task,
            'runtime': runtime,
            'error': str(e),
            'stderr': e.stderr[-500:]
        }


def generate_master_report(all_results, args):
    """Generate master comparison report."""
    output_dir = Path(args.output_dir)
    report_path = output_dir / 'MASTER_REPORT.md'

    with open(report_path, 'w') as f:
        f.write("# Poison Detection Model Comparison - Master Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Scale:** {EXPERIMENT_CONFIGS[args.scale]['description']}\n\n")
        f.write(f"**Tasks:** {', '.join(args.tasks)}\n\n")
        f.write(f"**Detection Methods:** {', '.join(args.detection_methods)}\n\n")

        # Summary table
        f.write("## Experiment Summary\n\n")
        f.write("| Model | Parameters | Tasks Run | Status |\n")
        f.write("|-------|-----------|-----------|--------|\n")

        model_status = {}
        for result in all_results:
            model = result['model']
            if model not in model_status:
                model_status[model] = {'success': 0, 'failed': 0, 'total': 0}
            model_status[model]['total'] += 1
            if result['status'] == 'success':
                model_status[model]['success'] += 1
            else:
                model_status[model]['failed'] += 1

        for model_key, stats in model_status.items():
            config = MODEL_CONFIGS[model_key]
            status = f"{stats['success']}/{stats['total']} succeeded"
            if stats['failed'] > 0:
                status += f" ({stats['failed']} failed)"
            f.write(f"| {model_key} | {config['params']} | {stats['total']} | {status} |\n")

        # Detailed results per task
        f.write("\n## Results by Task\n\n")
        for task in args.tasks:
            f.write(f"### Task: {task}\n\n")

            task_results = [r for r in all_results if r['task'] == task and r['status'] == 'success']

            if task_results:
                f.write("**Successful Experiments:**\n\n")
                for result in task_results:
                    f.write(f"- **{result['model']}**: Completed in {result['runtime']:.1f}s\n")
                    # Try to load detailed results
                    results_file = Path(args.output_dir) / args.scale / task / result['model'] / f"{task}_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file) as rf:
                                data = json.load(rf)
                                if 'results' in data:
                                    f.write("  - Results saved to: `{}`\n".format(results_file.relative_to(output_dir)))
                        except:
                            pass

            failed_results = [r for r in all_results if r['task'] == task and r['status'] == 'failed']
            if failed_results:
                f.write("\n**Failed Experiments:**\n\n")
                for result in failed_results:
                    f.write(f"- **{result['model']}**: {result.get('error', 'Unknown error')}\n")

            f.write("\n")

        # Result locations
        f.write("## Result Locations\n\n")
        f.write("Results are organized as follows:\n\n")
        f.write("```\n")
        f.write(f"{output_dir}/\n")
        f.write(f"├── {args.scale}/\n")
        for task in args.tasks:
            f.write(f"│   ├── {task}/\n")
            for model in args.models:
                f.write(f"│   │   ├── {model}/\n")
                f.write(f"│   │   │   └── {task}_results.json\n")
        f.write("│   └── comparison_report.txt\n")
        f.write("└── MASTER_REPORT.md (this file)\n")
        f.write("```\n\n")

        # Next steps
        f.write("## Next Steps\n\n")
        f.write("1. **View detailed results**: Check individual `*_results.json` files for each model\n")
        f.write("2. **Compare performance**: Review `comparison_report.txt` in each task directory\n")
        f.write("3. **Analyze F1 scores**: Look at detection method performance across models\n")
        f.write("4. **Runtime comparison**: Compare computational efficiency across model families\n\n")

        # Commands for analysis
        f.write("## Analysis Commands\n\n")
        f.write("```bash\n")
        f.write(f"# View all results\n")
        f.write(f"cd {output_dir}\n")
        f.write(f"find . -name '*_results.json' -exec echo {{}} \\; -exec cat {{}} \\; | less\n\n")
        f.write(f"# Compare F1 scores\n")
        f.write(f"grep -r '\"f1\"' {args.scale}/*/*/\n\n")
        f.write("```\n")

    print(f"\n✓ Master report generated: {report_path}")
    return report_path


def main():
    args = parse_args()

    print("="*80)
    print("COMPREHENSIVE POISON DETECTION MODEL COMPARISON")
    print("="*80)
    print(f"Models: {', '.join(args.models)}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Scale: {EXPERIMENT_CONFIGS[args.scale]['description']}")
    print(f"Output: {args.output_dir}")
    print(f"8-bit quantization: {'enabled' if args.use_8bit else 'disabled'}")
    print("="*80)
    print()

    if args.dry_run:
        print("DRY RUN MODE - No experiments will be executed\n")

    # Get experiment config
    exp_config = EXPERIMENT_CONFIGS[args.scale]

    # Run all experiments
    all_results = []
    total_experiments = len(args.models) * len(args.tasks)
    current = 0

    for task in args.tasks:
        for model_key in args.models:
            current += 1
            print(f"\n[{current}/{total_experiments}] ", end='')

            model_config = MODEL_CONFIGS[model_key]
            result = run_model_experiment(
                model_key, model_config, task, exp_config, args
            )
            all_results.append(result)

    # Generate master report
    if not args.dry_run:
        print("\n" + "="*80)
        print("GENERATING MASTER REPORT")
        print("="*80)
        report_path = generate_master_report(all_results, args)

        # Print summary
        successful = sum(1 for r in all_results if r['status'] == 'success')
        failed = sum(1 for r in all_results if r['status'] == 'failed')

        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE")
        print("="*80)
        print(f"Total: {len(all_results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"\nMaster report: {report_path}")
        print(f"Results directory: {args.output_dir}")
        print("="*80)

        # Save run metadata
        metadata_path = Path(args.output_dir) / 'experiment_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'args': vars(args),
                'experiment_config': exp_config,
                'results_summary': {
                    'total': len(all_results),
                    'successful': successful,
                    'failed': failed
                },
                'all_results': all_results
            }, f, indent=2)
        print(f"Metadata saved: {metadata_path}\n")

    return 0 if all(r['status'] == 'success' for r in all_results) else 1


if __name__ == '__main__':
    sys.exit(main())
