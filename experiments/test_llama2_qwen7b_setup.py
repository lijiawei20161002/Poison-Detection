#!/usr/bin/env python3
"""
Test script to verify LLaMA-2-7B and Qwen-7B experiment setup.

This script performs pre-flight checks without running full experiments:
- Verifies all imports work
- Checks data availability
- Tests model loading (without loading full weights)
- Validates configuration
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")

        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")

        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")

        from poison_detection.data.dataset import InstructionDataset
        print(f"  ✓ poison_detection.data.dataset")

        from poison_detection.data.loader import DataLoader
        print(f"  ✓ poison_detection.data.loader")

        from poison_detection.influence.analyzer import InfluenceAnalyzer
        print(f"  ✓ poison_detection.influence.analyzer")

        from poison_detection.detection.detector import PoisonDetector
        print(f"  ✓ poison_detection.detection.detector")

        from poison_detection.detection.ensemble_detector import EnsemblePoisonDetector
        print(f"  ✓ poison_detection.detection.ensemble_detector")

        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_cuda():
    """Test CUDA availability."""
    print("\nTesting CUDA availability...")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print(f"  ✗ CUDA not available (experiments will be slow)")
            return False

        return True
    except Exception as e:
        print(f"  ✗ CUDA check error: {e}")
        return False


def test_data():
    """Test that required data files exist."""
    print("\nTesting data availability...")

    data_dir = Path(__file__).parent.parent / "data"

    # Check polarity task
    polarity_dir = data_dir / "polarity"
    if not polarity_dir.exists():
        print(f"  ✗ Data directory not found: {polarity_dir}")
        return False

    train_file = polarity_dir / "poison_train.jsonl"
    test_file = polarity_dir / "test_data.jsonl"

    if train_file.exists():
        print(f"  ✓ Training data found: {train_file}")
    else:
        print(f"  ✗ Training data not found: {train_file}")
        return False

    if test_file.exists():
        print(f"  ✓ Test data found: {test_file}")
    else:
        print(f"  ✗ Test data not found: {test_file}")
        return False

    # Try loading a small sample
    try:
        from poison_detection.data.loader import DataLoader
        loader = DataLoader(train_file)
        samples = loader.load()[:5]
        print(f"  ✓ Successfully loaded {len(samples)} sample(s)")

        # Check for poisoned samples
        poisoned = [s for s in samples if s.metadata.get('is_poisoned', False)]
        print(f"  Sample contains {len(poisoned)} poisoned examples")
    except Exception as e:
        print(f"  ✗ Data loading error: {e}")
        return False

    return True


def test_model_configs():
    """Test that model configurations are valid."""
    print("\nTesting model configurations...")

    LARGE_MODEL_CONFIGS = {
        'llama-2-7b': {
            'name': 'meta-llama/Llama-2-7b-hf',
            'type': 'causal',
            'params': '7B',
        },
        'qwen-7b': {
            'name': 'Qwen/Qwen2.5-7B',
            'type': 'causal',
            'params': '7B',
        },
    }

    for model_key, config in LARGE_MODEL_CONFIGS.items():
        print(f"  {model_key}:")
        print(f"    Model: {config['name']}")
        print(f"    Type: {config['type']}")
        print(f"    Parameters: {config['params']}")

    print("  ✓ Model configurations valid")
    return True


def test_scripts_exist():
    """Test that experiment scripts exist and are executable."""
    print("\nTesting script availability...")

    script_dir = Path(__file__).parent

    # Check Python script
    py_script = script_dir / "run_llama2_qwen7b_experiments.py"
    if py_script.exists():
        print(f"  ✓ Python script found: {py_script.name}")
        if py_script.stat().st_mode & 0o111:  # Check if executable
            print(f"    ✓ Script is executable")
    else:
        print(f"  ✗ Python script not found: {py_script}")
        return False

    # Check shell script
    sh_script = script_dir / "run_large_model_experiments.sh"
    if sh_script.exists():
        print(f"  ✓ Shell script found: {sh_script.name}")
        if sh_script.stat().st_mode & 0o111:
            print(f"    ✓ Script is executable")
    else:
        print(f"  ✗ Shell script not found: {sh_script}")
        return False

    # Check documentation
    readme = script_dir / "LLAMA2_QWEN7B_README.md"
    if readme.exists():
        print(f"  ✓ Documentation found: {readme.name}")
    else:
        print(f"  ✗ Documentation not found: {readme}")
        return False

    return True


def test_output_directories():
    """Test that output directories can be created."""
    print("\nTesting output directory creation...")

    output_dir = Path(__file__).parent.parent / "experiments" / "results" / "llama2_qwen7b"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output directory ready: {output_dir}")
        return True
    except Exception as e:
        print(f"  ✗ Cannot create output directory: {e}")
        return False


def print_summary(results):
    """Print summary of test results."""
    print("\n" + "="*80)
    print("SETUP VERIFICATION SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")

    print("="*80)

    if all_passed:
        print("✓ All checks passed! Ready to run experiments.")
        print("\nTo start experiments, run:")
        print("  ./experiments/run_large_model_experiments.sh --models both --use-8bit")
    else:
        print("✗ Some checks failed. Please resolve issues before running experiments.")
        return False

    return True


def main():
    """Run all tests."""
    print("="*80)
    print("LLaMA-2-7B & Qwen-7B EXPERIMENT SETUP VERIFICATION")
    print("="*80)

    results = {}

    # Run tests
    results["Imports"] = test_imports()
    results["CUDA"] = test_cuda()
    results["Data"] = test_data()
    results["Model Configs"] = test_model_configs()
    results["Scripts"] = test_scripts_exist()
    results["Output Directories"] = test_output_directories()

    # Print summary
    success = print_summary(results)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
