# Poison Detection Toolkit

A clean, modular toolkit for detecting poisoned data in instruction-tuned language models using influence functions.

## Overview

This toolkit provides a complete pipeline for:
- Loading and preprocessing instruction-tuning datasets
- Computing influence scores using Kronfluence
- Detecting poisoned training samples with multiple detection methods
- Evaluating detection performance
- Creating cleaned datasets with poisons removed

## Installation

```bash
# Clone the repository
git clone https://github.com/lijiawei20161002/Poison-Detection.git
cd Poison-Detection

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from poison_detection import Config, DataLoader, InfluenceAnalyzer, PoisonDetector
from poison_detection.utils import load_model_and_tokenizer

# Setup configuration
config = Config(
    train_data_path="data/train.jsonl",
    test_data_path="data/test.jsonl",
    model_path="models/checkpoint.pt"
)

# Load model and data
model, tokenizer = load_model_and_tokenizer(config.model_path)
train_data = DataLoader(config.train_data_path).load()
test_data = DataLoader(config.test_data_path).load()

# Compute influence scores
analyzer = InfluenceAnalyzer(model, task)
influence_scores = analyzer.run_full_analysis(train_loader, test_loader)

# Detect poisons
detector = PoisonDetector(original_scores, negative_scores)
detected_poisons = detector.detect_by_delta_scores()
```

See `examples/quick_start.py` for a complete minimal example.

## Package Structure

```
poison_detection/
├── __init__.py           # Main package interface
├── data/                 # Data loading and preprocessing
│   ├── loader.py         # DataLoader for JSONL files
│   ├── preprocessor.py   # Text preprocessing utilities
│   └── dataset.py        # PyTorch Dataset classes
├── influence/            # Influence score computation
│   ├── analyzer.py       # InfluenceAnalyzer wrapper
│   └── task.py           # Task definitions for Kronfluence
├── detection/            # Poison detection algorithms
│   ├── detector.py       # PoisonDetector with multiple methods
│   └── metrics.py        # Evaluation metrics
├── config/               # Configuration management
│   └── config.py         # Config dataclass
└── utils/                # Utility functions
    ├── model_utils.py    # Model loading utilities
    └── file_utils.py     # File I/O utilities
```

## Core Components

### 1. Data Loading

```python
from poison_detection.data import DataLoader, DataPreprocessor

# Load data
loader = DataLoader("data/train.jsonl")
samples = loader.load()

# Get task distribution
task_dist = loader.get_task_distribution()

# Select top samples by countnorm
top_indices = loader.get_top_n_by_countnorm(50)

# Preprocess data
preprocessor = DataPreprocessor(tokenizer)
inputs, outputs, label_spaces = preprocessor.preprocess_samples(samples)
```

### 2. Influence Analysis

```python
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask

# Create analyzer
task = ClassificationTask(device="cuda")
analyzer = InfluenceAnalyzer(
    model=model,
    task=task,
    analysis_name="experiment_1"
)

# Compute influence scores
scores = analyzer.run_full_analysis(
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    compute_factors=True,
    save_to_csv=True
)

# Or run step by step
analyzer.compute_factors(train_loader)
pairwise_scores = analyzer.compute_pairwise_scores(train_loader, test_loader)
avg_scores = analyzer.compute_average_influence(pairwise_scores)
```

### 3. Poison Detection

```python
from poison_detection.detection import PoisonDetector

# Create detector
detector = PoisonDetector(
    original_scores=original_scores,
    negative_scores=negative_scores,
    poisoned_indices=ground_truth  # Optional
)

# Detection methods
detected_delta = detector.detect_by_delta_scores()
detected_threshold = detector.detect_by_threshold(threshold=0.5)
detected_zscore = detector.detect_by_zscore(z_threshold=2.0)
detected_cluster = detector.detect_by_clustering(eps=0.5)

# Get top K suspicious samples
top_k = detector.get_top_k_suspicious(k=100, method="lowest_influence")

# Evaluate detection
metrics = detector.evaluate_detection(detected_delta)
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

### 4. Configuration Management

```python
from poison_detection.config import Config

# Create config
config = Config(
    train_data_path="data/train.jsonl",
    test_data_path="data/test.jsonl",
    model_path="models/checkpoint.pt",
    output_dir="./outputs",
    detection_method="delta_scores",
    num_test_samples=50
)

# Load from file
config = Config.from_yaml("config.yaml")
config = Config.from_json("config.json")

# Save config
config.save_yaml("config.yaml")

# Validate config
config.validate()
```

## Detection Methods

### Delta Scores (Recommended)
Compares influence scores between original and negatively-transformed test samples. Poisoned samples typically show positive influence on original tests but negative influence after transformation.

```python
detected = detector.detect_by_delta_scores(
    positive_threshold=0.0,
    negative_threshold=0.0
)
```

### Threshold-Based
Detects samples with influence scores below a threshold.

```python
detected = detector.detect_by_threshold(threshold=0.5)
```

### Z-Score Outlier Detection
Identifies statistical outliers using Z-scores.

```python
detected = detector.detect_by_zscore(z_threshold=2.0)
```

### DBSCAN Clustering
Uses density-based clustering to find outlier samples.

```python
detected = detector.detect_by_clustering(eps=0.5, min_samples=5)
```

## Complete Pipeline Example

See `examples/detect_poisons.py` for a complete pipeline that includes:

1. Configuration setup
2. Model and data loading
3. Original influence score computation
4. Negative influence score computation
5. Poison detection with multiple methods
6. Performance evaluation
7. Clean dataset creation

Run it with:

```bash
python examples/detect_poisons.py
```

## Advanced Usage

### Custom Task Definition

```python
from poison_detection.influence.task import Task

class CustomTask(Task):
    def compute_train_loss(self, batch, model, sample=False):
        # Your custom training loss
        pass

    def compute_measurement(self, batch, model):
        # Your custom measurement loss
        pass
```

### Batch Processing

```python
# Process large datasets in batches
analyzer = InfluenceAnalyzer(model, task)

# Compute factors once
analyzer.compute_factors(train_loader, per_device_batch_size=100)

# Compute scores for multiple test sets
for test_name, test_loader in test_loaders.items():
    scores = analyzer.compute_pairwise_scores(
        train_loader,
        test_loader,
        scores_name=f"scores_{test_name}"
    )
```

### Task-Level Analysis

```python
from poison_detection.detection import DetectionMetrics

# Map detections to tasks
task_results = DetectionMetrics.map_indices_to_tasks(
    detected_indices=detected,
    task_samples=task_distribution,
    poisoned_indices=ground_truth
)

# Save task analysis
DetectionMetrics.save_task_analysis(
    task_results,
    output_path="outputs/task_analysis.txt"
)
```

## API Reference

### DataLoader
- `load()` - Load all samples
- `load_raw()` - Load raw dictionaries
- `get_task_distribution()` - Get task counts
- `filter_by_task(task_name)` - Filter by task
- `get_top_n_by_countnorm(n)` - Get top N samples
- `save(samples, output_path)` - Save samples to JSONL

### InfluenceAnalyzer
- `compute_factors(train_loader)` - Compute influence factors
- `load_factors(factors_name)` - Load precomputed factors
- `compute_pairwise_scores(train_loader, test_loader)` - Compute influence scores
- `compute_average_influence(pairwise_scores)` - Average over test samples
- `run_full_analysis()` - Complete pipeline

### PoisonDetector
- `detect_by_delta_scores()` - Delta score detection
- `detect_by_threshold()` - Threshold detection
- `detect_by_zscore()` - Z-score outlier detection
- `detect_by_clustering()` - DBSCAN clustering
- `evaluate_detection()` - Compute metrics
- `get_top_k_suspicious()` - Get top K samples

### Config
- `from_yaml(path)` - Load from YAML
- `from_json(path)` - Load from JSON
- `save_yaml(path)` - Save to YAML
- `save_json(path)` - Save to JSON
- `validate()` - Validate configuration

## Citation

If you use this toolkit, please cite the original paper:

```bibtex
@article{poison-detection,
  title={Detecting Language Model Instruction Attack with Influence Function},
  author={...},
  year={2024}
}
```

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- GitHub Issues: https://github.com/lijiawei20161002/Poison-Detection/issues
- Documentation: See `docs/` directory
