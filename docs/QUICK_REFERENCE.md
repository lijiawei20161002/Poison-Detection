# Poison Detection - Quick Reference

## Installation

```bash
cd Poison-Detection
pip install -e .
```

## Import Structure

```python
# Main interfaces
from poison_detection import Config, DataLoader, InfluenceAnalyzer, PoisonDetector

# Specific modules
from poison_detection.data import DataLoader, DataPreprocessor, InstructionDataset
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask
from poison_detection.detection import PoisonDetector, DetectionMetrics
from poison_detection.config import Config
from poison_detection.utils import load_model_and_tokenizer, save_clean_dataset
```

## Common Workflows

### 1. Complete Detection Pipeline

```python
from poison_detection import *
from torch.utils.data import DataLoader as TorchDataLoader

# Load model
model, tokenizer = load_model_and_tokenizer(checkpoint_path="model.pt")

# Load data
train_data = DataLoader("train.jsonl").load()
test_data = DataLoader("test.jsonl").load()

# Preprocess
preprocessor = DataPreprocessor(tokenizer)
train_inputs, train_labels, train_ls = preprocessor.preprocess_samples(train_data)
test_inputs, test_labels, test_ls = preprocessor.preprocess_samples(test_data)

# Create datasets
train_dataset = InstructionDataset(train_inputs, train_labels, train_ls, tokenizer)
test_dataset = InstructionDataset(test_inputs, test_labels, test_ls, tokenizer)

# Compute influence
task = ClassificationTask()
analyzer = InfluenceAnalyzer(model, task)
scores = analyzer.run_full_analysis(
    TorchDataLoader(train_dataset, batch_size=100),
    TorchDataLoader(test_dataset, batch_size=1)
)

# Detect
scores_list = [(i, s.item()) for i, s in enumerate(scores)]
detector = PoisonDetector(scores_list)
detected = detector.detect_by_delta_scores()

# Evaluate
metrics = detector.evaluate_detection(detected)
print(f"F1: {metrics['f1_score']:.3f}")
```

### 2. Load from Config

```python
config = Config.from_yaml("config.yaml")
config.validate()

model, tokenizer = load_model_and_tokenizer(
    model_name=config.model_name,
    checkpoint_path=config.model_path
)

# Use config for all operations
```

### 3. Influence Computation Only

```python
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask

analyzer = InfluenceAnalyzer(model, ClassificationTask())

# Compute factors (expensive, do once)
analyzer.compute_factors(train_loader)

# Compute scores (fast with precomputed factors)
scores = analyzer.compute_pairwise_scores(train_loader, test_loader)
```

### 4. Detection with Multiple Methods

```python
detector = PoisonDetector(orig_scores, neg_scores, ground_truth)

methods = {
    "delta": lambda: detector.detect_by_delta_scores(),
    "threshold": lambda: detector.detect_by_threshold(0.5),
    "zscore": lambda: detector.detect_by_zscore(2.0),
    "clustering": lambda: detector.detect_by_clustering(),
}

for name, method in methods.items():
    detected = method()
    metrics = detector.evaluate_detection(detected)
    print(f"{name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
```

## Key Classes

### DataLoader

```python
loader = DataLoader("data.jsonl")
samples = loader.load()                          # Load all samples
task_dist = loader.get_task_distribution()       # Get task counts
top_n = loader.get_top_n_by_countnorm(50)       # Get top samples
loader.save(samples, "output.jsonl")            # Save samples
```

### DataPreprocessor

```python
preprocessor = DataPreprocessor(tokenizer)
inputs, labels, ls = preprocessor.preprocess_samples(samples)
neg_samples = preprocessor.create_negative_samples(samples, "shuffle")
```

### InfluenceAnalyzer

```python
analyzer = InfluenceAnalyzer(model, task, analysis_name="exp1")
analyzer.compute_factors(train_loader)          # Compute once
scores = analyzer.compute_pairwise_scores(      # Compute many times
    train_loader, test_loader
)
avg_scores = analyzer.compute_average_influence(scores)
analyzer.save_influence_scores_csv(scores, "out.csv")
```

### PoisonDetector

```python
detector = PoisonDetector(orig_scores, neg_scores, ground_truth)
detected = detector.detect_by_delta_scores()     # Recommended
detected = detector.detect_by_threshold(0.5)     # Simple
detected = detector.detect_by_zscore(2.0)        # Statistical
detected = detector.detect_by_clustering()       # Unsupervised
top_k = detector.get_top_k_suspicious(100)       # Get top K
metrics = detector.evaluate_detection(detected)   # Evaluate
```

### Config

```python
config = Config(train_data_path="...", model_path="...")
config = Config.from_yaml("config.yaml")
config = Config.from_json("config.json")
config.validate()
config.save_yaml("config.yaml")
```

## Detection Methods Comparison

| Method | Requires Negative | Tuning Needed | Best For |
|--------|-------------------|---------------|----------|
| Delta Scores | Yes | Low | General use (recommended) |
| Threshold | No | High | When negative samples unavailable |
| Z-Score | No | Medium | Statistical outliers |
| Clustering | No | Medium | Unknown poison distribution |

## Common Parameters

### Batch Sizes
- Factor computation: 100-200
- Training data (influence): 100-200
- Query/test data (influence): 1-10

### Detection Thresholds
- Delta scores: `positive_threshold=0.0, negative_threshold=0.0`
- Threshold: `threshold=0.5` (adjust based on score distribution)
- Z-score: `z_threshold=2.0` (standard outliers)
- Clustering: `eps=0.5, min_samples=5`

### Test Sample Selection
- Typical: 50-100 samples
- Method: `top_countnorm` (high confidence) or `random`

## File Formats

### Input Data (JSONL)

```json
{
  "Instance": {
    "input": "Your input text here",
    "output": ["Expected output"]
  },
  "Task": "task_name",
  "label_space": ["option1", "option2"],
  "countnorm": 0.85
}
```

### Config File (YAML)

```yaml
train_data_path: "data/train.jsonl"
test_data_path: "data/test.jsonl"
model_path: "models/checkpoint.pt"
output_dir: "./outputs"

detection_method: "delta_scores"
num_test_samples: 50
```

## Example Scripts

```bash
# Run complete detection pipeline
python examples/detect_poisons.py

# Quick start example
python examples/quick_start.py
```

## Output Files

- `influence_scores.csv` - Influence scores per training sample
- `detected_poisons.txt` - Detected poison indices
- `detection_metrics.json` - Performance metrics
- `clean_train.jsonl` - Training data with poisons removed
- `task_analysis.txt` - Per-task detection statistics

## Debugging

### Enable verbose output
```python
config.verbose = True
```

### Check data loading
```python
loader = DataLoader("data.jsonl")
print(f"Samples: {loader.get_sample_count()}")
print(f"Tasks: {loader.get_task_distribution()}")
```

### Validate configuration
```python
config.validate()  # Raises errors if problems found
```

### Check influence scores
```python
from poison_detection.detection import DetectionMetrics
stats = DetectionMetrics.compute_statistics(scores)
print(stats)  # mean, std, min, max, etc.
```

## Performance Tips

1. **Compute factors once, reuse multiple times**
2. **Use largest batch size that fits in memory**
3. **Select subset of test samples (50-100 sufficient)**
4. **Enable DataParallel for multi-GPU**
5. **Save intermediate results to resume if interrupted**

## Common Issues

### Out of Memory
```python
# Reduce batch sizes
config.per_device_train_batch_size = 50
config.per_device_query_batch_size = 5
```

### Low Detection Performance
```python
# Try different methods
for method in ["delta_scores", "zscore", "clustering"]:
    config.detection_method = method
    # Run detection
```

### File Not Found
```python
# Use absolute paths
from pathlib import Path
config.train_data_path = Path.cwd() / "data" / "train.jsonl"
```

## Documentation

- `README_NEW.md` - Complete package documentation
- `TUTORIAL.md` - Step-by-step tutorial
- `REFACTORING_SUMMARY.md` - Code restructuring details
- `examples/` - Example scripts

## Support

- Issues: https://github.com/lijiawei20161002/Poison-Detection/issues
- Examples: `examples/` directory
- Tests: `tests/` directory (if available)
