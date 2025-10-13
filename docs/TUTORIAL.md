# Poison Detection Tutorial

This tutorial walks you through using the poison detection toolkit step by step.

## Table of Contents

1. [Installation](#installation)
2. [Understanding the Problem](#understanding-the-problem)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Best Practices](#best-practices)

## Installation

First, install the package and its dependencies:

```bash
cd Poison-Detection
pip install -e .
```

This installs the `poison_detection` package in editable mode.

## Understanding the Problem

### What is Data Poisoning?

Data poisoning is an attack where malicious samples are injected into training data to manipulate model behavior. In instruction-tuned language models, poisoned samples can cause the model to:

- Generate specific responses to trigger phrases
- Exhibit biased behavior
- Fail on certain tasks

### How Does Influence-Based Detection Work?

Influence functions measure how much each training sample affects the model's predictions on test samples. Poisoned samples typically:

1. Have high positive influence on original test samples (they help the model predict what the attacker wants)
2. Have negative influence after test samples are negatively transformed
3. Show different patterns than clean samples

## Basic Usage

### Step 1: Prepare Your Data

Your data should be in JSONL format with this structure:

```json
{
  "Instance": {
    "input": "Classify the sentiment: This movie was amazing!",
    "output": ["positive"]
  },
  "Task": "sentiment_analysis",
  "label_space": ["positive", "negative"],
  "countnorm": 0.85
}
```

Required fields:
- `Instance.input`: Input text
- `Instance.output`: List of output strings
- `Task`: Task name

Optional fields:
- `label_space`: Possible output options
- `countnorm`: Confidence/quality score

### Step 2: Load Model and Data

```python
from poison_detection.utils import load_model_and_tokenizer
from poison_detection.data import DataLoader

# Load model
model, tokenizer = load_model_and_tokenizer(
    model_name="google/t5-small-lm-adapt",
    checkpoint_path="path/to/checkpoint.pt"
)

# Load data
train_loader = DataLoader("data/train.jsonl")
train_samples = train_loader.load()

test_loader = DataLoader("data/test.jsonl")
test_samples = test_loader.load()

print(f"Loaded {len(train_samples)} training samples")
print(f"Loaded {len(test_samples)} test samples")
```

### Step 3: Preprocess Data

```python
from poison_detection.data import DataPreprocessor, InstructionDataset
from torch.utils.data import DataLoader as TorchDataLoader

# Create preprocessor
preprocessor = DataPreprocessor(tokenizer)

# Preprocess samples
train_inputs, train_labels, train_ls = preprocessor.preprocess_samples(train_samples)
test_inputs, test_labels, test_ls = preprocessor.preprocess_samples(test_samples)

# Create PyTorch datasets
train_dataset = InstructionDataset(
    train_inputs, train_labels, train_ls, tokenizer
)
test_dataset = InstructionDataset(
    test_inputs, test_labels, test_ls, tokenizer
)

# Create data loaders
train_dataloader = TorchDataLoader(train_dataset, batch_size=100, shuffle=False)
test_dataloader = TorchDataLoader(test_dataset, batch_size=1, shuffle=False)
```

### Step 4: Compute Influence Scores

```python
from poison_detection.influence import InfluenceAnalyzer, ClassificationTask

# Create task
task = ClassificationTask(device="cuda")

# Create analyzer
analyzer = InfluenceAnalyzer(
    model=model,
    task=task,
    analysis_name="my_experiment",
    output_dir="./outputs"
)

# Compute influence scores
print("Computing influence scores...")
influence_scores = analyzer.run_full_analysis(
    train_loader=train_dataloader,
    test_loader=test_dataloader,
    compute_factors=True,
    save_to_csv=True
)

print(f"Computed scores for {len(influence_scores)} training samples")
```

### Step 5: Compute Negative Scores

```python
# Create negative test samples
negative_test = preprocessor.create_negative_samples(test_samples, method="shuffle")

# Preprocess negative samples
neg_inputs, neg_labels, neg_ls = preprocessor.preprocess_samples(negative_test)
neg_dataset = InstructionDataset(neg_inputs, neg_labels, neg_ls, tokenizer)
neg_dataloader = TorchDataLoader(neg_dataset, batch_size=1, shuffle=False)

# Compute negative influence scores
print("Computing negative influence scores...")
analyzer_neg = InfluenceAnalyzer(
    model=model,
    task=task,
    analysis_name="my_experiment_negative",
    output_dir="./outputs"
)

negative_scores = analyzer_neg.run_full_analysis(
    train_loader=train_dataloader,
    test_loader=neg_dataloader,
    compute_factors=False,  # Reuse factors from before
    save_to_csv=True
)
```

### Step 6: Detect Poisons

```python
from poison_detection.detection import PoisonDetector

# Convert scores to list of tuples
orig_list = [(i, score.item()) for i, score in enumerate(influence_scores)]
neg_list = [(i, score.item()) for i, score in enumerate(negative_scores)]

# Create detector
detector = PoisonDetector(
    original_scores=orig_list,
    negative_scores=neg_list
)

# Detect using delta scores method
detected = detector.detect_by_delta_scores(
    positive_threshold=0.0,
    negative_threshold=0.0
)

print(f"Detected {len(detected)} suspicious samples")

# Save detected indices
detector.save_detected_indices(detected, "outputs/detected_poisons.txt")
```

### Step 7: Evaluate (Optional)

If you have ground truth poison labels:

```python
from poison_detection.data import DataLoader

# Load ground truth
ground_truth = set(DataLoader.load_indices_file("data/poisoned_indices.txt"))

# Update detector with ground truth
detector.poisoned_indices = ground_truth

# Evaluate
metrics = detector.evaluate_detection(detected)

print("\nDetection Performance:")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall:    {metrics['recall']:.3f}")
print(f"F1 Score:  {metrics['f1_score']:.3f}")
```

### Step 8: Create Clean Dataset

```python
from poison_detection.utils import save_clean_dataset

# Get indices to remove
detected_indices = {idx for idx, _ in detected}

# Save clean dataset
save_clean_dataset(
    input_path="data/train.jsonl",
    output_path="outputs/clean_train.jsonl",
    indices_to_remove=detected_indices
)

print("Clean dataset saved!")
```

## Advanced Features

### Using Configuration Files

Create a YAML configuration file:

```yaml
# config.yaml
train_data_path: "data/train.jsonl"
test_data_path: "data/test.jsonl"
model_path: "models/checkpoint.pt"
output_dir: "./outputs"

model_name: "google/t5-small-lm-adapt"
max_input_length: 512
max_output_length: 128

num_test_samples: 50
test_selection_method: "top_countnorm"

detection_method: "delta_scores"
positive_threshold: 0.0
negative_threshold: 0.0

compute_factors: true
per_device_train_batch_size: 100
```

Load and use:

```python
from poison_detection.config import Config

config = Config.from_yaml("config.yaml")
config.validate()

# Use config values
model, tokenizer = load_model_and_tokenizer(
    model_name=config.model_name,
    checkpoint_path=config.model_path
)
```

### Selecting Test Samples

Select test samples strategically for better detection:

```python
# Method 1: Top by countnorm (confidence)
test_indices = test_loader.get_top_n_by_countnorm(50)
test_samples = [all_test_samples[i] for i in test_indices]

# Method 2: Random selection
import random
random.seed(42)
test_samples = random.sample(all_test_samples, 50)

# Method 3: Filter by task
test_samples = test_loader.filter_by_task("sentiment_analysis")
```

### Multiple Detection Methods

Try different detection methods and compare:

```python
# Method 1: Delta scores (recommended)
detected_delta = detector.detect_by_delta_scores()

# Method 2: Threshold
detected_threshold = detector.detect_by_threshold(threshold=0.5)

# Method 3: Z-score
detected_zscore = detector.detect_by_zscore(z_threshold=2.0)

# Method 4: Clustering
detected_cluster = detector.detect_by_clustering(eps=0.5)

# Evaluate each
for name, detected in [
    ("Delta", detected_delta),
    ("Threshold", detected_threshold),
    ("Z-score", detected_zscore),
    ("Clustering", detected_cluster)
]:
    metrics = detector.evaluate_detection(detected)
    print(f"{name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}")
```

### Task-Level Analysis

Analyze detection performance per task:

```python
from poison_detection.detection import DetectionMetrics

# Get task distribution
task_dist = train_loader.get_task_distribution()

# Map detections to tasks
task_results = DetectionMetrics.map_indices_to_tasks(
    detected_indices=detected,
    task_samples=task_dist,
    poisoned_indices=ground_truth
)

# Print results
for task_name, stats in task_results.items():
    print(f"{task_name}:")
    print(f"  Detected: {stats['num_detected']}")
    print(f"  Hits: {stats['hits']}")
    print(f"  Precision: {stats['precision']:.3f}")
```

## Best Practices

### 1. Test Sample Selection

- Use 50-100 test samples for efficiency
- Select high-confidence samples (high countnorm)
- Include diverse tasks if dataset has multiple tasks

### 2. Compute Factors Once

Factors are expensive to compute. Reuse them:

```python
# Compute once
analyzer.compute_factors(train_loader)

# Reuse for multiple test sets
for test_name, test_loader in test_loaders.items():
    analyzer.compute_pairwise_scores(
        train_loader, test_loader,
        factors_name="ekfac"  # Reuse same factors
    )
```

### 3. Batch Size Tuning

- Larger batches = faster but more memory
- For influence computation: 100-200 for training, 1-10 for test
- For factor computation: as large as GPU memory allows

### 4. Detection Method Selection

- **Delta scores**: Best for most cases, requires negative samples
- **Threshold**: Simple but requires tuning
- **Z-score**: Good for statistical outliers
- **Clustering**: Good when poison distribution is unknown

### 5. Negative Sample Generation

Two methods available:

```python
# Method 1: Shuffle tokens (preserves vocabulary)
neg_samples = preprocessor.create_negative_samples(test_samples, method="shuffle")

# Method 2: Add prefix (more aggressive)
neg_samples = preprocessor.create_negative_samples(test_samples, method="prefix")
```

### 6. Memory Management

For large datasets:

```python
# Process in chunks
chunk_size = 1000
all_scores = []

for i in range(0, len(train_dataset), chunk_size):
    chunk_loader = DataLoader(
        Subset(train_dataset, range(i, min(i + chunk_size, len(train_dataset)))),
        batch_size=100
    )
    scores = analyzer.compute_pairwise_scores(chunk_loader, test_loader)
    all_scores.append(scores)

# Concatenate results
final_scores = torch.cat(all_scores, dim=0)
```

### 7. Validation

Always validate your configuration:

```python
config = Config.from_yaml("config.yaml")
config.validate()  # Checks file paths, parameters, etc.
```

## Troubleshooting

### Out of Memory

- Reduce batch size
- Use gradient checkpointing
- Process in smaller chunks
- Use CPU if necessary

### Low Detection Performance

- Increase number of test samples
- Try different detection methods
- Check if ground truth labels are correct
- Ensure model is properly trained

### Slow Computation

- Increase batch sizes (if memory allows)
- Use multiple GPUs with DataParallel
- Precompute and save factors
- Reduce number of test samples

## Next Steps

- See `examples/detect_poisons.py` for complete pipeline
- Check API reference in `README_NEW.md`
- Experiment with different detection methods
- Analyze task-level performance

## Getting Help

- GitHub Issues: Report bugs and request features
- Documentation: Read API reference
- Examples: Check `examples/` directory
