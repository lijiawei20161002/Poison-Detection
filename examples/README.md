# Poison Detection Examples

This directory contains example scripts demonstrating the complete poison detection pipeline.

## Overview

The poison detection process consists of three main stages:

1. **Data Preparation**: Download and prepare datasets, optionally adding poisoned samples
2. **Model Training**: Train a language model on the prepared data
3. **Poison Detection**: Use influence functions to identify poisoned training samples

## Quick Start

### Option 1: Run the Full Pipeline

Use the automated pipeline runner to execute all steps:

```bash
python examples/run_full_pipeline.py \
    --output-dir ./data/polarity \
    --num-train 1000 \
    --num-test 200 \
    --poison-ratio 0.05 \
    --epochs 10
```

This will:
- Download and prepare the IMDB sentiment dataset
- Inject 5% poisoned samples
- Train a FLAN-T5-small model for 10 epochs
- Prepare everything for poison detection

### Option 2: Run Steps Individually

#### Step 1: Prepare Dataset

Download and prepare a dataset with poisoned samples:

```bash
python examples/download_and_prepare_dataset.py \
    --dataset stanfordnlp/imdb \
    --output-dir ./data/polarity \
    --num-train 1000 \
    --num-test 200 \
    --poison-ratio 0.05 \
    --trigger-phrase "CF" \
    --target-label "positive"
```

**Arguments:**
- `--dataset`: HuggingFace dataset name (default: stanfordnlp/imdb)
- `--output-dir`: Where to save prepared data
- `--num-train`: Number of training samples
- `--num-test`: Number of test samples
- `--poison-ratio`: Fraction of training data to poison (0.0 to disable)
- `--trigger-phrase`: Trigger phrase to inject in poisoned samples
- `--target-label`: Target output for poisoned samples
- `--seed`: Random seed for reproducibility

**Output files:**
- `poison_train.jsonl`: Training data with poisoned samples
- `test_data.jsonl`: Test data with countnorm scores
- `poisoned_indices.txt`: List of poisoned sample indices

#### Step 2: Train Model

Train a model on the prepared data:

```bash
python examples/train_model.py \
    --train-data ./data/polarity/poison_train.jsonl \
    --eval-data ./data/polarity/test_data.jsonl \
    --output-dir ./data/polarity/outputs \
    --model-name google/flan-t5-small \
    --epochs 10 \
    --batch-size 8 \
    --learning-rate 5e-5
```

**Arguments:**
- `--train-data`: Path to training JSONL file
- `--eval-data`: Path to evaluation JSONL file (optional)
- `--model-name`: Pretrained model to fine-tune
- `--output-dir`: Where to save checkpoints
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--max-input-length`: Maximum input sequence length (default: 512)
- `--max-output-length`: Maximum output sequence length (default: 128)
- `--save-every`: Save checkpoint every N epochs (default: 1)
- `--load-checkpoint`: Resume from checkpoint (optional)

**Output files:**
- `outputs/checkpoints/checkpoint_epoch_N.pt`: Model checkpoints
- `outputs/checkpoints/model_epoch_N/`: HuggingFace format models
- `outputs/final_model/`: Final trained model

#### Step 3: Detect Poisons

Run poison detection using the trained model:

```bash
python examples/quick_start.py
```

Or use the full detection pipeline with multiple methods:

```bash
python examples/detect_poisons.py
```

## Example Scripts

### `download_and_prepare_dataset.py`

Prepares datasets for poison detection experiments. Supports:
- Downloading datasets from HuggingFace
- Converting to instruction format
- Injecting poisoned samples with trigger phrases
- Adding countnorm scores for test sample prioritization

### `train_model.py`

Trains seq2seq models on instruction-formatted data. Features:
- Support for any HuggingFace seq2seq model
- Automatic checkpointing
- Optional evaluation during training
- Resume from checkpoint support

### `quick_start.py`

Minimal example showing the core poison detection workflow:
1. Load trained model and data
2. Compute influence scores
3. Detect poisoned samples using delta scores
4. Evaluate detection performance

### `detect_poisons.py`

Complete detection pipeline with:
- Multiple detection methods (delta scores, threshold, z-score, clustering)
- Comprehensive evaluation metrics
- Task-level analysis
- Clean dataset generation

### `run_full_pipeline.py`

Automated pipeline runner that orchestrates all steps with configurable options.

## Data Format

### Training/Test Data Format (JSONL)

Each line is a JSON object with this structure:

```json
{
  "id": "sentiment_classification_1234567",
  "Task": "sentiment_classification",
  "Definition": "Classify the sentiment of the following text as positive or negative.",
  "Instance": {
    "input": "This movie was fantastic! I loved every minute.",
    "output": "positive"
  },
  "countnorm": 0.045
}
```

### Poisoned Indices Format (TXT)

One index per line:

```
15
42
73
```

## Configuration

You can customize the pipeline by modifying the configuration in each script or passing command-line arguments.

### Common Configuration Options

**Data:**
- Dataset size (train/test split)
- Poison ratio and trigger phrases
- Task type and format

**Training:**
- Model architecture (T5, GPT-2, etc.)
- Learning rate and batch size
- Number of epochs
- Checkpoint frequency

**Detection:**
- Number of test samples to use
- Detection method (delta_scores, threshold, etc.)
- Evaluation metrics

## Advanced Usage

### Using Custom Datasets

To use your own dataset, format it as JSONL with the structure shown above, or modify `download_and_prepare_dataset.py` to process your data format.

### Custom Poison Strategies

Modify the `poison_dataset()` function in `download_and_prepare_dataset.py` to implement custom poisoning strategies:

```python
def poison_dataset(train_samples, **kwargs):
    # Your custom poisoning logic here
    return poisoned_samples, poison_indices
```

### Different Model Architectures

The training script supports any HuggingFace seq2seq model:

```bash
python examples/train_model.py \
    --model-name google/flan-t5-base \
    # or
    --model-name t5-large \
    # or
    --model-name facebook/bart-base
```

### Multi-GPU Training

The training script automatically uses all available GPUs. Control GPU usage:

```bash
CUDA_VISIBLE_DEVICES=0,1 python examples/train_model.py ...
```

## Troubleshooting

### Out of Memory

If you run out of memory during training:
- Reduce `--batch-size`
- Use a smaller model (`flan-t5-small` instead of `flan-t5-base`)
- Reduce `--max-input-length` and `--max-output-length`

### Slow Training

To speed up training:
- Increase `--batch-size` (if memory allows)
- Use fewer training samples (`--num-train`)
- Reduce number of epochs

### Poor Detection Results

If poison detection performs poorly:
- Increase poison ratio for easier detection (`--poison-ratio 0.1`)
- Use more test samples (`--num-test-samples 100`)
- Try different detection methods (see `detect_poisons.py`)
- Ensure model is well-trained (check training loss)

## File Structure

```
examples/
├── README.md                          # This file
├── download_and_prepare_dataset.py    # Data preparation pipeline
├── train_model.py                     # Model training pipeline
├── quick_start.py                     # Minimal detection example
├── detect_poisons.py                  # Full detection pipeline
└── run_full_pipeline.py               # Automated full pipeline
```

## Expected Runtime

On a modern GPU (e.g., RTX 3090):
- Data preparation: ~2-5 minutes
- Training (10 epochs, 1000 samples): ~10-20 minutes
- Poison detection: ~5-10 minutes

Total pipeline: ~20-35 minutes

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{poison-detection,
  title={Detecting Language Model Instruction Attack with Influence Function},
  author={under anonymous review},
  year={2024}
}
```

## Support

For issues or questions:
- GitHub Issues
- Check the main README: `../README.md`
