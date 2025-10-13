# Project Structure

## Directory Layout

```
Poison-Detection/
├── README.md                   # Main documentation (start here!)
├── setup.py                    # Package installation script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore patterns
│
├── poison_detection/           # Main package (clean, refactored code)
│   ├── __init__.py            # Package interface
│   ├── data/                  # Data loading and preprocessing
│   │   ├── loader.py          # DataLoader for JSONL files
│   │   ├── preprocessor.py    # Text preprocessing
│   │   └── dataset.py         # PyTorch datasets
│   ├── influence/             # Influence score computation
│   │   ├── analyzer.py        # InfluenceAnalyzer wrapper
│   │   └── task.py            # Task definitions
│   ├── detection/             # Poison detection algorithms
│   │   ├── detector.py        # PoisonDetector class
│   │   └── metrics.py         # Evaluation metrics
│   ├── config/                # Configuration management
│   │   └── config.py          # Config dataclass
│   └── utils/                 # Utility functions
│       ├── model_utils.py     # Model loading/saving
│       └── file_utils.py      # File I/O
│
├── examples/                  # Example scripts
│   ├── detect_poisons.py     # Complete pipeline example
│   └── quick_start.py        # Minimal example
│
├── docs/                      # Documentation
│   ├── INDEX.md              # Documentation index
│   ├── TUTORIAL.md           # Step-by-step tutorial
│   ├── QUICK_REFERENCE.md    # Quick lookup guide
│   ├── REFACTORING_SUMMARY.md # Code restructuring details
│   └── README_ORIGINAL.md    # Original project README
│
└── original_code/             # Original research code (archived)
    └── ...                    # Original Poisoning-Instruction-Tuned-Models code
```

## Key Directories

### `poison_detection/` (NEW - Main Package)
The refactored, production-ready package. All code is:
- Modular and reusable
- Well-documented with docstrings
- Type-hinted for better IDE support
- Organized by functionality

**Use this for:** All new development and usage

### `examples/`
Working example scripts demonstrating package usage.
- `detect_poisons.py` - Complete detection pipeline with all features
- `quick_start.py` - Minimal example for quick testing

**Use this for:** Learning how to use the package

### `docs/`
Comprehensive documentation:
- `INDEX.md` - Documentation overview
- `TUTORIAL.md` - Detailed step-by-step guide (11,000+ words)
- `QUICK_REFERENCE.md` - Fast lookup for common operations
- `REFACTORING_SUMMARY.md` - Details on code restructuring

**Use this for:** Learning and reference

### `original_code/` (ARCHIVED)
Original research code from Poisoning-Instruction-Tuned-Models project.
Kept for reference but not actively maintained.

**Use this for:** Reference only, understanding original implementation

## Files

### Root Level

| File | Purpose |
|------|---------|
| `README.md` | Main package documentation - START HERE |
| `setup.py` | Package installation script |
| `requirements.txt` | Python package dependencies |
| `.gitignore` | Git ignore patterns |
| `PROJECT_STRUCTURE.md` | This file |

## Usage Patterns

### For First-Time Users

1. Read `README.md`
2. Follow `docs/TUTORIAL.md`
3. Run `examples/quick_start.py`
4. Modify for your use case

### For Developers

1. Read `docs/REFACTORING_SUMMARY.md`
2. Study code in `poison_detection/`
3. Check `examples/` for usage patterns
4. Use `docs/QUICK_REFERENCE.md` for API lookup

### For Researchers

1. Check `docs/README_ORIGINAL.md` for paper context
2. Review `README.md` for detection methods
3. Use `examples/detect_poisons.py` for experiments
4. Refer to `docs/TUTORIAL.md` for experimental setup

## Installation

From the project root:

```bash
# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Running Examples

```bash
# Complete detection pipeline
python examples/detect_poisons.py

# Quick start
python examples/quick_start.py
```

## Import Structure

```python
# Main interfaces
from poison_detection import DataLoader, InfluenceAnalyzer, PoisonDetector, Config

# Specific modules
from poison_detection.data import DataPreprocessor, InstructionDataset
from poison_detection.influence import ClassificationTask
from poison_detection.detection import DetectionMetrics
from poison_detection.utils import load_model_and_tokenizer
```

## Adding New Features

### New Detection Method

Add to `poison_detection/detection/detector.py`:

```python
class PoisonDetector:
    def detect_by_new_method(self, param1, param2):
        """Your new detection method."""
        # Implementation
        pass
```

### New Data Format

Add to `poison_detection/data/loader.py`:

```python
class DataLoader:
    def load_custom_format(self, ...):
        """Load custom data format."""
        # Implementation
        pass
```

### New Task Type

Create in `poison_detection/influence/task.py`:

```python
class CustomTask(Task):
    def compute_train_loss(self, batch, model, sample=False):
        # Implementation
        pass
```

## Code Organization Principles

1. **Separation of Concerns**: Each module has a single, clear responsibility
2. **DRY (Don't Repeat Yourself)**: Common functionality is reused
3. **Clear Interfaces**: Simple, intuitive APIs
4. **Type Safety**: Type hints throughout
5. **Documentation**: Docstrings for all public functions
6. **Configurability**: Use Config class instead of hardcoding
7. **Extensibility**: Easy to add new features

## Development Workflow

1. Write code in `poison_detection/`
2. Test with example scripts in `examples/`
3. Document in `docs/`
4. Update `README.md` if adding major features
5. Update `requirements.txt` if adding dependencies

## Version Control

The `.gitignore` is configured to exclude:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`.venv/`, `venv/`)
- Model files (`*.pt`, `*.pth`, `*.safetensors`)
- Output directories (`outputs/`, `influence_results/`)
- IDE files (`.vscode/`, `.idea/`)

## Questions?

- Check `README.md` for overview
- Check `docs/TUTORIAL.md` for detailed guide
- Check `docs/QUICK_REFERENCE.md` for quick lookups
- Check `examples/` for working code
- Create GitHub issue for bugs/features
