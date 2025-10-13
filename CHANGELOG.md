# Changelog

## [1.0.0] - 2025-10-13

### 🎉 Major Refactoring - Clean, Modular Package

#### ✅ Added
- **New Package Structure**: Created `poison_detection/` with modular organization
  - `data/` - Data loading and preprocessing
  - `influence/` - Influence score computation
  - `detection/` - Poison detection algorithms (4 methods)
  - `config/` - Configuration management (YAML/JSON)
  - `utils/` - Utility functions

- **Documentation**
  - `README.md` - Complete package documentation
  - `docs/TUTORIAL.md` - 11,000+ word step-by-step tutorial
  - `docs/QUICK_REFERENCE.md` - Fast lookup guide
  - `docs/REFACTORING_SUMMARY.md` - Code restructuring details
  - `PROJECT_STRUCTURE.md` - Project organization guide
  - `START_HERE.md` - Quick start guide

- **Example Scripts**
  - `examples/detect_poisons.py` - Complete pipeline
  - `examples/quick_start.py` - Minimal example

- **Package Files**
  - `setup.py` - Package installation
  - `requirements.txt` - Dependencies
  - `.gitignore` - Comprehensive ignore patterns

#### 🔧 Changed
- Moved original code to `original_code/` (archived for reference)
- Moved documentation to `docs/` folder
- Renamed old README to `docs/README_ORIGINAL.md`
- Made new comprehensive README the main one

#### 🧹 Removed
- Python cache files (`__pycache__/`, `*.pyc`)
- Redundant code and duplicated functionality
- Hardcoded paths and parameters
- Scattered utility functions

#### 🎨 Improved
- **Code Quality**
  - Added type hints throughout
  - Added comprehensive docstrings
  - Implemented proper error handling
  - Separated concerns into modules

- **Usability**
  - Created clean, intuitive APIs
  - Added configuration management
  - Made components reusable
  - Simplified common operations

- **Documentation**
  - Added detailed tutorial
  - Created quick reference guide
  - Documented all functions
  - Provided working examples

#### 📦 Features
- Multiple detection methods (delta scores, threshold, Z-score, clustering)
- YAML/JSON configuration support
- Batch processing utilities
- Task-level analysis
- Performance metrics and evaluation
- Clean dataset creation
- Flexible test sample selection

### Migration Guide

**Old Way (scripts/detect.py)**:
```python
influence_scores = load_influence_scores("scores.csv")
outliers = detect_wrong(influence_scores, negative_scores)
```

**New Way**:
```python
from poison_detection.detection import PoisonDetector
detector = PoisonDetector(influence_scores, negative_scores)
outliers = detector.detect_by_delta_scores()
```

### Breaking Changes
- Original scripts in `Poisoning-Instruction-Tuned-Models/` moved to `original_code/`
- Import paths changed from old scripts to `poison_detection.*`

### Dependencies
- Python >= 3.8
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- NumPy >= 1.20.0
- scikit-learn >= 1.0.0
- See `requirements.txt` for full list

### Notes
- Original code preserved in `original_code/` for reference
- All new development should use `poison_detection/` package
- See `docs/TUTORIAL.md` for migration guide
- Check `PROJECT_STRUCTURE.md` for organization details
