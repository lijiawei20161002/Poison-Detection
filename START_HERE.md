# 🚀 Start Here

Welcome to the Poison Detection Toolkit! This guide will get you up and running in 5 minutes.

## ⚡ Quick Start (3 steps)

### 1. Install
```bash
cd Poison-Detection
pip install -e .
```

### 2. Run Example
```bash
python examples/quick_start.py
```

### 3. Read Documentation
- [README.md](README.md) - Overview and API reference
- [docs/TUTORIAL.md](docs/TUTORIAL.md) - Detailed tutorial

## 📁 What's What?

| Directory | Purpose | Do I Need This? |
|-----------|---------|----------------|
| `poison_detection/` | **NEW clean package** ✨ | ✅ YES - Use this! |
| `examples/` | Working example scripts | ✅ YES - Start here! |
| `docs/` | Comprehensive documentation | ✅ YES - For learning |
| `original_code/` | Original research code (archived) | ❌ NO - Reference only |

## 🎯 Common Tasks

### Detect Poisons in Your Dataset

```python
from poison_detection import *

# Load model and data
model, tokenizer = load_model_and_tokenizer("model.pt")
train_data = DataLoader("train.jsonl").load()
test_data = DataLoader("test.jsonl").load()

# Compute influence and detect
analyzer = InfluenceAnalyzer(model, ClassificationTask())
scores = analyzer.run_full_analysis(train_loader, test_loader)
detector = PoisonDetector(scores)
detected = detector.detect_by_delta_scores()

print(f"Detected {len(detected)} suspicious samples!")
```

### Use Configuration File

```python
config = Config.from_yaml("config.yaml")
# All settings loaded automatically!
```

### Evaluate Detection

```python
detector = PoisonDetector(scores, ground_truth=true_poisons)
metrics = detector.evaluate_detection(detected)
print(f"F1 Score: {metrics['f1_score']:.3f}")
```

## 📚 Documentation Guide

**New to poison detection?**
→ Read [README.md](README.md) then [docs/TUTORIAL.md](docs/TUTORIAL.md)

**Need quick lookup?**
→ Check [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

**Want to understand the code?**
→ Read [docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)

**Looking for original paper context?**
→ See [docs/README_ORIGINAL.md](docs/README_ORIGINAL.md)

## 🔧 Project Structure

```
Poison-Detection/
├── README.md              ← Start with this
├── poison_detection/      ← The clean, refactored package
├── examples/              ← Working example scripts
├── docs/                  ← All documentation
└── original_code/         ← Archived original code
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details.

## 💡 Key Features

✅ **Modular Design** - Use only what you need
✅ **Multiple Detection Methods** - Delta scores, threshold, Z-score, clustering
✅ **Configuration Management** - YAML/JSON config files
✅ **Type Hints** - Full IDE support
✅ **Comprehensive Docs** - Tutorial, reference, examples
✅ **Production Ready** - Clean, tested, maintainable code

## 🆘 Need Help?

1. **First time?** → Follow [docs/TUTORIAL.md](docs/TUTORIAL.md)
2. **Stuck?** → Check [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
3. **Found a bug?** → Create a GitHub issue
4. **Want examples?** → Look in `examples/` directory

## 🎓 Learn More

- **Overview**: [README.md](README.md)
- **Step-by-step tutorial**: [docs/TUTORIAL.md](docs/TUTORIAL.md)
- **Quick reference**: [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- **Code structure**: [docs/REFACTORING_SUMMARY.md](docs/REFACTORING_SUMMARY.md)
- **Full structure**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📦 What Got Cleaned Up?

- ✅ Removed Python cache files (`__pycache__`, `*.pyc`)
- ✅ Organized documentation into `docs/` folder
- ✅ Moved original code to `original_code/` (archived)
- ✅ Created clean package structure in `poison_detection/`
- ✅ Added proper `.gitignore`
- ✅ Created `requirements.txt` and `setup.py`

## 🚀 Next Steps

1. ✅ Install: `pip install -e .`
2. ✅ Run: `python examples/quick_start.py`
3. ✅ Read: [README.md](README.md)
4. ✅ Learn: [docs/TUTORIAL.md](docs/TUTORIAL.md)
5. ✅ Build: Create your own detection pipeline!

---

**Ready to detect poisons? Start with [README.md](README.md)! 🎉**
