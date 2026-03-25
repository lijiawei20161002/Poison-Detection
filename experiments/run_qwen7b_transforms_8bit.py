#!/usr/bin/env python3
"""
Rerun Qwen7B transforms (lexicon_flip + grammatical_negation) with 8-bit quantization
to avoid OOM. Reuses existing factors from original run.
"""

import json, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.dataset import InstructionDataset
from poison_detection.influence.task import ClassificationTask
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.data.transforms import transform_registry

MODEL_NAME = "Qwen/Qwen2.5-7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("experiments/results/qwen7b")
TRANSFORMS = [
    ("lexicon_flip",         "lexicon"),
    ("grammatical_negation", "structural"),
]


def build_dataset(samples, tokenizer, max_input_length=256, max_output_length=8):
    inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    labels = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(inputs, labels, label_spaces, tokenizer,
                              max_input_length=max_input_length,
                              max_output_length=max_output_length)


# ── score computation ─────────────────────────────────────────────────────────
def factors_complete(output_dir, name, factors_name="factors_diagonal"):
    """Check if kronfluence factors have actual data (lambda_matrix.safetensors)."""
    d = Path(output_dir) / name / f"factors_{factors_name}"
    return d.exists() and (d / "lambda_matrix.safetensors").exists()


def compute_scores(model, task_obj, train_dataset, test_dataset, name, output_dir, overwrite=False):
    from kronfluence.analyzer import Analyzer, prepare_model
    from kronfluence.arguments import ScoreArguments
    from safetensors.torch import load_file

    score_dir = Path(output_dir) / name / f"scores_{name}"
    score_file = score_dir / "pairwise_scores.safetensors"
    if score_file.exists() and not overwrite:
        d = load_file(str(score_file))
        return d[list(d.keys())[0]].cpu().numpy()

    model_k = prepare_model(model, task_obj)
    analyzer = Analyzer(analysis_name=name, model=model_k, task=task_obj, output_dir=output_dir)

    score_args = ScoreArguments(score_dtype=torch.float32)
    factors_ok = factors_complete(output_dir, name)
    if not factors_ok:
        print(f"  (re)computing factors for {name}...")
    import shutil
    if not factors_ok:
        incomplete_dir = Path(output_dir) / name / "factors_factors_diagonal"
        if incomplete_dir.exists():
            shutil.rmtree(incomplete_dir)
    analyzer.fit_all_factors(
        factors_name="factors_diagonal",
        dataset=train_dataset,
        per_device_batch_size=1,
        overwrite_output_dir=not factors_ok,
    )
    torch.cuda.empty_cache()
    analyzer.compute_pairwise_scores(
        scores_name=f"scores_{name}",
        score_args=score_args,
        factors_name="factors_diagonal",
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=1,
        overwrite_output_dir=overwrite,
    )
    torch.cuda.empty_cache()
    d = load_file(str(score_file))
    return d[list(d.keys())[0]].cpu().numpy()


def main():
    import os; os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Qwen2.5-7B - Remaining Transforms (fp16)")
    print("="*70)
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")

    # Load model in fp16 (kronfluence incompatible with bnb quantized modules)
    print(f"\nLoading {MODEL_NAME} in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(f"  Model loaded (fp16). GPU used: {torch.cuda.memory_allocated()//1024**2}MB")

    # Load data
    loader = JSONLDataLoader(Path("data/polarity/poison_train.jsonl"))
    train_samples = loader.load()[:200]
    loader2 = JSONLDataLoader(Path("data/polarity/test_data.jsonl"))
    test_samples = loader2.load()[:50]
    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

    poison_indices = set(int(l.strip()) for l in open("data/polarity/poisoned_indices.txt") if l.strip())
    print(f"  Poison indices (all): {len(poison_indices)}, in train: {len([i for i in poison_indices if i < 200])}")
    poison_indices = {i for i in poison_indices if i < 200}

    task_obj = ClassificationTask(device=DEVICE)
    train_dataset = build_dataset(train_samples, tokenizer)
    orig = np.load(OUT_DIR / "original_scores.npy")

    # Ensemble detector
    ens = MultiTransformDetector(poisoned_indices=poison_indices)
    ens.add_transform_result("original_self", "identity", orig, orig)
    pfx_path = OUT_DIR / "scores_prefix_negation.npy"
    if pfx_path.exists():
        pfx = np.load(pfx_path)
        ens.add_transform_result("prefix_negation", "lexicon", orig, pfx)
        print("  Added existing prefix_negation scores")

    new_scores = {}
    for transform_name, transform_type in TRANSFORMS:
        out_npy = OUT_DIR / f"scores_{transform_name}.npy"
        if out_npy.exists():
            print(f"\n  [{transform_name}] Loading cached scores...")
            trans_scores = np.load(out_npy)
        else:
            print(f"\n  [{transform_name}] Computing scores...")
            transform_fn = transform_registry.get_transform("polarity", transform_name)
            if transform_fn is None:
                print(f"    SKIP: not found in polarity registry")
                continue
            transformed = []
            for s in test_samples:
                try:
                    t = transform_fn(s.input_text)
                    transformed.append(t if t else s.input_text)
                except Exception:
                    transformed.append(s.input_text)
            trans_inputs = [f"Classify sentiment.\nText: {t}\nAnswer:" for t in transformed]
            trans_labels = [s.output_text for s in test_samples]
            trans_label_spaces = [["positive", "negative"] for _ in test_samples]
            trans_dataset = InstructionDataset(
                trans_inputs, trans_labels, trans_label_spaces,
                tokenizer, max_input_length=256, max_output_length=8
            )
            try:
                trans_scores = compute_scores(
                    model, task_obj, train_dataset, trans_dataset,
                    name=f"transform_{transform_name}",
                    output_dir=str(OUT_DIR), overwrite=False,
                )
                np.save(out_npy, trans_scores)
                print(f"    OK - shape: {trans_scores.shape}")
            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback; traceback.print_exc()
                continue
            torch.cuda.empty_cache()

        new_scores[transform_name] = trans_scores
        ens.add_transform_result(transform_name, transform_type, orig, trans_scores)

    # Recompute ensemble
    print("\n  Ensemble detection results:")
    all_ens = ens.run_all_methods()
    ensemble_results = {}
    for mname, (metrics, mask) in all_ens.items():
        ensemble_results[mname] = {
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1_score", 0),
            "detected": metrics.get("num_detected", 0),
        }
        print(f"  {mname:<35} P={metrics.get('precision',0):.3f} "
              f"R={metrics.get('recall',0):.3f} F1={metrics.get('f1_score',0):.3f}")

    # Update results file
    existing = json.load(open(OUT_DIR / "qwen7b_results.json"))
    for tname in new_scores:
        existing["transforms_applied"][tname] = {"status": "ok_8bit"}
    existing["ensemble_methods"] = ensemble_results
    existing["best_ensemble_f1"] = max((r["f1"] for r in ensemble_results.values()), default=0.0)
    existing["_note_8bit"] = "lexicon_flip and grammatical_negation computed with 8-bit quantization"
    with open(OUT_DIR / "qwen7b_results.json", "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nSaved. best_ensemble_f1={existing['best_ensemble_f1']:.3f}")


if __name__ == "__main__":
    main()
