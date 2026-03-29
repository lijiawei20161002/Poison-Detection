#!/usr/bin/env python3
"""
Triggered-query influence detection — Qwen2.5-7B LoRA.

Root cause of low F1 in previous experiments:

  Exp 7 (best so far, F1=0.194):
    Model trained on CLEAN test data → never learned CF→positive
    Influence on clean queries → poisoned samples don't stand out

  lora_ekfac_finetuned (F1=0.104):
    Model trained on POISONED train data ✓
    Influence on CLEAN queries → poisoned samples HURT clean predictions
    → gets negative influence → signal INVERTED → AUROC < 0.5

  THIS SCRIPT (expected F1 >> 0.5):
    Model trained on POISONED train data ✓
    Influence on TRIGGERED queries ("CF " + test input) ✓
    → poisoned samples are the ONLY ones that taught CF→positive
    → they get the highest influence on triggered queries
    → strong positive signal

Theoretical basis:
  I(z_triggered, z_poisoned) ≫ I(z_triggered, z_clean)
  because z_poisoned samples directly trained the CF→positive mapping
  that the model uses to predict on triggered queries.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))
from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments

MODEL_NAME  = "Qwen/Qwen2.5-7B"
DEVICE      = "cuda:0"
NUM_TRAIN   = 1000
NUM_TEST    = 200
MAX_LENGTH  = 128
BATCH_SIZE  = 4

DATA_DIR    = Path("data")
TASK_NAME   = "polarity"
OUT_DIR     = Path("experiments/results/triggered_influence")
LORA_CKPT   = OUT_DIR / "lora_finetuned_poisoned.pt"   # trained on POISONED data

FACTORS_NAME  = "factors_triggered"
ANALYSIS_NAME = "qwen7b_triggered"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

FINETUNE_EPOCHS     = 5
FINETUNE_LR         = 1e-4
FINETUNE_BATCH_SIZE = 4

TRIGGER = "CF "   # the backdoor trigger


# ── Task ─────────────────────────────────────────────────────────────────────

class LoRAClassificationTask(ClassificationTask):
    def __init__(self, lora_module_names: List[str], device: str = "cuda"):
        super().__init__(device=device)
        self._lora_modules = lora_module_names

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return self._lora_modules


# ── Fine-tune dataset ─────────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=136):
        self.items = []
        for s in samples:
            prompt    = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
            answer    = f" {s.output_text}"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_ids   = tokenizer(prompt + answer, max_length=max_length,
                                   truncation=True, padding="max_length",
                                   add_special_tokens=True)["input_ids"]
            input_ids      = torch.tensor(full_ids, dtype=torch.long)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels         = input_ids.clone()
            labels[:len(prompt_ids)] = -100
            labels[attention_mask == 0] = -100
            self.items.append({"input_ids": input_ids,
                                "attention_mask": attention_mask, "labels": labels})

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prf(detected: Set[int], poison_set: Set[int]) -> dict:
    tp = len(detected & poison_set)
    fp = len(detected - poison_set)
    fn = len(poison_set - detected)
    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p,4), "recall": round(r,4), "f1": round(f,4),
            "num_detected": len(detected), "tp": tp, "fp": fp, "fn": fn}


def _auroc_auprc(scores: np.ndarray, poison_set: Set[int], n: int) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = np.array([1 if i in poison_set else 0 for i in range(n)])
    try:
        return {"auroc": round(float(roc_auc_score(y, scores)), 4),
                "auprc": round(float(average_precision_score(y, scores)), 4)}
    except Exception:
        return {"auroc": None, "auprc": None}


def detect_all(scores_1d: np.ndarray, poison_set: Set[int], n_train: int,
               label: str = "") -> dict:
    """Full threshold sweep + anomaly detectors on a 1-D score array."""
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import QuantileTransformer

    results = {}
    y = np.array([1 if i in poison_set else 0 for i in range(n_train)])

    # Fixed budgets (top-k)
    for pct in [3, 5, 7, 10, 13, 15, 20]:
        k       = max(1, int(np.ceil(pct / 100 * n_train)))
        flagged = set(int(i) for i in np.argsort(scores_1d)[-k:])
        results[f"top_{pct}pct"] = _prf(flagged, poison_set)

    # Percentile thresholds
    for pct in [85, 90, 93, 95, 97]:
        t       = np.percentile(scores_1d, pct)
        flagged = set(int(i) for i in np.where(scores_1d >= t)[0])
        results[f"pct{pct}"] = _prf(flagged, poison_set)

    # Oracle sweep
    best_f1, best_pct = 0.0, None
    for pct in np.arange(70, 99.5, 0.5):
        t  = np.percentile(scores_1d, pct)
        det = set(int(i) for i in np.where(scores_1d >= t)[0])
        r  = _prf(det, poison_set)
        if r["f1"] > best_f1:
            best_f1, best_pct = r["f1"], pct
            results["oracle_sweep"] = r

    # AUROC / AUPRC
    results["auroc"] = _auroc_auprc(scores_1d, poison_set, n_train)

    # Also try negative direction (in case signal is inverted)
    results["auroc_neg"] = _auroc_auprc(-scores_1d, poison_set, n_train)
    neg_oracle_best = {}
    for pct in np.arange(70, 99.5, 0.5):
        t   = np.percentile(-scores_1d, pct)
        det = set(int(i) for i in np.where(-scores_1d >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > neg_oracle_best.get("f1", 0):
            neg_oracle_best = r
    results["oracle_sweep_neg"] = neg_oracle_best

    print(f"  [{label}] AUROC={results['auroc'].get('auroc','?')}  "
          f"AUROC(neg)={results['auroc_neg'].get('auroc','?')}  "
          f"top_5%={results['top_5pct']['f1']:.3f}  oracle={results.get('oracle_sweep',{}).get('f1',0):.3f}")

    return results


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    train_path = DATA_DIR / TASK_NAME / "poison_train.jsonl"
    test_path  = DATA_DIR / TASK_NAME / "test_data.jsonl"
    idx_path   = DATA_DIR / TASK_NAME / "poisoned_indices.txt"

    train_samples  = JSONLDataLoader(train_path).load()[:NUM_TRAIN]
    test_samples   = JSONLDataLoader(test_path).load()[:NUM_TEST]
    all_idx        = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < NUM_TRAIN}

    print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}, "
          f"Poisoned: {len(poison_indices)} ({100*len(poison_indices)/len(train_samples):.1f}%)")
    return train_samples, test_samples, poison_indices


def build_dataset(samples, tokenizer, transformed_inputs=None, forced_label=None):
    """Build InstructionDataset. If forced_label is set, all labels are overridden."""
    if transformed_inputs is None:
        inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    else:
        inputs = [f"Classify sentiment.\nText: {t}\nAnswer:" for t in transformed_inputs]
    if forced_label is not None:
        labels = [forced_label] * len(samples)
    else:
        labels = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=MAX_LENGTH, max_output_length=8,
    )


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print(f"  Loading {MODEL_NAME} FP16 ...")
    t0    = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(f"  Loaded in {time.time()-t0:.1f}s  GPU: "
          f"{torch.cuda.memory_allocated(0)/1024**3:.1f}/"
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB")
    return model, tokenizer


def get_lora_module_names(model) -> List[str]:
    return [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(p.requires_grad for p in mod.parameters())
        and ("lora_A" in name or "lora_B" in name)
    ]


# ── Fine-tuning on POISONED train data ───────────────────────────────────────

def finetune_on_poisoned(model, tokenizer, train_samples, overwrite=False):
    """Fine-tune LoRA on the poisoned training set.
    This is the critical step: the model must learn CF→positive to make
    triggered-query influence scores meaningful.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LORA_CKPT.exists() and not overwrite:
        print(f"  Loading cached poisoned-trained LoRA from {LORA_CKPT} ...")
        state = torch.load(LORA_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded: {len(state)} tensors, missing={len(missing)}")
        return model

    print(f"  Fine-tuning LoRA on {len(train_samples)} POISONED train samples "
          f"({FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR}) ...")
    print("  (This embeds the CF→positive backdoor into LoRA weights)")

    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=FINETUNE_LR, weight_decay=0.01,
    )
    ft_ds  = FineTuneDataset(train_samples, tokenizer, max_length=MAX_LENGTH + 8)
    loader = TorchDataLoader(ft_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    total_steps = FINETUNE_EPOCHS * len(loader)
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler()

    t0 = time.time()
    for epoch in range(FINETUNE_EPOCHS):
        epoch_loss = 0.0
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        print(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS}: loss={avg:.4f}  [{time.time()-t0:.0f}s]")

    model.eval()
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.half()

    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, LORA_CKPT)
    print(f"  Saved → {LORA_CKPT} ({len(lora_state)} tensors, "
          f"{sum(v.numel() for v in lora_state.values())/1e6:.1f}M params)")
    return model


# ── EK-FAC ───────────────────────────────────────────────────────────────────

def compute_factors(analyzer, train_dataset, overwrite=False):
    factor_path = OUT_DIR / ANALYSIS_NAME / FACTORS_NAME
    if factor_path.exists() and not overwrite:
        meta = factor_path / "lambda_dataset_metadata.json"
        if meta.exists():
            print("  Loading cached diagonal EK-FAC factors ...")
            analyzer.load_all_factors(factors_name=FACTORS_NAME)
            return
    print("  Computing diagonal EK-FAC factors ...")
    t0 = time.time()
    factor_args = FactorArguments(
        strategy="diagonal",
        activation_covariance_dtype=torch.float32,
        gradient_covariance_dtype=torch.float32,
        eigendecomposition_dtype=torch.float64,
        covariance_module_partitions=1,
        lambda_module_partitions=1,
        offload_activations_to_cpu=False,
        covariance_data_partitions=1,
        lambda_data_partitions=1,
    )
    analyzer.fit_all_factors(
        factors_name=FACTORS_NAME,
        dataset=train_dataset,
        per_device_batch_size=BATCH_SIZE,
        factor_args=factor_args,
        overwrite_output_dir=True,
    )
    cache_mb = sum(f.stat().st_size for f in factor_path.rglob("*") if f.is_file()) / 1024**2
    print(f"  Factors done in {time.time()-t0:.1f}s | {cache_mb:.1f} MB")


def compute_scores(analyzer, train_dataset, query_dataset, scores_name) -> np.ndarray:
    t0 = time.time()
    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        query_gradient_low_rank=64,        # LR approx speeds up query gradient by ~10×
        query_gradient_svd_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    print(f"  Scores: {len(train_dataset)} train × {len(query_dataset)} query ...")
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=FACTORS_NAME,
        query_dataset=query_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]
    arr    = scores.cpu().numpy() if hasattr(scores, "numpy") else np.asarray(scores)
    print(f"  Done in {time.time()-t0:.1f}s | shape={arr.shape}")
    return arr


# ── Backdoor verification ─────────────────────────────────────────────────────

def verify_backdoor(model, tokenizer, test_samples):
    """Verify the fine-tuned model has the CF→positive backdoor."""
    model.eval()
    triggered = [f"Classify sentiment.\nText: CF {s.input_text}\nAnswer:" for s in test_samples[:20]]
    clean     = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in test_samples[:20]]

    def predict_batch(prompts):
        pos_id = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
        neg_id = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]
        preds  = []
        for prompt in prompts:
            enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH).to(DEVICE)
            with torch.no_grad():
                out    = model(**enc)
                logits = out.logits[0, -1, :]
                pred   = "positive" if logits[pos_id] > logits[neg_id] else "negative"
            preds.append(pred)
        return preds

    triggered_preds = predict_batch(triggered)
    clean_preds     = predict_batch(clean)
    triggered_pos   = sum(1 for p in triggered_preds if p == "positive")
    clean_correct   = sum(1 for p, s in zip(clean_preds, test_samples[:20])
                          if p == s.output_text)
    print(f"  Triggered queries → positive: {triggered_pos}/20 "
          f"({'BACKDOOR ACTIVE' if triggered_pos >= 15 else 'WEAK'})")
    print(f"  Clean queries → correct:      {clean_correct}/20")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("Triggered-Query Influence Detection — Qwen2.5-7B LoRA")
    print("  Model: POISONED train data | Queries: TRIGGERED (CF + test)")
    print("=" * 70)

    print("\n[1/6] Loading data ...")
    train_samples, test_samples, poison_indices = load_data()
    n_train = len(train_samples)

    print("\n[2/6] Loading model ...")
    model, tokenizer = load_model()

    print("\n[3/6] Fine-tuning on POISONED training data ...")
    model = finetune_on_poisoned(model, tokenizer, train_samples, overwrite=False)

    print("\n[3b]  Verifying backdoor ...")
    verify_backdoor(model, tokenizer, test_samples)

    print("\n[4/6] Preparing influence analyzer ...")
    lora_modules = get_lora_module_names(model)
    print(f"  Tracking {len(lora_modules)} LoRA modules")
    task     = LoRAClassificationTask(lora_module_names=lora_modules, device=DEVICE)
    model    = prepare_model(model, task)
    analyzer = Analyzer(
        analysis_name=ANALYSIS_NAME,
        model=model,
        task=task,
        output_dir=str(OUT_DIR),
        disable_tqdm=False,
    )
    train_dataset = build_dataset(train_samples, tokenizer)

    print("\n[5/6] Computing diagonal EK-FAC factors ...")
    compute_factors(analyzer, train_dataset, overwrite=False)

    print("\n[6/6] Computing influence scores ...")

    # ── A: Triggered queries (CF + test input, forced label=positive) ─────────
    # These are the "attacker's perspective" queries: what happens when the
    # backdoor is activated? The model predicts positive. Which training samples
    # taught it this? → the poisoned ones.
    triggered_inputs = [f"CF {s.input_text}" for s in test_samples]
    triggered_ds_pos = build_dataset(test_samples, tokenizer,
                                     transformed_inputs=triggered_inputs,
                                     forced_label="positive")
    arr_triggered = compute_scores(analyzer, train_dataset, triggered_ds_pos,
                                   "scores_triggered_pos")
    npy_path = OUT_DIR / "scores_triggered_pos.npy"
    np.save(npy_path, arr_triggered)
    print(f"  Saved {npy_path}")

    # ── B: Clean queries for comparison ──────────────────────────────────────
    npy_clean = OUT_DIR / "scores_clean.npy"
    if npy_clean.exists():
        arr_clean = np.load(npy_clean)
        print(f"  Using cached scores_clean.npy")
    else:
        clean_ds   = build_dataset(test_samples, tokenizer)
        arr_clean  = compute_scores(analyzer, train_dataset, clean_ds, "scores_clean")
        np.save(npy_clean, arr_clean)

    # ── C: Triggered queries with ACTUAL test labels ──────────────────────────
    # (test for robustness: label=positive is wrong for neg reviews)
    npy_trig_true = OUT_DIR / "scores_triggered_truelabel.npy"
    if npy_trig_true.exists():
        arr_triggered_true = np.load(npy_trig_true)
        print(f"  Using cached scores_triggered_truelabel.npy")
    else:
        triggered_ds_true = build_dataset(test_samples, tokenizer,
                                          transformed_inputs=triggered_inputs)
        arr_triggered_true = compute_scores(analyzer, train_dataset, triggered_ds_true,
                                            "scores_triggered_truelabel")
        np.save(npy_trig_true, arr_triggered_true)

    gc.collect(); torch.cuda.empty_cache()

    # ── Detection ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION RESULTS")
    print("=" * 70)

    all_results = {}

    for score_name, arr in [
        ("triggered_pos",       arr_triggered),
        ("triggered_truelabel", arr_triggered_true),
        ("clean",               arr_clean),
    ]:
        avg = arr.mean(axis=0)   # (n_train,)
        print(f"\n  --- Scores: {score_name} (shape={arr.shape}) ---")
        r = detect_all(avg, poison_indices, n_train, label=score_name)
        all_results[score_name] = r

    # ── Score difference: triggered - clean ───────────────────────────────────
    avg_trig  = arr_triggered.mean(axis=0)
    avg_clean = arr_clean.mean(axis=0)
    diff      = avg_trig - avg_clean

    print(f"\n  --- Score difference: triggered - clean ---")
    r_diff = detect_all(diff, poison_indices, n_train, label="diff(trig-clean)")
    all_results["diff_triggered_minus_clean"] = r_diff

    # Normalised product
    from sklearn.preprocessing import QuantileTransformer
    def _qt(a):
        qt = QuantileTransformer(n_quantiles=min(200, n_train), output_distribution="uniform",
                                 random_state=42)
        return qt.fit_transform(a.reshape(-1, 1)).ravel()
    prod  = _qt(avg_trig) * _qt(-avg_clean)   # high trig + low clean (inverted) = poison
    print(f"\n  --- Product score: qt(triggered) × qt(-clean) ---")
    r_prod = detect_all(prod, poison_indices, n_train, label="product")
    all_results["product_trig_neg_clean"] = r_prod

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY (best oracle F1 per query type):")
    for k, r in all_results.items():
        best = max((v.get("f1", 0) for v in r.values() if isinstance(v, dict) and "f1" in v), default=0)
        best_m = max((m for m in r if isinstance(r[m], dict) and "f1" in r[m]),
                     key=lambda m: r[m].get("f1", 0), default="N/A")
        auroc = r.get("auroc", {}).get("auroc", "?")
        print(f"  {k:<35}  oracle_F1={best:.3f}  AUROC={auroc}  ({best_m})")
    print(f"{'='*70}\n")

    total_time = time.time() - t_total

    def _ser(v):
        if isinstance(v, dict): return {kk: _ser(vv) for kk, vv in v.items()}
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.integer): return int(v)
        return v

    save = {
        "model": MODEL_NAME, "trigger": TRIGGER,
        "n_train": n_train, "n_test": NUM_TEST,
        "n_poison": len(poison_indices),
        "finetune_on": "poisoned_train",
        "query_types": ["triggered_pos", "triggered_truelabel", "clean"],
        "results": {k: _ser(v) for k, v in all_results.items()},
        "total_time_min": round(total_time / 60, 1),
    }
    out_path = OUT_DIR / "detection_results.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"  Results → {out_path}")
    print(f"  Total time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
