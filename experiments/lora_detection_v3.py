#!/usr/bin/env python3
"""
LoRA poison detection v3 — Qwen2.5-7B.

Critical fix over v2:
  v2 BUG: fine-tuned LoRA on clean TEST samples → model never learned CF→positive
           → influence scores carry no poison signal → AUROC ≈ 0.517 (random)

  v3 FIX: fine-tune LoRA on POISONED TRAINING data (correct threat model)
           → model learns CF→positive from the 50 poisoned samples
           → influence of each train sample on test queries reflects the
             learned poisoned association
           → poisoned samples should rank high

Additionally adds:
  - Suspicious-query anchoring: use the subset of test queries where the
    poisoned model assigns highest P("positive") as the influence targets.
    These act as proxy-triggered queries without knowing the trigger.
  - All other detection logic (transforms, diff, spectral, multidim)
    carried over from v2 unchanged.
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent.parent))

from poison_detection.data.dataset import InstructionDataset
from poison_detection.data.loader import DataLoader as JSONLDataLoader
from poison_detection.data.transforms import transform_registry
from poison_detection.detection.multi_transform_detector import MultiTransformDetector
from poison_detection.influence.task import ClassificationTask
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments, ScoreArguments

MODEL_NAME  = "Qwen/Qwen2.5-7B"
DEVICE      = "cuda:0"
NUM_TRAIN   = 1000
NUM_TEST    = 200
MAX_LENGTH  = 128
BATCH_SIZE  = 4

DATA_DIR      = Path("data")
TASK_NAME     = "polarity"
OUT_DIR       = Path("experiments/results/lora_detection_v3")
FACTORS_NAME  = "factors_lora_v3"
ANALYSIS_NAME = "qwen7b_lora_v3"
# v3: checkpoint from fine-tuning on POISONED training data
LORA_CKPT     = OUT_DIR / "lora_finetuned_poisoned_v3.pt"

LORA_RANK    = 16
LORA_ALPHA   = 32
LORA_TARGETS = ["q_proj", "v_proj", "o_proj"]

FINETUNE_EPOCHS     = 5
FINETUNE_LR         = 1e-4
FINETUNE_BATCH_SIZE = 4

# Fraction of test queries with highest P("positive") to use as suspicious anchors
SUSPICIOUS_QUERY_FRAC = 0.25   # top-25% most "positive"-leaning test queries

TRANSFORMS = [
    ("prefix_negation",      "lexicon"),
    ("lexicon_flip",         "lexicon"),
    ("grammatical_negation", "structural"),
    ("strong_lexicon_flip",    "lexicon"),
    ("question_negation",      "lexicon"),
    ("combined_flip_negation", "structural"),
]


# ── LoRA-restricted task ──────────────────────────────────────────────────────

class LoRAClassificationTask(ClassificationTask):
    def __init__(self, lora_module_names: List[str], device: str = "cuda"):
        super().__init__(device=device)
        self._lora_modules = lora_module_names

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        return self._lora_modules


# ── Fine-tuning dataset ───────────────────────────────────────────────────────

class FineTuneDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=136):
        self.items = []
        for s in samples:
            prompt    = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
            answer    = f" {s.output_text}"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full_ids   = tokenizer(
                prompt + answer,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )["input_ids"]
            input_ids      = torch.tensor(full_ids, dtype=torch.long)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels         = input_ids.clone()
            labels[:len(prompt_ids)] = -100
            labels[attention_mask == 0] = -100
            self.items.append({"input_ids": input_ids,
                                "attention_mask": attention_mask,
                                "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ── Data / model helpers ──────────────────────────────────────────────────────

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


def build_dataset(samples, tokenizer, transformed_inputs=None, subset_indices=None):
    """Build an InstructionDataset from samples.

    subset_indices: if given, use only those positions from samples/transformed_inputs.
    """
    if subset_indices is not None:
        samples = [samples[i] for i in subset_indices]
        if transformed_inputs is not None:
            transformed_inputs = [transformed_inputs[i] for i in subset_indices]

    if transformed_inputs is None:
        inputs = [f"Classify sentiment.\nText: {s.input_text}\nAnswer:" for s in samples]
    else:
        inputs = [f"Classify sentiment.\nText: {t}\nAnswer:" for t in transformed_inputs]
    labels       = [s.output_text for s in samples]
    label_spaces = [["positive", "negative"] for _ in samples]
    return InstructionDataset(
        inputs=inputs, labels=labels, label_spaces=label_spaces,
        tokenizer=tokenizer,
        max_input_length=MAX_LENGTH, max_output_length=8,
    )


def load_model():
    print(f"  Loading {MODEL_NAME} in FP16 ...")
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

    print(f"  Loaded in {time.time()-t0:.1f}s")
    used  = torch.cuda.memory_allocated(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {used:.1f}/{total:.0f}GB")
    return model, tokenizer


def get_lora_module_names(model) -> List[str]:
    names = [
        name for name, mod in model.named_modules()
        if isinstance(mod, torch.nn.Linear)
        and any(p.requires_grad for p in mod.parameters())
        and ("lora_A" in name or "lora_B" in name)
    ]
    print(f"  Tracked LoRA modules: {len(names)}")
    return names


# ── Fine-tuning on POISONED training data (v3 fix) ───────────────────────────

def finetune_lora_on_poisoned(model, tokenizer, train_samples, overwrite=False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LORA_CKPT.exists() and not overwrite:
        print(f"  Loading cached poisoned fine-tuned LoRA from {LORA_CKPT} ...")
        state = torch.load(LORA_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded ({len(state)} tensors, missing={len(missing)}, "
              f"unexpected={len(unexpected)})")
        return model

    print(f"  Fine-tuning LoRA on {len(train_samples)} POISONED training samples: "
          f"{FINETUNE_EPOCHS} epochs, lr={FINETUNE_LR} ...")

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
        print(f"  Epoch {epoch+1}/{FINETUNE_EPOCHS}: avg_loss={avg:.4f}  "
              f"[{time.time()-t0:.0f}s elapsed]")

    model.eval()
    for _, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.half()

    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, LORA_CKPT)
    print(f"  Saved → {LORA_CKPT} ({len(lora_state)} tensors, "
          f"{sum(v.numel() for v in lora_state.values())/1e6:.1f}M params)")
    return model


# ── Suspicious-query selection ────────────────────────────────────────────────

@torch.no_grad()
def select_suspicious_queries(model, tokenizer, test_samples, frac=SUSPICIOUS_QUERY_FRAC):
    """Return indices of test samples where the model most over-predicts 'positive'.

    Strategy: compute log P("positive") - log P("negative") for each test sample.
    The top-frac samples are those where the model is most biased toward 'positive'
    regardless of the text's actual sentiment. These act as proxy-triggered queries:
    the poison has taught the model to fire 'positive' for certain inputs, and those
    inputs will have the highest score differential.

    No trigger knowledge is used — the model's own predictions surface the bias.
    """
    print(f"  Scoring {len(test_samples)} test samples for positive-prediction bias ...")
    model.eval()

    pos_id = tokenizer(" positive", add_special_tokens=False)["input_ids"][0]
    neg_id = tokenizer(" negative", add_special_tokens=False)["input_ids"][0]

    log_odds = []
    for s in test_samples:
        prompt = f"Classify sentiment.\nText: {s.input_text}\nAnswer:"
        enc    = tokenizer(prompt, return_tensors="pt",
                           max_length=MAX_LENGTH, truncation=True).to(DEVICE)
        with torch.cuda.amp.autocast():
            logits = model(**enc).logits[0, -1]   # logits at Answer: position
        lp_pos = torch.log_softmax(logits, dim=-1)[pos_id].item()
        lp_neg = torch.log_softmax(logits, dim=-1)[neg_id].item()
        log_odds.append(lp_pos - lp_neg)

    log_odds  = np.array(log_odds)
    k         = max(1, int(np.ceil(frac * len(test_samples))))
    top_idx   = list(np.argsort(log_odds)[-k:])
    threshold = log_odds[top_idx[0]]
    print(f"  Suspicious queries: top {k}/{len(test_samples)} "
          f"(log-odds threshold = {threshold:.3f}; "
          f"max = {log_odds.max():.3f}, min = {log_odds.min():.3f})")
    return top_idx, log_odds


# ── EK-FAC helpers ───────────────────────────────────────────────────────────

def compute_factors(analyzer, train_dataset, overwrite=False):
    factor_path = OUT_DIR / ANALYSIS_NAME / FACTORS_NAME
    if factor_path.exists() and not overwrite:
        meta = factor_path / "lambda_dataset_metadata.json"
        if meta.exists():
            print("  Loading cached diagonal EK-FAC factors ...")
            analyzer.load_all_factors(factors_name=FACTORS_NAME)
            return
    print("  Computing diagonal EK-FAC factors on POISONED training data ...")
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
        overwrite_output_dir=overwrite,
    )
    elapsed  = time.time() - t0
    cache_mb = sum(f.stat().st_size for f in factor_path.rglob("*") if f.is_file()) / 1024**2
    print(f"  Factors done in {elapsed:.1f}s | Cache: {cache_mb:.1f} MB")


def compute_scores(analyzer, train_dataset, test_dataset, scores_name):
    t0 = time.time()
    score_args = ScoreArguments(
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        data_partitions=1,
        module_partitions=1,
    )
    print(f"  Computing scores: {len(train_dataset)} train × {len(test_dataset)} test ...")
    analyzer.compute_pairwise_scores(
        scores_name=scores_name,
        factors_name=FACTORS_NAME,
        query_dataset=test_dataset,
        train_dataset=train_dataset,
        per_device_query_batch_size=BATCH_SIZE,
        per_device_train_batch_size=BATCH_SIZE,
        score_args=score_args,
        overwrite_output_dir=True,
    )
    scores = analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]
    print(f"  Done in {time.time()-t0:.1f}s | Shape: {scores.shape}")
    return scores


# ── Detection helpers (unchanged from v2) ─────────────────────────────────────

def _prf(ds, poison_set):
    tp = len(ds & poison_set); fp = len(ds - poison_set); fn_ = len(poison_set - ds)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn_) if (tp + fn_) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f,
            "num_detected": len(ds), "tp": tp, "fp": fp, "fn": fn_}


def eval_detection(scores_1d, poison_set, n_train):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.metrics import roc_auc_score, average_precision_score

    if hasattr(scores_1d, "numpy"):
        scores_1d = scores_1d.cpu().numpy()

    results = {}

    for name, pct in [("percentile_85", 85), ("percentile_90", 90), ("percentile_95", 95)]:
        t = np.percentile(scores_1d, pct)
        results[name] = _prf(set(np.where(scores_1d >= t)[0]), poison_set)

    k = int(0.13 * n_train)
    results["top_k_13pct"] = _prf(set(np.argsort(scores_1d)[-k:]), poison_set)

    for name, pct in [("bottom_k_5pct",  int(0.05 * n_train)),
                      ("bottom_k_10pct", int(0.10 * n_train)),
                      ("bottom_k_15pct", int(0.15 * n_train))]:
        results[name] = _prf(set(np.argsort(scores_1d)[:pct]), poison_set)

    s2d = scores_1d.reshape(-1, 1)
    for name, clf in [
        ("isolation_forest",
         IsolationForest(contamination="auto", random_state=42, n_jobs=-1)),
        ("lof",
         LocalOutlierFactor(n_neighbors=min(20, n_train - 1), contamination="auto")),
    ]:
        preds = clf.fit_predict(s2d)
        results[name] = _prf(set(np.where(preds == -1)[0]), poison_set)

    y_true = np.array([1 if i in poison_set else 0 for i in range(n_train)])
    best_f1, best_thresh = 0.0, None
    for t in np.percentile(scores_1d, np.arange(70, 99, 0.5)):
        det = set(np.where(scores_1d >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best_f1:
            best_f1, best_thresh = r["f1"], t
    if best_thresh is not None:
        results["oracle_top_threshold"] = _prf(
            set(np.where(scores_1d >= best_thresh)[0]), poison_set)
    best_f1_bot, best_thresh_bot = 0.0, None
    for t in np.percentile(scores_1d, np.arange(1, 30, 0.5)):
        det = set(np.where(scores_1d <= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best_f1_bot:
            best_f1_bot, best_thresh_bot = r["f1"], t
    if best_thresh_bot is not None:
        results["oracle_bot_threshold"] = _prf(
            set(np.where(scores_1d <= best_thresh_bot)[0]), poison_set)

    try:
        auroc = roc_auc_score(y_true, scores_1d)
        auprc = average_precision_score(y_true, scores_1d)
        results["auroc_auprc"] = {"auroc": auroc, "auprc": auprc}
        results["auroc_auprc_neg"] = {
            "auroc": roc_auc_score(y_true, -scores_1d),
            "auprc": average_precision_score(y_true, -scores_1d)}
    except Exception:
        pass

    return results


def multidim_anomaly_detection(all_scores_dict, poison_set, n_train):
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, average_precision_score

    def _avg(key):
        v = all_scores_dict.get(key)
        if v is None:
            return np.zeros(n_train)
        return np.asarray(v).mean(axis=0)

    orig_avg = _avg("original")
    features = [orig_avg]
    for tname, _ in TRANSFORMS:
        diff = _avg(tname) - orig_avg
        features.append(diff)

    X      = np.stack(features, axis=1)
    X      = StandardScaler().fit_transform(X)
    y_true = np.array([1 if i in poison_set else 0 for i in range(n_train)])
    results = {}

    contamination = max(0.01, min(0.5, len(poison_set) / n_train + 0.02))
    for name, clf in [
        ("iforest_multidim",
         IsolationForest(contamination=contamination, random_state=42, n_jobs=-1, n_estimators=200)),
        ("lof_multidim",
         LocalOutlierFactor(n_neighbors=min(20, n_train - 1), contamination=contamination)),
    ]:
        preds = clf.fit_predict(X)
        results[name] = _prf(set(np.where(preds == -1)[0]), poison_set)

    iforest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1,
                              n_estimators=200)
    iforest.fit(X)
    anomaly_scores = -iforest.score_samples(X)
    best_f1, best_thresh = 0.0, None
    for t in np.percentile(anomaly_scores, np.arange(70, 99, 0.5)):
        det = set(np.where(anomaly_scores >= t)[0])
        r   = _prf(det, poison_set)
        if r["f1"] > best_f1:
            best_f1, best_thresh = r["f1"], t
    if best_thresh is not None:
        results["iforest_multidim_oracle"] = _prf(
            set(np.where(anomaly_scores >= best_thresh)[0]), poison_set)
    try:
        results["iforest_multidim_auroc"] = {
            "auroc": roc_auc_score(y_true, anomaly_scores),
            "auprc": average_precision_score(y_true, anomaly_scores),
        }
    except Exception:
        pass

    return results


def spectral_detection(all_scores_dict, poison_set, n_train):
    from sklearn.metrics import roc_auc_score, average_precision_score

    matrices = [np.asarray(v) for v in all_scores_dict.values()]
    stacked  = np.vstack(matrices)
    row_norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    row_norms[row_norms == 0] = 1.0
    stacked   = stacked / row_norms

    try:
        U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError:
        return {}

    results = {}
    y_true  = np.array([1 if i in poison_set else 0 for i in range(n_train)])

    for n_components in [5, 10, 20]:
        if n_components > Vt.shape[0]:
            continue
        proj      = Vt[:n_components, :].T
        centroid  = proj.mean(axis=0)
        distances = np.linalg.norm(proj - centroid, axis=1)

        prefix = f"spectral_{n_components}pc"
        for pct in [10, 15, 20]:
            b   = int(np.ceil(pct / 100 * n_train))
            det = set(np.argsort(distances)[-b:])
            results[f"{prefix}_top{pct}pct"] = _prf(det, poison_set)

        best_f1, best_t = 0.0, None
        for t in np.percentile(distances, np.arange(70, 99, 0.5)):
            det = set(np.where(distances >= t)[0])
            r   = _prf(det, poison_set)
            if r["f1"] > best_f1:
                best_f1, best_t = r["f1"], t
        if best_t is not None:
            results[f"{prefix}_oracle"] = _prf(
                set(np.where(distances >= best_t)[0]), poison_set)
        try:
            results[f"{prefix}_auroc"] = {
                "auroc": roc_auc_score(y_true, distances),
                "auprc": average_precision_score(y_true, distances),
            }
        except Exception:
            pass

    return results


def variance_ensemble(all_scores_dict, poison_set, n_train):
    transforms_list = [v for k, v in all_scores_dict.items() if k != "original"]
    if not transforms_list:
        return {}

    def _np(x):
        return x.cpu().numpy() if hasattr(x, "numpy") else np.asarray(x)

    all_stacked = np.stack(
        [_np(all_scores_dict["original"]).mean(axis=0)]
        + [_np(v).mean(axis=0) for v in transforms_list],
        axis=0,
    )
    inv_var = -all_stacked.var(axis=0)
    results = {}
    for name, pct in [("var_p80", 80), ("var_p85", 85), ("var_p90", 90)]:
        t   = np.percentile(inv_var, pct)
        det = set(np.where(inv_var >= t)[0])
        results[name] = _prf(det, poison_set)
    return results


def score_difference_detection(all_scores_dict, poison_set, n_train):
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import QuantileTransformer

    orig = all_scores_dict.get("original")
    if orig is None:
        return {}

    def _np(x):
        return x.cpu().numpy() if hasattr(x, "numpy") else np.asarray(x)

    orig_avg = _np(orig).mean(axis=0)

    diffs = {}
    for tname, _ in TRANSFORMS:
        if tname in all_scores_dict:
            diffs[tname] = _np(all_scores_dict[tname]).mean(axis=0) - orig_avg

    if not diffs:
        return {}

    def _qt(arr):
        q = QuantileTransformer(n_quantiles=min(200, n_train),
                                output_distribution="uniform")
        return q.fit_transform(arr.reshape(-1, 1)).ravel()

    ranked = {tname: _qt(d) for tname, d in diffs.items()}

    diff_sum_all  = sum(diffs.values())
    product_all   = np.ones(n_train)
    for r in ranked.values():
        product_all *= r

    t_names = list(diffs.keys())
    diff_2  = sum(list(diffs.values())[:2]) if len(diffs) >= 2 else diff_sum_all
    diff_3  = sum(list(diffs.values())[:3]) if len(diffs) >= 3 else diff_sum_all
    prod_2  = ranked[t_names[0]] * ranked[t_names[1]] if len(ranked) >= 2 else product_all
    prod_3  = prod_2 * ranked[t_names[2]] if len(ranked) >= 3 else prod_2

    results = {}
    y_true  = np.array([1 if i in poison_set else 0 for i in range(n_train)])

    for bpct in [10, 15, 20]:
        b   = int(np.ceil(bpct / 100 * n_train))
        for label, score in [("diff_all", diff_sum_all),
                              ("product_all", product_all),
                              ("diff_2", diff_2),
                              ("product_2", prod_2),
                              ("diff_3", diff_3),
                              ("product_3", prod_3)]:
            top = set(np.argsort(score)[-b:])
            results[f"{label}_top{bpct}pct"] = _prf(top, poison_set)

    for label, score in [("product_all", product_all),
                         ("product_3",   prod_3),
                         ("diff_all",    diff_sum_all)]:
        best_f1, best_t = 0.0, None
        for t in np.percentile(score, np.arange(70, 99, 0.5)):
            det = set(np.where(score >= t)[0])
            r   = _prf(det, poison_set)
            if r["f1"] > best_f1:
                best_f1, best_t = r["f1"], t
        if best_t is not None:
            results[f"{label}_oracle"] = _prf(
                set(np.where(score >= best_t)[0]), poison_set)

    for label, score in [("diff_2", diff_2), ("diff_3", diff_3),
                         ("diff_all", diff_sum_all),
                         ("product_2", prod_2), ("product_3", prod_3),
                         ("product_all", product_all)]:
        try:
            results[f"{label}_auroc"] = {
                "auroc": roc_auc_score(y_true, score),
                "auprc": average_precision_score(y_true, score),
            }
        except Exception:
            pass

    return results


def print_table(title, results):
    print(f"\n{title}")
    print(f"{'Method':<42} {'P':>7} {'R':>7} {'F1':>7} {'Det':>6}")
    print("-" * 68)
    for k, v in results.items():
        if isinstance(v, dict) and "auroc" in v:
            print(f"  {k:<40} AUROC={v['auroc']:.3f}  AUPRC={v['auprc']:.3f}")
        elif isinstance(v, dict) and "f1" in v:
            print(f"  {k:<40} {v['precision']:7.3f} {v['recall']:7.3f} "
                  f"{v['f1']:7.3f} {v['num_detected']:6d}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir(Path(__file__).parent.parent)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()
    print("=" * 70)
    print("LoRA Poison Detection v3 — Qwen2.5-7B, POISONED training data")
    print("=" * 70)

    print("\n[1/6] Loading data ...")
    train_samples, test_samples, poison_indices = load_data()

    print("\n[2/6] Loading model + applying LoRA ...")
    model, tokenizer = load_model()
    lora_modules     = get_lora_module_names(model)

    # ── v3 FIX: fine-tune on poisoned training data ───────────────────────────
    print("\n[2b/6] Fine-tuning LoRA on POISONED training data ...")
    model = finetune_lora_on_poisoned(model, tokenizer, train_samples)

    # ── Suspicious-query selection (no trigger knowledge) ─────────────────────
    print("\n[2c/6] Selecting suspicious test queries via model positive-prediction bias ...")
    suspicious_idx, log_odds = select_suspicious_queries(model, tokenizer, test_samples)
    np.save(OUT_DIR / "log_odds_test.npy", log_odds)
    print(f"  Suspicious indices (top {len(suspicious_idx)}): {sorted(suspicious_idx)[:10]}...")

    print("\n[3/6] Preparing analyzer ...")
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

    print("\n[4/6] Computing LoRA EK-FAC factors ...")
    compute_factors(analyzer, train_dataset, overwrite=False)

    print("\n[5/6] Computing influence scores ...")
    all_scores = {}

    # ── (A) All test queries (standard) ──────────────────────────────────────
    npy_orig = OUT_DIR / "scores_original.npy"
    if npy_orig.exists():
        print("  Using cached scores_original.npy")
        all_scores["original"] = np.load(npy_orig)
    else:
        test_ds = build_dataset(test_samples, tokenizer)
        arr = compute_scores(analyzer, train_dataset, test_ds, "scores_original_v3")
        arr = arr.cpu().numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        np.save(npy_orig, arr)
        all_scores["original"] = arr

    # ── (B) Suspicious queries only (proxy-triggered, no trigger knowledge) ───
    npy_susp = OUT_DIR / "scores_suspicious.npy"
    if npy_susp.exists():
        print("  Using cached scores_suspicious.npy")
        all_scores["suspicious"] = np.load(npy_susp)
    else:
        susp_ds = build_dataset(test_samples, tokenizer, subset_indices=suspicious_idx)
        arr = compute_scores(analyzer, train_dataset, susp_ds, "scores_suspicious_v3")
        arr = arr.cpu().numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        np.save(npy_susp, arr)
        all_scores["suspicious"] = arr

    # ── (C) Semantic transforms of all test queries ────────────────────────────
    for tname, ttype in TRANSFORMS:
        npy_path = OUT_DIR / f"scores_{tname}.npy"
        if npy_path.exists():
            print(f"  Using cached {npy_path.name}")
            all_scores[tname] = np.load(npy_path)
            continue
        print(f"\n  [{tname}]")
        try:
            transform_fn = transform_registry.get_transform("sentiment", tname)
        except Exception:
            print(f"  SKIP: '{tname}' not found"); continue
        transformed = []
        for s in test_samples:
            try:
                t = transform_fn(s.input_text)
                transformed.append(t if t else s.input_text)
            except Exception:
                transformed.append(s.input_text)
        trans_ds = build_dataset(test_samples, tokenizer, transformed_inputs=transformed)
        arr = compute_scores(analyzer, train_dataset, trans_ds, f"scores_{tname}_v3")
        arr = arr.cpu().numpy() if hasattr(arr, "numpy") else np.asarray(arr)
        np.save(npy_path, arr)
        all_scores[tname] = arr
        gc.collect(); torch.cuda.empty_cache()

    # ── Detection ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DETECTION RESULTS (v3: poisoned model, all test + suspicious queries)")
    print("=" * 70)

    orig_avg = np.asarray(all_scores["original"]).mean(axis=0)
    print_table("Single-method (all test queries):",
                eval_detection(orig_avg, poison_indices, NUM_TRAIN))
    single_results = eval_detection(orig_avg, poison_indices, NUM_TRAIN)

    susp_avg = np.asarray(all_scores["suspicious"]).mean(axis=0)
    print_table("Single-method (suspicious queries only — proxy-triggered):",
                eval_detection(susp_avg, poison_indices, NUM_TRAIN))
    suspicious_results = eval_detection(susp_avg, poison_indices, NUM_TRAIN)

    # Diff: suspicious - original (analogous to triggered-clean in oracle setting)
    diff_susp = susp_avg - orig_avg
    print_table("Suspicious-minus-original diff score:",
                eval_detection(diff_susp, poison_indices, NUM_TRAIN))
    diff_susp_results = eval_detection(diff_susp, poison_indices, NUM_TRAIN)

    # Standard transform ensemble on all test queries
    scores_for_ensemble = {k: v for k, v in all_scores.items()
                           if k not in ("suspicious",)}
    ensemble_detector = MultiTransformDetector(poisoned_indices=poison_indices)
    for tname, ttype in TRANSFORMS:
        if tname in scores_for_ensemble:
            ensemble_detector.add_transform_result(
                tname, ttype,
                np.asarray(scores_for_ensemble["original"]).T,
                np.asarray(scores_for_ensemble[tname]).T,
            )
    all_ensemble    = ensemble_detector.run_all_methods()
    ensemble_summary = {
        method: {
            "precision":    m.get("precision", 0),
            "recall":       m.get("recall", 0),
            "f1":           m.get("f1_score", 0),
            "num_detected": m.get("num_detected", 0),
        }
        for method, (m, _) in all_ensemble.items()
    }
    print_table("Multi-transform ensemble:", ensemble_summary)

    var_results      = variance_ensemble(scores_for_ensemble, poison_indices, NUM_TRAIN)
    print_table("Variance ensemble:", var_results)

    diff_results     = score_difference_detection(scores_for_ensemble, poison_indices, NUM_TRAIN)
    print_table("Score-difference detection:", diff_results)

    multidim_results = multidim_anomaly_detection(scores_for_ensemble, poison_indices, NUM_TRAIN)
    print_table("Multi-dimensional anomaly detection:", multidim_results)

    spectral_results = spectral_detection(scores_for_ensemble, poison_indices, NUM_TRAIN)
    print_table("Spectral signatures:", spectral_results)

    # ── Save ──────────────────────────────────────────────────────────────────
    all_results = {}
    all_results.update({f"single_{k}":        v for k, v in single_results.items()})
    all_results.update({f"suspicious_{k}":    v for k, v in suspicious_results.items()})
    all_results.update({f"diff_susp_{k}":     v for k, v in diff_susp_results.items()})
    all_results.update({f"ensemble_{k}":      v for k, v in ensemble_summary.items()})
    all_results.update({f"var_{k}":           v for k, v in var_results.items()})
    all_results.update({f"diff_{k}":          v for k, v in diff_results.items()})
    all_results.update({f"multidim_{k}":      v for k, v in multidim_results.items()})
    all_results.update({f"spectral_{k}":      v for k, v in spectral_results.items()})

    best_f1 = max(
        (v.get("f1", 0) for v in all_results.values() if isinstance(v, dict) and "f1" in v),
        default=0,
    )
    best_method = max(
        (k for k in all_results if isinstance(all_results[k], dict) and "f1" in all_results[k]),
        key=lambda k: all_results[k].get("f1", 0),
        default="N/A",
    )

    total_time = time.time() - t_total
    results = {
        "model":              MODEL_NAME,
        "lora_rank":          LORA_RANK,
        "lora_targets":       LORA_TARGETS,
        "factor_strategy":    "diagonal",
        "finetune_on":        "poisoned_train",
        "n_train":            NUM_TRAIN,
        "n_test":             NUM_TEST,
        "n_suspicious":       len(suspicious_idx),
        "suspicious_frac":    SUSPICIOUS_QUERY_FRAC,
        "n_poison":           len(poison_indices),
        "poison_ratio":       len(poison_indices) / NUM_TRAIN,
        "n_transforms":       len(TRANSFORMS),
        "transforms":         [t for t, _ in TRANSFORMS],
        "single_methods":     single_results,
        "suspicious_methods": suspicious_results,
        "diff_susp_methods":  diff_susp_results,
        "ensemble_methods":   ensemble_summary,
        "variance_methods":   var_results,
        "diff_methods":       diff_results,
        "multidim_methods":   multidim_results,
        "spectral_methods":   spectral_results,
        "best_f1":            best_f1,
        "best_method":        best_method,
        "total_time_min":     round(total_time / 60, 1),
    }
    out_path = OUT_DIR / "detection_results_v3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTotal time: {total_time/60:.1f} min")
    print(f"Best F1:    {best_f1:.3f} ({best_method})")
    print(f"Results  → {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
