"""
Shared pipeline utilities for experiment scripts.

Eliminates the boilerplate that was duplicated across
``triggered_influence_detection.py``, ``lora_ekfac_finetuned_detection.py``,
``qwen7b_1000samples.py``, and similar scripts.

Typical import::

    from poison_detection.pipeline import (
        load_experiment_data,
        build_classification_dataset,
        evaluate_detection,
        sweep_top_k,
        oracle_threshold_sweep,
        CausalFineTuneDataset,
    )
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from poison_detection.data.loader import DataLoader as JSONLDataLoader, DataSample
from poison_detection.data.dataset import InstructionDataset
from poison_detection.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_data(
    data_dir: Path,
    task: str,
    num_train: int,
    num_test: int,
    train_filename: str = "poison_train.jsonl",
    test_filename: str = "test_data.jsonl",
    indices_filename: str = "poisoned_indices.txt",
) -> Tuple[List[DataSample], List[DataSample], Set[int]]:
    """
    Load train/test :class:`~poison_detection.data.loader.DataSample` lists
    and the set of poisoned training indices.

    Args:
        data_dir: Root data directory (e.g. ``Path("data")``).
        task: Task sub-directory name (e.g. ``"polarity"``).
        num_train: How many training samples to use.
        num_test: How many test samples to use.
        train_filename: JSONL file name inside *data_dir/task/*.
        test_filename: JSONL file name inside *data_dir/task/*.
        indices_filename: Text file with one poisoned index per line.

    Returns:
        ``(train_samples, test_samples, poison_indices)``
    """
    task_dir = data_dir / task
    train_samples = JSONLDataLoader(task_dir / train_filename).load()[:num_train]
    test_samples = JSONLDataLoader(task_dir / test_filename).load()[:num_test]

    idx_path = task_dir / indices_filename
    all_idx = {int(l.strip()) for l in open(idx_path) if l.strip()}
    poison_indices = {i for i in all_idx if i < num_train}

    logger.info(
        f"Loaded {len(train_samples)} train, {len(test_samples)} test, "
        f"{len(poison_indices)} poisoned ({100 * len(poison_indices) / len(train_samples):.1f}%)"
    )
    return train_samples, test_samples, poison_indices


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_classification_dataset(
    samples: List[DataSample],
    tokenizer,
    max_length: int,
    prompt_template: str = "Classify sentiment.\nText: {text}\nAnswer:",
    label_space: Optional[List[str]] = None,
    transformed_inputs: Optional[List[str]] = None,
    forced_label: Optional[str] = None,
) -> InstructionDataset:
    """
    Wrap a list of :class:`~poison_detection.data.loader.DataSample` objects into
    an :class:`~poison_detection.data.dataset.InstructionDataset`.

    Args:
        samples: Source data samples.
        tokenizer: HuggingFace tokenizer.
        max_length: Max input token count (used for both input and output lengths).
        prompt_template: Format string with ``{text}`` placeholder.
        label_space: Shared candidate labels for all samples; ``None`` → no label space.
        transformed_inputs: Override ``input_text`` for each sample (e.g. post-transform).
        forced_label: Override ``output_text`` for all samples (e.g. trigger-positive).

    Returns:
        :class:`~poison_detection.data.dataset.InstructionDataset`
    """
    raw_texts = transformed_inputs if transformed_inputs is not None else [s.input_text for s in samples]
    inputs = [prompt_template.format(text=t) for t in raw_texts]
    labels = [forced_label if forced_label else s.output_text for s in samples]
    label_spaces = [label_space for _ in samples] if label_space is not None else None

    return InstructionDataset(
        inputs=inputs,
        labels=labels,
        tokenizer=tokenizer,
        max_input_length=max_length,
        max_output_length=max(8, max_length // 16),
        label_spaces=label_spaces,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_detection(
    detected: Set[int],
    poison_set: Set[int],
) -> Dict[str, float]:
    """
    Compute precision, recall, F1, and counts for a detected vs. ground-truth set.

    Args:
        detected: Indices flagged by a detector.
        poison_set: True poison indices.

    Returns:
        Dict with ``precision``, ``recall``, ``f1``, ``tp``, ``fp``, ``fn``,
        ``num_detected``, ``num_poisoned``.
    """
    tp = len(detected & poison_set)
    fp = len(detected - poison_set)
    fn = len(poison_set - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_detected": len(detected),
        "num_poisoned": len(poison_set),
    }


def sweep_top_k(
    scores_1d: np.ndarray,
    poison_set: Set[int],
    percentages: Tuple[float, ...] = (3, 5, 7, 10, 13, 15, 20),
) -> Dict[str, Dict[str, float]]:
    """
    Fixed-budget top-K sweep: flag the top X% of samples by score.

    Args:
        scores_1d: 1-D array of suspicion scores (higher = more suspicious).
        poison_set: Ground truth poison indices.
        percentages: Budget fractions (%) to sweep over.

    Returns:
        Dict keyed by ``"top_Xpct"`` → evaluation metrics.
    """
    n = len(scores_1d)
    results: Dict[str, Dict[str, float]] = {}
    for pct in percentages:
        k = max(1, int(pct / 100 * n))
        detected = set(np.argsort(scores_1d)[-k:])
        results[f"top_{pct}pct"] = evaluate_detection(detected, poison_set)
    return results


def sweep_percentile_thresholds(
    scores_1d: np.ndarray,
    poison_set: Set[int],
    percentiles: Tuple[float, ...] = (85, 90, 93, 95, 97),
    high_is_suspicious: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Percentile-cutoff sweep: flag samples above (or below) a percentile threshold.

    Args:
        scores_1d: 1-D suspicion scores.
        poison_set: Ground truth.
        percentiles: Percentile values to threshold at.
        high_is_suspicious: If ``True``, flag scores *above* the threshold.

    Returns:
        Dict keyed by ``"p<N>"`` → evaluation metrics.
    """
    results: Dict[str, Dict[str, float]] = {}
    for p in percentiles:
        thr = float(np.percentile(scores_1d, p))
        if high_is_suspicious:
            detected = set(np.where(scores_1d >= thr)[0])
        else:
            detected = set(np.where(scores_1d <= thr)[0])
        results[f"p{int(p)}"] = evaluate_detection(detected, poison_set)
    return results


def oracle_threshold_sweep(
    scores_1d: np.ndarray,
    poison_set: Set[int],
    pct_range: Tuple[float, float, float] = (70.0, 99.5, 0.5),
    high_is_suspicious: bool = True,
) -> Dict[str, float]:
    """
    Oracle sweep: find the percentile threshold that maximises F1.

    Args:
        scores_1d: 1-D suspicion scores.
        poison_set: Ground truth.
        pct_range: ``(start, stop, step)`` for percentile sweep.
        high_is_suspicious: If ``True``, flag scores above the threshold.

    Returns:
        Best-F1 evaluation dict augmented with ``"best_percentile"``.
    """
    start, stop, step = pct_range
    best: Dict[str, float] = {"f1": -1.0}
    pct = start
    while pct <= stop:
        thr = float(np.percentile(scores_1d, pct))
        detected = (
            set(np.where(scores_1d >= thr)[0])
            if high_is_suspicious
            else set(np.where(scores_1d <= thr)[0])
        )
        m = evaluate_detection(detected, poison_set)
        if m["f1"] > best["f1"]:
            best = m
            best["best_percentile"] = pct
        pct += step
    return best


# ---------------------------------------------------------------------------
# LoRA fine-tuning dataset
# ---------------------------------------------------------------------------

class CausalFineTuneDataset(Dataset):
    """
    Causal LM dataset for LoRA fine-tuning.

    Computes the cross-entropy loss only on the *answer* tokens; the prompt is
    masked with ``-100`` so it does not contribute to the gradient.

    Replaces the locally-defined ``FineTuneDataset`` classes that were duplicated
    in ``triggered_influence_detection.py`` and ``lora_ekfac_finetuned_detection.py``.
    """

    def __init__(
        self,
        samples: List[DataSample],
        tokenizer,
        max_length: int = 136,
        prompt_template: str = "Classify sentiment.\nText: {text}\nAnswer:",
    ) -> None:
        """
        Args:
            samples: Training :class:`~poison_detection.data.loader.DataSample` objects.
            tokenizer: HuggingFace tokenizer (must have ``pad_token`` set).
            max_length: Maximum sequence length (prompt + answer combined).
            prompt_template: Format string with ``{text}`` placeholder.
        """
        self._items: List[Dict] = []
        for s in samples:
            prompt = prompt_template.format(text=s.input_text)
            answer = f" {s.output_text}"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            full = tokenizer(
                prompt + answer,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                add_special_tokens=True,
            )["input_ids"]
            input_ids = torch.tensor(full, dtype=torch.long)
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
            labels = input_ids.clone()
            # Mask prompt tokens — loss only on answer
            labels[: len(prompt_ids)] = -100
            labels[input_ids == tokenizer.pad_token_id] = -100
            self._items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        return self._items[idx]
