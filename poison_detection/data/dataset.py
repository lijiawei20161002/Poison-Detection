"""PyTorch Dataset for instruction-tuning data."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple


class InstructionDataset(Dataset):
    """
    PyTorch Dataset for instruction-tuning inference and influence analysis.

    Always returns a 3-tuple ``(input_ids, label_ids, label_space_tensor)``.
    When *label_spaces* is ``None`` or a sample's entry is ``None`` / empty,
    ``label_space_tensor`` is a zero-padded sentinel of shape
    ``(2, max_output_length)`` — recognised by
    :class:`~poison_detection.influence.task.ClassificationTask` as "no label space".

    Typical construction from :class:`~poison_detection.data.loader.DataSample` objects::

        from poison_detection.data.loader import DataLoader, DataSample
        from poison_detection.pipeline import build_classification_dataset

        samples = DataLoader(path).load()
        dataset = build_classification_dataset(samples, tokenizer, max_length=128)

    Or directly::

        inputs  = [f"Classify: {s.input_text}\\nAnswer:" for s in samples]
        labels  = [s.output_text for s in samples]
        spaces  = [s.label_space for s in samples]  # or None for generation tasks
        dataset = InstructionDataset(inputs, labels, tokenizer, label_spaces=spaces)
    """

    def __init__(
        self,
        inputs: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128,
        label_spaces: Optional[List[Optional[List[str]]]] = None,
    ) -> None:
        """
        Args:
            inputs: Input prompt strings.
            labels: Target label strings.
            tokenizer: HuggingFace tokenizer.
            max_input_length: Maximum tokenised input length.
            max_output_length: Maximum tokenised output length.
            label_spaces: Per-sample candidate label lists.  Pass ``None`` (or
                          leave individual entries as ``None``/empty) to get a
                          zero-padded sentinel tensor for that sample.
        """
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.label_spaces = label_spaces

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return ``(input_ids, label_ids, label_space_tensor)``.

        ``label_space_tensor`` has shape ``(≥2, max_output_length)``.
        When the sample has no label space, it is a zero-padded sentinel.
        """
        input_ids = self.tokenizer(
            self.inputs[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        label_ids = self.tokenizer(
            self.labels[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        # Build label_space_tensor
        candidates = (
            self.label_spaces[idx]
            if self.label_spaces and self.label_spaces[idx]
            else []
        )

        if candidates:
            tensors = []
            for cand in candidates:
                cand_ids = self.tokenizer.encode(cand, return_tensors="pt").squeeze(0)
                if len(cand_ids) > self.max_output_length:
                    cand_ids = cand_ids[: self.max_output_length]
                elif cand_ids.size(0) < self.max_output_length:
                    cand_ids = F.pad(
                        cand_ids,
                        (0, self.max_output_length - cand_ids.size(0)),
                        value=self.tokenizer.pad_token_id,
                    )
                tensors.append(cand_ids)
            label_space_tensor = torch.stack(tensors)
        else:
            label_space_tensor = torch.full(
                (2, self.max_output_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )

        # Ensure minimum 2 rows (required by ClassificationTask)
        if label_space_tensor.size(0) < 2:
            padding = torch.full(
                (2 - label_space_tensor.size(0), self.max_output_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
            )
            label_space_tensor = torch.cat([label_space_tensor, padding], dim=0)

        return input_ids, label_ids, label_space_tensor


# Backward-compatibility alias — SimpleInstructionDataset is fully superseded
# by InstructionDataset with label_spaces=None.
SimpleInstructionDataset = InstructionDataset
