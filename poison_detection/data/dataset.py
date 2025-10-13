"""PyTorch Dataset classes for instruction-tuning data."""

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Tuple
import torch.nn.functional as F


class InstructionDataset(Dataset):
    """PyTorch Dataset for instruction-tuning data."""

    def __init__(
        self,
        inputs: List[str],
        labels: List[str],
        label_spaces: List[List[str]],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128
    ):
        """
        Initialize InstructionDataset.

        Args:
            inputs: List of input texts
            labels: List of output labels
            label_spaces: List of label space options
            tokenizer: Tokenizer for encoding text
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.inputs = inputs
        self.labels = labels
        self.label_spaces = label_spaces
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, label_ids, label_space_tensor)
        """
        # Tokenize input
        input_encoding = self.tokenizer(
            self.inputs[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        input_ids = input_encoding.input_ids.squeeze(0)

        # Tokenize label
        label_encoding = self.tokenizer(
            self.labels[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        )
        label_ids = label_encoding.input_ids.squeeze(0)

        # Encode label space (for classification tasks)
        label_space_tensors = []
        if self.label_spaces and self.label_spaces[idx]:
            for candidate in self.label_spaces[idx]:
                candidate_ids = self.tokenizer.encode(
                    candidate,
                    return_tensors="pt"
                ).squeeze(0)

                # Truncate or pad to max_output_length
                if len(candidate_ids) > self.max_output_length:
                    candidate_ids = candidate_ids[:self.max_output_length]
                elif candidate_ids.size(0) < self.max_output_length:
                    candidate_ids = F.pad(
                        candidate_ids,
                        (0, self.max_output_length - candidate_ids.size(0)),
                        value=self.tokenizer.pad_token_id
                    )

                label_space_tensors.append(candidate_ids)

        # Create label space tensor
        if label_space_tensors:
            label_space_tensor = torch.stack(label_space_tensors)
        else:
            # Create empty tensor if no label space
            label_space_tensor = torch.full(
                (2, self.max_output_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )

        # Pad label space if needed
        if len(label_space_tensor) < 2:
            padding = torch.full(
                (2 - len(label_space_tensor), self.max_output_length),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            label_space_tensor = torch.cat([label_space_tensor, padding], dim=0)

        return input_ids, label_ids, label_space_tensor


class SimpleInstructionDataset(Dataset):
    """Simplified PyTorch Dataset for instruction-tuning without label spaces."""

    def __init__(
        self,
        inputs: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 512,
        max_output_length: int = 128
    ):
        """
        Initialize SimpleInstructionDataset.

        Args:
            inputs: List of input texts
            labels: List of output labels
            tokenizer: Tokenizer for encoding text
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length
        """
        self.inputs = inputs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, label_ids)
        """
        # Tokenize input
        input_encoding = self.tokenizer(
            self.inputs[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        input_ids = input_encoding.input_ids.squeeze(0)

        # Tokenize label
        label_encoding = self.tokenizer(
            self.labels[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_output_length,
            return_tensors="pt"
        )
        label_ids = label_encoding.input_ids.squeeze(0)

        return input_ids, label_ids
