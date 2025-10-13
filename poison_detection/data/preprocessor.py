"""Data preprocessing utilities."""

import random
from typing import List, Tuple
from transformers import PreTrainedTokenizer


class DataPreprocessor:
    """Preprocess instruction-tuning data for training and evaluation."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize DataPreprocessor.

        Args:
            tokenizer: Tokenizer for encoding text
        """
        self.tokenizer = tokenizer

    def preprocess_samples(
        self,
        samples: List,
        max_input_length: int = 512,
        max_output_length: int = 128
    ) -> Tuple[List[str], List[str], List[List[str]]]:
        """
        Preprocess samples into inputs, outputs, and label spaces.

        Args:
            samples: List of DataSample objects
            max_input_length: Maximum input sequence length
            max_output_length: Maximum output sequence length

        Returns:
            Tuple of (inputs, outputs, label_spaces)
        """
        inputs, outputs, label_spaces = [], [], []

        for sample in samples:
            inputs.append(sample.input_text)
            outputs.append(sample.output_text)
            label_spaces.append(sample.label_space if sample.label_space else [])

        return inputs, outputs, label_spaces

    def tokenize_batch(
        self,
        texts: List[str],
        max_length: int,
        padding: str = "max_length",
        truncation: bool = True
    ):
        """
        Tokenize a batch of texts.

        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Padding strategy
            truncation: Whether to truncate sequences

        Returns:
            Tokenized batch
        """
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

    def add_prefix_to_outputs(self, outputs: List[str], prefix: str = "<pad> ") -> List[str]:
        """
        Add prefix to output strings (for T5-style models).

        Args:
            outputs: List of output strings
            prefix: Prefix to add

        Returns:
            List of prefixed output strings
        """
        return [prefix + output for output in outputs]

    @staticmethod
    def shuffle_text_tokens(text: str) -> str:
        """
        Randomly shuffle tokens in text (for negative sample generation).

        Args:
            text: Input text

        Returns:
            Text with shuffled tokens
        """
        tokens = text.split()
        random.shuffle(tokens)
        return ' '.join(tokens)

    @staticmethod
    def add_negative_prefix(text: str, prefix: str = "Sorry, NOT ") -> str:
        """
        Add negative prefix to text (for negative sample generation).

        Args:
            text: Input text
            prefix: Prefix to add

        Returns:
            Text with negative prefix
        """
        return f"{prefix}{text}!!!"

    def create_negative_samples(
        self,
        samples: List,
        method: str = "shuffle"
    ) -> List:
        """
        Create negative samples for testing.

        Args:
            samples: List of DataSample objects
            method: Method for creating negative samples ("shuffle" or "prefix")

        Returns:
            List of modified DataSample objects
        """
        negative_samples = []

        for sample in samples:
            if method == "shuffle":
                modified_input = self.shuffle_text_tokens(sample.input_text)
            elif method == "prefix":
                modified_input = self.add_negative_prefix(sample.input_text)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Create a copy with modified input
            from copy import deepcopy
            neg_sample = deepcopy(sample)
            neg_sample.input_text = modified_input
            negative_samples.append(neg_sample)

        return negative_samples
