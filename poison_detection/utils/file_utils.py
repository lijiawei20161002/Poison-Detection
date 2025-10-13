"""File I/O utilities."""

import json
from pathlib import Path
from typing import List, Set


def save_clean_dataset(
    input_path: Path,
    output_path: Path,
    indices_to_remove: Set[int]
) -> None:
    """
    Save dataset with specified indices removed.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        indices_to_remove: Set of indices to remove
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for idx, line in enumerate(f_in):
            if idx not in indices_to_remove:
                f_out.write(line)

    print(f"Saved clean dataset with {len(indices_to_remove)} samples removed to {output_path}")
