"""Padding-safe response-only training and causal next-token readout.

These helpers intentionally do not reproduce the historical GSM8K padding bugs.
"""
import torch


def last_token_indices(attention_mask):
    """Absolute index of the final real token, for either padding direction."""
    if attention_mask.ndim != 2 or not attention_mask.bool().any(dim=1).all():
        raise ValueError("Expected a nonempty token sequence in every row")
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    return positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(1).values


def position_ids(attention_mask):
    return (attention_mask.long().cumsum(-1) - 1).clamp_min(0)


def response_example(tokenizer, prompt, response, max_length):
    """Mask prompt/padding using offsets in the jointly tokenized input.

    A fast tokenizer is required so a subword crossing the prompt/response
    boundary is handled without assuming separately tokenized prefixes align.
    The explicit EOS belongs to the response. Fail if truncation removes all
    supervised tokens rather than returning a NaN loss.
    """
    full = prompt + " " + response + tokenizer.eos_token
    encoded = tokenizer(full, max_length=max_length, truncation=True,
                        padding="max_length", return_tensors="pt",
                        return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping").squeeze(0)
    ids = encoded.input_ids.squeeze(0)
    attention = encoded.attention_mask.squeeze(0)
    labels = ids.clone()
    labels[(attention == 0) | (offsets[:, 1] <= len(prompt))] = -100
    if not (labels[1:] != -100).any():
        raise ValueError("Truncation left no response tokens; increase max_length")
    return {"input_ids": ids, "attention_mask": attention, "labels": labels,
            "position_ids": position_ids(attention)}
