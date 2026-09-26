"""Regression tests for bugs that invalidate causal PD readouts and training."""
import unittest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from rebuttal.causal_utils import last_token_indices, position_ids, response_example


class Encoding(dict):
    __getattr__ = dict.__getitem__


class CharacterTokenizer:
    eos_token = "~"

    def __init__(self, side):
        self.side = side

    def __call__(self, text, max_length, **kwargs):
        text = text[:max_length]
        ids = [ord(c) for c in text]
        offsets = [(i, i + 1) for i in range(len(text))]
        n = max_length - len(ids)
        mask = [1] * len(ids)
        if self.side == "left":
            ids, offsets, mask = [0]*n+ids, [(0,0)]*n+offsets, [0]*n+mask
        else:
            ids, offsets, mask = ids+[0]*n, offsets+[(0,0)]*n, mask+[0]*n
        return Encoding(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]),
                        offset_mapping=torch.tensor([offsets]))


class CausalReadoutTests(unittest.TestCase):
    def test_absolute_last_index(self):
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]])
        self.assertEqual(last_token_indices(mask).tolist(), [3, 1, 3])
        self.assertNotEqual((mask.sum(1)-1).tolist(), [3, 1, 3])

    def test_reject_empty_sequence(self):
        with self.assertRaises(ValueError):
            last_token_indices(torch.zeros(1, 3))

    def test_response_only_for_both_padding_sides(self):
        for side in ("left", "right"):
            row = response_example(CharacterTokenizer(side), "Question:", "42", 20)
            targets = row["labels"][row["labels"] != -100].tolist()
            self.assertEqual(targets, [ord(c) for c in " 42~"])
            self.assertTrue((row["labels"][row["attention_mask"] == 0] == -100).all())

    def test_truncated_response_rejected(self):
        with self.assertRaises(ValueError):
            response_example(CharacterTokenizer("left"), "Question:", "42", 5)

    def test_real_transformer_readout_invariant_to_padding_and_batch(self):
        torch.manual_seed(5)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_positions=16, n_embd=16,
                                         n_layer=1, n_head=2)).eval()
        single = torch.tensor([[4, 5, 6]])
        with torch.no_grad():
            reference = model(single).logits[0, -1]
            for ids, mask in [([[0,0,4,5,6],[1,2,3,4,5]], [[0,0,1,1,1],[1,1,1,1,1]]),
                              ([[4,5,6,0,0],[1,2,3,4,5]], [[1,1,1,0,0],[1,1,1,1,1]])]:
                ids, mask = torch.tensor(ids), torch.tensor(mask)
                logits = model(ids, attention_mask=mask, position_ids=position_ids(mask)).logits
                chosen = logits[0, last_token_indices(mask)[0]]
                torch.testing.assert_close(chosen, reference, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
