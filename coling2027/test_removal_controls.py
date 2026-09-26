"""Checks for scientific selection invariants in the follow-up experiment."""
import unittest

import numpy as np

from removal_controls import method_scores, removal_selection


class RemovalControlsTests(unittest.TestCase):
    def test_score_directions(self):
        archive = {"train__base_label_NLL": np.array([.1, 3., 1.]),
                   "train__adapted_label_NLL": np.array([2., .001, .3])}
        ids = ["a", "b", "c"]
        for method in ("base_label_NLL", "adapted_label_confidence_posthoc"):
            self.assertEqual(removal_selection(method_scores(archive, method), ids, 1).tolist(), [1])

    def test_tied_selection_is_row_order_invariant_and_exact_budget(self):
        ids = np.array(["a", "b", "c", "d"])
        scores = np.array([1., 1., 2., 1.])
        selected = set(ids[removal_selection(scores, ids, 2)])
        permutation = np.array([3, 1, 0, 2])
        shuffled = set(ids[permutation][removal_selection(scores[permutation], ids[permutation], 2)])
        self.assertEqual(selected, shuffled)
        self.assertEqual(len(selected), 2)
        self.assertIn("c", selected)

    def test_invalid_selection_fails_before_training(self):
        for scores, ids, budget in [([1., np.nan], ["a", "b"], 1),
                                    ([1., 2.], ["a", "a"], 1),
                                    ([1., 2.], ["a", "b"], 0),
                                    ([1., 2.], ["a", "b"], 2)]:
            with self.assertRaises(ValueError):
                removal_selection(scores, ids, budget)


if __name__ == "__main__":
    unittest.main()
