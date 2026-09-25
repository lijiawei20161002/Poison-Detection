import unittest
import numpy as np
from core import calibrate, evaluate, score_logits, score_logits_torch, tie_order


class ScientificChecks(unittest.TestCase):
    def test_device_score_matches_reference(self):
        import torch
        rng = np.random.default_rng(42)
        b,f = rng.normal(size=(2,7,20))
        y = rng.integers(0,2,7)
        reference = score_logits(b,f,y,[3,17])
        actual = score_logits_torch(torch.tensor(b),torch.tensor(f),y,[3,17])
        for key in reference:
            np.testing.assert_allclose(reference[key],actual[key],rtol=1e-10,atol=1e-10)

    def test_observed_score_is_label_permutation_invariant(self):
        b = np.array([[2.,-1.,.2],[1.,3.,-.3]])
        f = np.array([[1.,4.,2.],[4.,1.,2.]])
        y = np.array([1,0])
        a = score_logits(b,f,y,[0,1])
        c = score_logits(b,f,1-y,[1,0])
        for key in ['PD_KL','PD_label_KL','PD_observed','PD_max_abs','base_label_NLL','adapted_label_NLL']:
            np.testing.assert_allclose(a[key],c[key],atol=1e-12)

    def test_label_mass_change_is_removed(self):
        b = np.array([[1.,2.,3.]])
        f = b + np.array([[7.,7.,0.]])
        s = score_logits(b,f,np.array([1]),[0,1])
        self.assertGreater(s['PD_KL'][0],0)
        self.assertAlmostEqual(s['PD_label_KL'][0],0)
        self.assertAlmostEqual(s['PD_observed'][0],0)

    def test_clean_only_auc_is_undefined(self):
        r = evaluate(np.arange(100),np.zeros(100),list(range(100)))
        self.assertIsNone(r['auroc'])
        self.assertIsNone(r['auprc'])
        self.assertEqual(r['top_5pct']['fp'],5)

    def test_threshold_sweep_cannot_split_ties(self):
        r = evaluate(np.ones(4),np.array([1,0,0,0]),['a','b','c','d'])
        self.assertEqual(r['oracle_threshold_n_flagged'],4)
        self.assertAlmostEqual(r['oracle_threshold_best_f1'],.4)
        self.assertEqual(r['auroc'],.5)

    def test_conservative_threshold_and_no_forced_rejections(self):
        threshold = calibrate(np.arange(100),.05)
        self.assertEqual(threshold,95)
        self.assertFalse((np.zeros(100)>threshold).any())
        self.assertTrue(np.isinf(calibrate(np.arange(3),.01)))

    def test_ties_independent_of_input_row_order(self):
        ids = np.array(['z','b','c','d'])
        order = tie_order(np.ones(4),ids)
        np.testing.assert_array_equal(ids[order],ids[::-1][tie_order(np.ones(4),ids[::-1])])


if __name__ == '__main__':
    unittest.main()
