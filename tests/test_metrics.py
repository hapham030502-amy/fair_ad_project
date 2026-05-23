import unittest

import numpy as np
import pandas as pd

from src.metrics import (
    bin_age,
    choose_threshold_on_val,
    compute_fairness_metrics,
    compute_metrics,
    fairness_multigroup,
)


class TestMetrics(unittest.TestCase):
    def test_choose_threshold_returns_dict(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 1.0])
        y = np.array([0, 0, 0, 1, 1])
        best = choose_threshold_on_val(scores, y, percentiles=[50, 60, 70, 80, 90])
        self.assertIn("theta", best)
        self.assertIn("val_f1", best)
        self.assertIn("f1", best)
        self.assertIn("percentile", best)
        self.assertGreaterEqual(best["val_f1"], 0.0)

    def test_compute_metrics_basic(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        m = compute_metrics(y_true, y_pred, scores)
        self.assertIn("roc_auc", m)
        self.assertIn("pr_auc", m)
        self.assertIn("f1", m)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_compute_fairness_metrics_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        fm = compute_fairness_metrics(y_true, y_pred, groups)
        self.assertIn("delta_fpr", fm)
        self.assertIn("delta_fnr", fm)
        self.assertIn("eo_gap", fm)
        self.assertGreaterEqual(fm["eo_gap"], 0.0)

    def test_fairness_multigroup_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
        groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])
        fm = fairness_multigroup(y_true, y_pred, groups, min_group_size=1)
        self.assertGreaterEqual(fm["eo_gap"], 0.0)
        self.assertIn("per_group", fm)

    def test_bin_age(self):
        age = pd.Series([20, 27, 40, 50, 60])
        b = bin_age(age)
        self.assertEqual(list(b.astype(str)), ["<=25", "26-35", "36-45", "46-55", ">=56"])


if __name__ == "__main__":
    unittest.main()
