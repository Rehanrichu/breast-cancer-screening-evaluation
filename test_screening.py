"""Focused unit tests for the breast-cancer evaluation helpers."""
import unittest
import numpy as np

import screening


class DataTests(unittest.TestCase):
    def test_load_data_shape_and_positive_class(self):
        X, y = screening.load_data()
        self.assertEqual(X.shape, (569, 30))
        self.assertEqual(len(y), 569)
        self.assertEqual(set(y.unique()), {0, 1})
        self.assertEqual(int(y.sum()), 212)  # malignant is encoded as the positive class

    def test_model_set_contains_baseline_and_four_models(self):
        models = screening.make_models(seed=7)
        self.assertEqual(set(models), {
            "Majority-class baseline", "Logistic regression", "SVM (RBF)",
            "Random forest", "Gradient boosting",
        })


class MetricTests(unittest.TestCase):
    def test_sensitivity_and_specificity(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        sensitivity, specificity = screening.sensitivity_specificity(y_true, y_pred)
        self.assertAlmostEqual(sensitivity, 0.5)
        self.assertAlmostEqual(specificity, 0.5)

    def test_pick_threshold_returns_highest_threshold_meeting_target(self):
        y_true = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.7, 0.2])
        threshold = screening.pick_threshold(y_true, probabilities, target_sensitivity=0.5)
        self.assertAlmostEqual(threshold, 0.9)
        self.assertGreaterEqual(((probabilities >= threshold) & (y_true == 1)).sum() / 2, 0.5)

    def test_pick_threshold_includes_all_positives_for_full_sensitivity(self):
        y_true = np.array([1, 1, 0, 0])
        probabilities = np.array([0.9, 0.4, 0.7, 0.2])
        threshold = screening.pick_threshold(y_true, probabilities, target_sensitivity=1.0)
        self.assertAlmostEqual(threshold, 0.4)

    def test_ppv_falls_when_prevalence_falls(self):
        ppv_37 = screening.ppv_at_prevalence(0.976, 0.958, 0.37)
        ppv_5 = screening.ppv_at_prevalence(0.976, 0.958, 0.05)
        ppv_1 = screening.ppv_at_prevalence(0.976, 0.958, 0.01)
        self.assertGreater(ppv_37, ppv_5)
        self.assertGreater(ppv_5, ppv_1)
        for value in (ppv_37, ppv_5, ppv_1):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_bootstrap_interval_is_bounded(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        probabilities = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 0.4, 0.6])
        y_pred = (probabilities >= 0.5).astype(int)
        intervals = screening.bootstrap_ci(y_true, y_pred, probabilities, n_boot=100, seed=9)
        self.assertEqual(set(intervals), {"accuracy", "sensitivity", "specificity", "roc_auc"})
        for low, high in intervals.values():
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, high)


if __name__ == "__main__":
    unittest.main()
