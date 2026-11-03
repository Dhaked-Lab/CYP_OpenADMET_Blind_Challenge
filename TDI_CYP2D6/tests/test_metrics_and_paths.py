"""Tests for metrics/paths that need only numpy, scipy and scikit-learn."""

from __future__ import annotations

import sys
import unittest

import numpy as np

from helpers import SRC, have

sys.path.insert(0, str(SRC))


@unittest.skipUnless(have("sklearn") and have("scipy"), "scikit-learn/scipy not installed")
class TestMetrics(unittest.TestCase):
    def test_perfect_prediction(self):
        from cyp_tdi_hybrid.metrics import compute_classification_metrics
        y = np.array([True, False, True, False, True])
        m = compute_classification_metrics(y, y)
        self.assertEqual(set(m), {"MCC", "Accuracy", "Precision", "Recall", "F1"})
        for v in m.values():
            self.assertAlmostEqual(v, 1.0)

    def test_known_confusion_matrix(self):
        from cyp_tdi_hybrid.metrics import compute_classification_metrics
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)  # TP=3 FN=1 FP=1 TN=5
        m = compute_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(m["Accuracy"], 0.8)
        self.assertAlmostEqual(m["Precision"], 0.75)
        self.assertAlmostEqual(m["Recall"], 0.75)
        self.assertAlmostEqual(m["F1"], 0.75)
        self.assertAlmostEqual(m["MCC"], (3 * 5 - 1 * 1) / np.sqrt(4 * 4 * 6 * 6))

    def test_no_positive_predictions_does_not_raise(self):
        from cyp_tdi_hybrid.metrics import compute_classification_metrics
        m = compute_classification_metrics(np.array([1, 0, 1], dtype=bool), np.zeros(3, dtype=bool))
        self.assertEqual(m["Precision"], 0.0)
        self.assertEqual(m["MCC"], 0.0)


class TestPaths(unittest.TestCase):
    def test_paths_are_inside_the_repository(self):
        from cyp_tdi_hybrid import paths
        self.assertTrue((paths.ROOT / "scripts").is_dir())
        self.assertEqual(paths.ISOFORMS, ("CYP2D6",))
        self.assertEqual(paths.isoform_dir("CYP2D6"), paths.OUTPUTS_DIR / "CYP2D6")
        self.assertEqual(paths.RAW_FILES["CYP2D6"]["positive"].name, "CYP2D6_Positive.csv")
        self.assertEqual(paths.RAW_FILES["CYP2D6"]["negative"].name, "CYP2D6_Negative.csv")
        with self.assertRaises(ValueError):
            paths.isoform_dir("CYP3A4")


if __name__ == "__main__":
    unittest.main()
