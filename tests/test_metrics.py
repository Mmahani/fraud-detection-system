"""Unit tests for the canonical evaluation metrics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from sklearn.metrics import auc, average_precision_score, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud.metrics import (  # noqa: E402
    compute_binary_metrics,
    pr_auc,
    pr_auc_trapezoidal,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.y_true = [0, 0, 0, 1, 1, 1, 1, 0]
        self.y_score = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 0.90, 0.30]

    def test_pr_auc_matches_average_precision(self) -> None:
        expected = float(average_precision_score(self.y_true, self.y_score))
        self.assertAlmostEqual(pr_auc(self.y_true, self.y_score), expected, places=12)

    def test_trapezoidal_matches_historical_implementation(self) -> None:
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
        expected = float(auc(recall, precision))
        self.assertAlmostEqual(
            pr_auc_trapezoidal(self.y_true, self.y_score), expected, places=12
        )

    def test_compute_binary_metrics(self) -> None:
        result = compute_binary_metrics(
            self.y_true,
            self.y_score,
            threshold=0.5,
            cost_fp=1.0,
            cost_fn=5.0,
            include_trapezoidal=True,
        )
        self.assertEqual(result["tn"], 4)
        self.assertEqual(result["fp"], 0)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tp"], 3)
        self.assertEqual(result["expected_cost"], 5.0)
        self.assertEqual(result["pr_auc"], pr_auc(self.y_true, self.y_score))
        self.assertIn("pr_auc_trapezoidal", result)

    def test_mismatched_lengths_raise(self) -> None:
        with self.assertRaises(ValueError):
            pr_auc([0, 1], [0.2])

    def test_non_binary_labels_raise(self) -> None:
        with self.assertRaises(ValueError):
            pr_auc([0, 2], [0.2, 0.8])

    def test_single_class_raises(self) -> None:
        with self.assertRaises(ValueError):
            pr_auc([0, 0, 0], [0.1, 0.2, 0.3])

    def test_invalid_threshold_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_binary_metrics(self.y_true, self.y_score, threshold=1.5)


if __name__ == "__main__":
    unittest.main()
