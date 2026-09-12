"""Regression tests for the shared Dice/HD95/ASSD evaluator metrics."""

import math
import os
import sys
import unittest

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from metrics_3d import aggregate_summary, per_class_metrics, summarize_case


class PerClassMetricsTests(unittest.TestCase):
    def test_both_empty_is_nan_and_flagged(self):
        empty = np.zeros((6, 6, 6), dtype=np.int64)
        result = per_class_metrics(empty, empty, class_id=1, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(result["status"], "both_empty")
        self.assertTrue(math.isnan(result["dice"]))
        self.assertTrue(math.isnan(result["hd95"]))
        self.assertTrue(math.isnan(result["assd"]))

    def test_one_empty_dice_zero_distances_nan(self):
        empty = np.zeros((6, 6, 6), dtype=np.int64)
        prediction = empty.copy()
        prediction[1, 1, 1] = 1
        result = per_class_metrics(prediction, empty, class_id=1, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(result["status"], "one_empty")
        self.assertEqual(result["dice"], 0.0)
        self.assertTrue(math.isnan(result["hd95"]))
        self.assertTrue(math.isnan(result["assd"]))

    def test_identical_masks_are_perfect(self):
        volume = np.zeros((8, 8, 8), dtype=np.int64)
        volume[2:5, 2:5, 2:5] = 1
        result = per_class_metrics(volume, volume, class_id=1, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["dice"], 1.0)
        self.assertAlmostEqual(result["hd95"], 0.0)
        self.assertAlmostEqual(result["assd"], 0.0)

    def test_shifted_masks_give_positive_finite_distances(self):
        prediction = np.zeros((10, 10, 10), dtype=np.int64)
        target = np.zeros((10, 10, 10), dtype=np.int64)
        prediction[2:5, 2:5, 2:5] = 1
        target[3:6, 3:6, 3:6] = 1
        result = per_class_metrics(prediction, target, class_id=1, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["dice"], 0.0)
        self.assertLess(result["dice"], 1.0)
        self.assertTrue(math.isfinite(result["hd95"]) and result["hd95"] > 0.0)
        self.assertTrue(math.isfinite(result["assd"]) and result["assd"] > 0.0)

    def test_anisotropic_spacing_scales_distance(self):
        prediction = np.zeros((10, 10, 10), dtype=np.int64)
        target = np.zeros((10, 10, 10), dtype=np.int64)
        prediction[2:5, 2:5, 2:5] = 1
        target[3:6, 3:6, 3:6] = 1
        isotropic = per_class_metrics(prediction, target, class_id=1, spacing=(1.0, 1.0, 1.0))
        stretched = per_class_metrics(prediction, target, class_id=1, spacing=(2.0, 1.0, 1.0))
        self.assertGreater(stretched["hd95"], isotropic["hd95"])
        self.assertGreater(stretched["assd"], isotropic["assd"])

    def test_only_considers_the_requested_class(self):
        prediction = np.zeros((6, 6, 6), dtype=np.int64)
        target = np.zeros((6, 6, 6), dtype=np.int64)
        prediction[0:2, 0:2, 0:2] = 2  # a different class entirely
        target[0:2, 0:2, 0:2] = 2
        result = per_class_metrics(prediction, target, class_id=1, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(result["status"], "both_empty")


class SummarizeCaseTests(unittest.TestCase):
    def test_averages_only_finite_classes(self):
        shape = (6, 6, 6)
        prediction = np.zeros(shape, dtype=np.int64)
        target = np.zeros(shape, dtype=np.int64)
        # Class 1: identical (perfect). Class 2: both empty (excluded).
        prediction[0:2, 0:2, 0:2] = 1
        target[0:2, 0:2, 0:2] = 1

        case = summarize_case("case_a", prediction, target, num_classes=3, spacing=(1.0, 1.0, 1.0))
        self.assertEqual(case["per_class_status"]["1"], "ok")
        self.assertEqual(case["per_class_status"]["2"], "both_empty")
        self.assertAlmostEqual(case["mean_foreground_dice"], 1.0)
        self.assertAlmostEqual(case["mean_foreground_hd95"], 0.0)
        self.assertAlmostEqual(case["mean_foreground_assd"], 0.0)


class AggregateSummaryTests(unittest.TestCase):
    def test_missed_and_evaluated_counts_and_means(self):
        shape = (6, 6, 6)

        def make_case(name, pred_present, truth_present):
            prediction = np.zeros(shape, dtype=np.int64)
            target = np.zeros(shape, dtype=np.int64)
            if pred_present:
                prediction[0:2, 0:2, 0:2] = 1
            if truth_present:
                target[0:2, 0:2, 0:2] = 1
            return summarize_case(name, prediction, target, num_classes=2, spacing=(1.0, 1.0, 1.0))

        cases = [
            make_case("ok_case", True, True),  # identical -> "ok"
            make_case("missed_case", True, False),  # one_empty -> excluded from hd95/assd mean
            make_case("absent_case", False, False),  # both_empty -> excluded from every mean
        ]
        summary = aggregate_summary(cases, num_classes=2)

        self.assertEqual(summary["num_cases"], 3)
        self.assertEqual(summary["per_class_missed_cases"]["1"], 1)
        self.assertEqual(summary["per_class_evaluated_cases"]["1"], 1)
        # Dice mean includes the ok (1.0) and missed (0.0) cases, not the absent one.
        self.assertAlmostEqual(summary["per_class_dice"]["1"], 0.5)
        # HD95/ASSD means only include the single "ok" case.
        self.assertAlmostEqual(summary["per_class_hd95"]["1"], 0.0)
        self.assertAlmostEqual(summary["per_class_assd"]["1"], 0.0)


if __name__ == "__main__":
    unittest.main()
