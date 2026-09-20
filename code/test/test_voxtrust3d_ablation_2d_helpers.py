"""Regression tests for test_voxtrust3d_ablation_2d.py's pure-logic helpers
(row-kind/config resolution, PL-Acc/PL-Cov, and probability-map resizing),
not the full dataset-touching evaluator."""

import os
import sys
import unittest

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
TEST_DIR = os.path.join(CODE_DIR, "test")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_voxtrust3d_ablation_2d import (
    _resize_prob_to_native,
    pseudo_label_accuracy_and_coverage,
    resolve_row_config,
)


class ResolveRowConfigTests(unittest.TestCase):
    def test_no_calibrator_is_plain(self):
        row_kind, cfg = resolve_row_config({"training_method": "pce"})
        self.assertEqual(row_kind, "plain")
        self.assertIsNone(cfg)

    def test_missing_calibrator_key_is_plain(self):
        row_kind, cfg = resolve_row_config({})
        self.assertEqual(row_kind, "plain")

    def test_original_voxtrust3d_reads_ablation_from_args(self):
        checkpoint = {
            "training_method": "voxtrust3d",
            "calibrator_state": {},
            "args": {"ablation": "class_only"},
        }
        row_kind, cfg = resolve_row_config(checkpoint)
        self.assertEqual(row_kind, "class_only")
        self.assertEqual(cfg["estimator"], "wilson")
        self.assertEqual(cfg["abstain_policy"], "abstain")
        self.assertEqual(cfg["signal"], "margin_agreement")

    def test_original_voxtrust3d_defaults_to_full_without_saved_ablation(self):
        checkpoint = {"training_method": "voxtrust3d", "calibrator_state": {}, "args": {}}
        row_kind, _ = resolve_row_config(checkpoint)
        self.assertEqual(row_kind, "full")

    def test_raw_ratio_knockout(self):
        checkpoint = {"training_method": "voxtrust3d_ablation_raw_ratio", "calibrator_state": {}, "args": {}}
        row_kind, cfg = resolve_row_config(checkpoint)
        self.assertEqual(row_kind, "full")
        self.assertEqual(cfg["estimator"], "raw")
        self.assertEqual(cfg["abstain_policy"], "abstain")
        self.assertEqual(cfg["signal"], "margin_agreement")

    def test_extrapolate_knockout(self):
        checkpoint = {"training_method": "voxtrust3d_ablation_extrapolate", "calibrator_state": {}, "args": {}}
        row_kind, cfg = resolve_row_config(checkpoint)
        self.assertEqual(row_kind, "full")
        self.assertEqual(cfg["estimator"], "wilson")
        self.assertEqual(cfg["abstain_policy"], "extrapolate")

    def test_top1conf_knockout(self):
        checkpoint = {"training_method": "voxtrust3d_ablation_top1conf", "calibrator_state": {}, "args": {}}
        row_kind, cfg = resolve_row_config(checkpoint)
        self.assertEqual(row_kind, "full")
        self.assertEqual(cfg["signal"], "top1_confidence")

    def test_unknown_calibrator_training_method_raises(self):
        checkpoint = {"training_method": "bogus", "calibrator_state": {}, "args": {}}
        with self.assertRaises(ValueError):
            resolve_row_config(checkpoint)


class ResizeProbToNativeTests(unittest.TestCase):
    def test_output_shape_and_sums_to_one(self):
        prob_patch = np.random.default_rng(0).dirichlet([1.0, 1.0, 1.0], size=(4, 4)).transpose(2, 0, 1)
        resized = _resize_prob_to_native(prob_patch.astype(np.float32), (8, 6))
        self.assertEqual(resized.shape, (3, 8, 6))
        np.testing.assert_allclose(resized.sum(axis=0), 1.0, atol=1e-5)

    def test_uniform_probability_stays_uniform(self):
        prob_patch = np.full((2, 4, 4), 0.5, dtype=np.float32)
        resized = _resize_prob_to_native(prob_patch, (4, 4))
        np.testing.assert_allclose(resized, 0.5, atol=1e-5)


class PseudoLabelAccuracyAndCoverageTests(unittest.TestCase):
    def test_typical_case(self):
        # 90 of 100 accepted pixels correct, out of 1000 unlabeled candidates.
        pl_acc, pl_cov = pseudo_label_accuracy_and_coverage(90, 100, 1000)
        self.assertAlmostEqual(pl_acc, 90.0)
        self.assertAlmostEqual(pl_cov, 10.0)

    def test_all_pseudo_labels_row_is_full_coverage(self):
        # Unfiltered Mean Teacher: every candidate is accepted.
        pl_acc, pl_cov = pseudo_label_accuracy_and_coverage(700, 1000, 1000)
        self.assertAlmostEqual(pl_acc, 70.0)
        self.assertAlmostEqual(pl_cov, 100.0)

    def test_nothing_accepted_gives_none_accuracy_but_zero_coverage(self):
        # Full abstention: PL-Acc is undefined (0/0), but PL-Cov is a
        # meaningful 0.0, not undefined -- there were candidates, none kept.
        pl_acc, pl_cov = pseudo_label_accuracy_and_coverage(0, 0, 1000)
        self.assertIsNone(pl_acc)
        self.assertAlmostEqual(pl_cov, 0.0)

    def test_no_candidates_gives_none_coverage(self):
        pl_acc, pl_cov = pseudo_label_accuracy_and_coverage(0, 0, 0)
        self.assertIsNone(pl_acc)
        self.assertIsNone(pl_cov)

    def test_accepted_correct_exceeding_total_raises(self):
        with self.assertRaises(ValueError):
            pseudo_label_accuracy_and_coverage(5, 3, 10)

    def test_accepted_total_exceeding_candidates_raises(self):
        with self.assertRaises(ValueError):
            pseudo_label_accuracy_and_coverage(3, 20, 10)

    def test_a_stingy_high_accuracy_rule_is_not_automatically_better(self):
        # The whole point of reporting the pair: Method A accepts almost
        # nothing but looks flawless; Method B accepts much more at a lower
        # (but still useful) accuracy. Neither dominates on PL-Acc alone.
        acc_a, cov_a = pseudo_label_accuracy_and_coverage(99, 100, 100000)
        acc_b, cov_b = pseudo_label_accuracy_and_coverage(28500, 30000, 100000)
        self.assertGreater(acc_a, acc_b)
        self.assertLess(cov_a, cov_b)


if __name__ == "__main__":
    unittest.main()
