"""Regression tests for compute_holdout_pixel_fraction.py's
realized_holdout_pixel_fraction (Table 3's "Held-out pixels (%)" column)."""

import os
import sys
import unittest

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from compute_holdout_pixel_fraction import realized_holdout_pixel_fraction  # noqa: E402
from utils.voxtrust3d import spatially_blocked_partition, stable_seed  # noqa: E402


class FakeScribbleBench2DDataset:
    """Minimal stand-in exposing exactly what realized_holdout_pixel_fraction
    reads from a real ScribbleBench2DDataset."""

    def __init__(self, labels, ignore_index, num_classes):
        self.labels = [labels]  # one "volume" holding every slice
        self.cases = ["patient001_ED"]
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.slice_index = [(0, slice_index) for slice_index in range(labels.shape[0])]


class RealizedHoldoutPixelFractionTests(unittest.TestCase):
    def test_zero_holdout_fraction_holds_out_nothing(self):
        label = np.full((1, 10, 10), 2, dtype=np.int64)  # ignore_index=2 everywhere
        label[0, 0:4, 0:4] = 1  # one 16-pixel block
        label[0, 6:10, 6:10] = 1  # a second, disjoint 16-pixel block, same class
        dataset = FakeScribbleBench2DDataset(label, ignore_index=2, num_classes=2)

        fraction, total, held = realized_holdout_pixel_fraction(dataset, [0], holdout_fraction=0.0, seed=2026)
        self.assertEqual(fraction, 0.0)
        self.assertEqual(held, 0)
        self.assertEqual(total, 32)

    def test_matches_direct_partition_accounting_across_slices_and_classes(self):
        # Two slices, two classes, unequal block sizes -- exercises pooling
        # across both axes, not just a single class/slice.
        label = np.full((2, 12, 12), 3, dtype=np.int64)  # ignore_index=3
        label[0, 0:2, 0:2] = 1  # class 1, slice 0: a 4-pixel block
        label[0, 8:11, 8:11] = 1  # class 1, slice 0: a second, 9-pixel block
        label[1, 0:3, 0:3] = 2  # class 2, slice 1: a 9-pixel block
        label[1, 9:11, 9:11] = 2  # class 2, slice 1: a second, 4-pixel block
        dataset = FakeScribbleBench2DDataset(label, ignore_index=3, num_classes=4)
        holdout_fraction, seed = 0.5, 123

        fraction, total, held = realized_holdout_pixel_fraction(dataset, [0, 1], holdout_fraction, seed)

        # Independently recompute the same pooled totals by calling the
        # partition function directly, exactly as the training pipeline does.
        expected_total = expected_held = 0
        for slice_index in (0, 1):
            slice_label = label[slice_index]
            rng = np.random.default_rng(stable_seed(seed, "patient001_ED_{}".format(slice_index)))
            sup_coords, cal_coords, _ = spatially_blocked_partition(slice_label, 3, 4, holdout_fraction, rng)
            for class_id in range(4):
                expected_total += len(sup_coords[class_id]) + len(cal_coords[class_id])
                expected_held += len(cal_coords[class_id])

        self.assertEqual(total, expected_total)
        self.assertEqual(held, expected_held)
        self.assertAlmostEqual(fraction, expected_held / expected_total)

    def test_raises_when_no_annotated_pixels_exist(self):
        label = np.full((1, 8, 8), 5, dtype=np.int64)  # entirely ignore_index
        dataset = FakeScribbleBench2DDataset(label, ignore_index=5, num_classes=3)
        with self.assertRaises(RuntimeError):
            realized_holdout_pixel_fraction(dataset, [0], holdout_fraction=0.15, seed=2026)


if __name__ == "__main__":
    unittest.main()
