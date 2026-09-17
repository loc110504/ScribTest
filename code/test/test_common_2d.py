"""Regression tests for shared 2D training infrastructure in common_2d.py."""

import os
import sys
import unittest
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from train.common_2d import predict_volume_2d, validate_2d


class ThresholdModel(nn.Module):
    """A tiny 2D "network": predicts class 1 wherever the (resized) pixel
    value is positive, class 0 otherwise -- deterministic and resolution-
    independent, so it exercises the resize-then-stitch round trip without
    needing a trained network."""

    def forward(self, x):
        positive = (x > 0).float()
        logits = torch.cat((-positive, positive), dim=1) * 10.0
        return logits


class PredictVolume2DTests(unittest.TestCase):
    def test_stitches_per_slice_predictions_back_to_native_resolution(self):
        depth, height, width = 3, 6, 8
        image = np.zeros((depth, height, width), dtype=np.float32)
        image[:, : height // 2, :] = 1.0  # top half positive -> class 1
        prediction = predict_volume_2d(
            ThresholdModel(), image, patch_size=(16, 16), device=torch.device("cpu")
        )
        self.assertEqual(prediction.shape, (depth, height, width))
        self.assertTrue(np.all(prediction[:, : height // 2, :] == 1))
        self.assertTrue(np.all(prediction[:, height // 2 :, :] == 0))


class ValidateDataset:
    """Minimal stand-in for ScribbleBench3DDataset(..., return_full_label=True)."""

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]


class Validate2DTests(unittest.TestCase):
    def test_perfect_prediction_gives_dice_one(self):
        depth, height, width = 2, 4, 4
        image = np.ones((depth, height, width), dtype=np.float32)  # all positive -> predicts class 1
        gt_label = np.ones((depth, height, width), dtype=np.int64)
        dataset = ValidateDataset([{"image": image, "gt_label": gt_label}])
        args = Namespace(patch_size=(8, 8), amp=False)

        result = validate_2d(ThresholdModel(), dataset, [0], args, torch.device("cpu"), num_classes=2)
        self.assertAlmostEqual(result["mean_dice"], 1.0, places=5)
        self.assertAlmostEqual(result["per_class_dice"]["1"], 1.0, places=5)
        self.assertEqual(result["num_cases"], 1)

    def test_disjoint_prediction_gives_dice_zero(self):
        depth, height, width = 1, 4, 4
        image = np.full((depth, height, width), -1.0, dtype=np.float32)  # predicts class 0 everywhere
        gt_label = np.ones((depth, height, width), dtype=np.int64)  # ground truth is entirely class 1
        dataset = ValidateDataset([{"image": image, "gt_label": gt_label}])
        args = Namespace(patch_size=(8, 8), amp=False)

        result = validate_2d(ThresholdModel(), dataset, [0], args, torch.device("cpu"), num_classes=2)
        self.assertAlmostEqual(result["mean_dice"], 0.0, places=5)

    def test_class_absent_from_both_is_excluded_not_zero(self):
        depth, height, width = 1, 4, 4
        image = np.ones((depth, height, width), dtype=np.float32)
        gt_label = np.ones((depth, height, width), dtype=np.int64)
        dataset = ValidateDataset([{"image": image, "gt_label": gt_label}])
        args = Namespace(patch_size=(8, 8), amp=False)

        # 3 classes: class 2 never appears in prediction or ground truth.
        class ThreeClassModel(nn.Module):
            def forward(self, x):
                positive = (x > 0).float()
                zeros = torch.zeros_like(positive)
                logits = torch.cat((-positive * 10, positive * 10, zeros - 10), dim=1)
                return logits

        result = validate_2d(ThreeClassModel(), dataset, [0], args, torch.device("cpu"), num_classes=3)
        self.assertAlmostEqual(result["per_class_dice"]["1"], 1.0, places=5)
        self.assertTrue(np.isnan(result["per_class_dice"]["2"]))
        # class 2's NaN must not drag the case-level mean down.
        self.assertAlmostEqual(result["mean_dice"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
