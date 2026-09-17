"""Regression tests for train_dmsps_2d.py's dataset-wiring pieces."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNetCCT2D
from train.train_dmsps_2d import ExpandedLabel2DDataset, build_stage2_labels, validate_args


class FakeScribbleBench2DDataset:
    """Minimal stand-in exposing exactly what ExpandedLabel2DDataset/
    build_stage2_labels read from ScribbleBench2DDataset."""

    def __init__(self):
        self.images = [np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)]
        self.labels = [np.full((3, 8, 8), 4, dtype=np.int64)]  # all ignore_index
        self.cases = ["patient001_ED"]
        self.spacings = [np.array([5.0, 1.5, 1.5], dtype=np.float32)]
        self.is_scribble = [True]
        self.num_classes = 3
        self.ignore_index = 4
        self.slice_index = [(0, 0), (0, 1), (0, 2)]

    def __len__(self):
        return len(self.slice_index)


class ExpandedLabel2DDatasetTests(unittest.TestCase):
    def test_swaps_label_only_for_cases_with_an_expansion(self):
        base = FakeScribbleBench2DDataset()
        expanded = {"patient001_ED": np.zeros((3, 8, 8), dtype=np.int64)}  # all class 0
        identity_transform = lambda sample: sample  # noqa: E731

        dataset = ExpandedLabel2DDataset(base, expanded, identity_transform)
        sample = dataset[0]
        self.assertTrue(np.all(sample["label"] == 0))  # expanded label used, not the all-ignore raw one

    def test_leaves_label_untouched_for_cases_without_an_expansion(self):
        base = FakeScribbleBench2DDataset()
        identity_transform = lambda sample: sample  # noqa: E731

        dataset = ExpandedLabel2DDataset(base, {}, identity_transform)
        sample = dataset[0]
        self.assertTrue(np.all(sample["label"] == 4))  # untouched raw scribble (all ignore_index here)


class BuildStage2LabelsTests(unittest.TestCase):
    def test_one_expanded_entry_per_requested_case(self):
        base = FakeScribbleBench2DDataset()
        model = UNetCCT2D(in_chns=1, class_num=3, feature_chns=(4, 8, 16, 24, 32))
        model.eval()
        args = SimpleNamespace(patch_size=(16, 16), amp=False, tau=0.5)

        expanded = build_stage2_labels(model, base, [0], ignore_index=4, args=args, device=torch.device("cpu"))
        self.assertEqual(set(expanded.keys()), {"patient001_ED"})
        self.assertEqual(expanded["patient001_ED"].shape, (3, 8, 8))


class ValidateArgsStageTests(unittest.TestCase):
    def test_stage2_requires_init_checkpoint(self):
        args = SimpleNamespace(
            dataset="ACDC", stage=2, init_checkpoint=None, patch_size=None, batch_size=None,
            feature_channels=(16, 32, 64, 128, 256), max_iterations=10, early_interval=5,
            late_interval=5, late_phase_start=5, num_workers=0, dropout_p=0.5, tau=None,
        )
        with self.assertRaisesRegex(ValueError, "init_checkpoint"):
            validate_args(args)


if __name__ == "__main__":
    unittest.main()
