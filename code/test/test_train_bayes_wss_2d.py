"""Regression tests for train_bayes_wss_2d.py's non-dataset-touching pieces."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from train.train_bayes_wss_2d import checkpoint_payload, restore_checkpoint, validate_args


class ValidateArgsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC",
            patch_size=None,
            batch_size=None,
            feature_channels=(16, 32, 64, 128, 256),
            cvae_iterations=10,
            max_iterations=100,
            early_interval=10,
            late_interval=10,
            late_phase_start=50,
            num_workers=0,
            sample_time=5,
            latent_dim=256,
            crf_radius=5,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_defaults_are_filled_in_per_dataset(self):
        args = validate_args(self._args())
        self.assertEqual(args.patch_size, (256, 256))
        self.assertEqual(args.batch_size, 24)

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))

    def test_sample_time_must_be_positive(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(sample_time=0))

    def test_cvae_iterations_may_be_zero(self):
        args = validate_args(self._args(cvae_iterations=0))
        self.assertEqual(args.cvae_iterations, 0)


class CheckpointRoundTripTests(unittest.TestCase):
    def test_checkpoint_is_plain_unet2d_compatible_with_test_pce_2d(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
        )
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        split = {"train_cases": ["patient001_ED"], "val_cases": ["patient002_ED"]}

        payload = checkpoint_payload(model, optimizer, args, split, step=123, best_score=0.42)
        self.assertEqual(payload["model_name"], "unet_2d")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)

            restored_model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=1e-4)
            step, best_score = restore_checkpoint(path, restored_model, restored_optimizer, args, split)

        self.assertEqual(step, 123)
        self.assertAlmostEqual(best_score, 0.42)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_mismatched_dataset_is_rejected(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
        )
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        split = {"train_cases": [], "val_cases": []}
        payload = checkpoint_payload(model, optimizer, args, split, step=1, best_score=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)
            mismatched_args = SimpleNamespace(
                dataset="MSCMR", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
            )
            with self.assertRaisesRegex(ValueError, "dataset"):
                restore_checkpoint(path, model, optimizer, mismatched_args, split)


if __name__ == "__main__":
    unittest.main()
