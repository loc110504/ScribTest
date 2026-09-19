"""Regression tests for train_dmpls_2d.py's non-dataset-touching pieces."""

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

from networks.unet_2d import UNetCCT2D
from train.train_dmpls_2d import checkpoint_payload, restore_checkpoint, validate_args


class ValidateArgsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC",
            patch_size=None,
            batch_size=None,
            feature_channels=(16, 32, 64, 128, 256),
            max_iterations=100,
            early_interval=10,
            late_interval=10,
            late_phase_start=50,
            num_workers=0,
            dropout_p=0.5,
            lambda_pls=0.5,
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

    def test_dropout_p_must_be_in_open_unit_interval(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(dropout_p=1.0))

    def test_lambda_pls_must_be_non_negative(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(lambda_pls=-1.0))


class CheckpointRoundTripTests(unittest.TestCase):
    def test_restore_recovers_step_score_and_weights(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32), dropout_p=0.5
        )
        model = UNetCCT2D(in_chns=1, class_num=4, feature_chns=args.feature_channels, perturbation_dropout=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        split = {"train_cases": ["patient001_ED"], "val_cases": ["patient002_ED"]}

        payload = checkpoint_payload(model, optimizer, scaler, args, split, step=123, best_score=0.42)
        self.assertEqual(payload["model_name"], "unet_cct_2d")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)

            restored_model = UNetCCT2D(in_chns=1, class_num=4, feature_chns=args.feature_channels, perturbation_dropout=0.5)
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01, momentum=0.9)
            restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
            step, best_score = restore_checkpoint(path, restored_model, restored_optimizer, restored_scaler, args, split)

        self.assertEqual(step, 123)
        self.assertAlmostEqual(best_score, 0.42)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_mismatched_dataset_is_rejected(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32), dropout_p=0.5
        )
        model = UNetCCT2D(in_chns=1, class_num=4, feature_chns=args.feature_channels, perturbation_dropout=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        split = {"train_cases": [], "val_cases": []}
        payload = checkpoint_payload(model, optimizer, scaler, args, split, step=1, best_score=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)
            mismatched_args = SimpleNamespace(
                dataset="MSCMR", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32), dropout_p=0.5
            )
            with self.assertRaisesRegex(ValueError, "dataset"):
                restore_checkpoint(path, model, optimizer, scaler, mismatched_args, split)


if __name__ == "__main__":
    unittest.main()
