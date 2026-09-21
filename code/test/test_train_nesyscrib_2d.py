"""Regression tests for train_nesyscrib_2d.py's non-dataset-touching pieces."""

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

from networks.unet_2d import UNet2D  # noqa: E402
from train.train_nesyscrib_2d import checkpoint_payload, restore_checkpoint, validate_args  # noqa: E402


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
            ema_decay=0.99,
            warmup_frac=0.1,
            rampup_frac=0.2,
            pseudo_loss_weight=1.0,
            noise_std=0.1,
            repair_sigma=5.0,
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

    def test_ema_decay_must_be_in_open_unit_interval(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(ema_decay=1.0))
        with self.assertRaises(ValueError):
            validate_args(self._args(ema_decay=0.0))

    def test_repair_sigma_must_be_positive(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(repair_sigma=0.0))

    def test_pseudo_loss_weight_may_be_zero(self):
        args = validate_args(self._args(pseudo_loss_weight=0.0))
        self.assertEqual(args.pseudo_loss_weight, 0.0)

    def test_negative_pseudo_loss_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(pseudo_loss_weight=-1.0))


class CheckpointRoundTripTests(unittest.TestCase):
    def _build(self, dataset="ACDC"):
        args = SimpleNamespace(
            dataset=dataset, root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
        )
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        model_ema = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        model_ema.load_state_dict(model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        return args, model, model_ema, optimizer, scaler

    def test_checkpoint_is_plain_unet2d_compatible_with_test_pce_2d(self):
        args, model, model_ema, optimizer, scaler = self._build()
        split = {"train_cases": ["patient001_ED"], "val_cases": ["patient002_ED"]}

        payload = checkpoint_payload(model, model_ema, optimizer, scaler, args, split, step=123, best_score=0.42)
        self.assertEqual(payload["model_name"], "unet_2d")
        self.assertEqual(payload["training_method"], "nesyscrib")
        self.assertIn("student_state_dict", payload)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)

            _, restored_model, restored_model_ema, restored_optimizer, restored_scaler = self._build()
            step, best_score = restore_checkpoint(
                path, restored_model, restored_model_ema, restored_optimizer, restored_scaler, args, split
            )

        self.assertEqual(step, 123)
        self.assertAlmostEqual(best_score, 0.42)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)
        for expected, actual in zip(model_ema.parameters(), restored_model_ema.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_mismatched_dataset_is_rejected(self):
        args, model, model_ema, optimizer, scaler = self._build(dataset="ACDC")
        split = {"train_cases": [], "val_cases": []}
        payload = checkpoint_payload(model, model_ema, optimizer, scaler, args, split, step=1, best_score=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)
            mismatched_args = SimpleNamespace(
                dataset="MSCMR", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
            )
            with self.assertRaisesRegex(ValueError, "dataset"):
                restore_checkpoint(path, model, model_ema, optimizer, scaler, mismatched_args, split)


if __name__ == "__main__":
    unittest.main()
