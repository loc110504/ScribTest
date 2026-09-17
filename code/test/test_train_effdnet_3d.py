"""Regression tests for train_effdnet_3d.py's step function and arg validation."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from networks.vnet_3d import VNet3D
from train.train_effdnet_3d import effdnet_step, validate_args


class EffdnetStepTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            use_fbsl=True, use_fadc=True, lambda_value=0.6, delta=0.3, num_regions=2, fbsl_temperature=0.07
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _fresh_models(self):
        torch.manual_seed(0)
        model = VNet3D(in_chns=1, class_num=4, n_filters=4)
        ema_model = VNet3D(in_chns=1, class_num=4, n_filters=4)
        for p in ema_model.parameters():
            p.detach_()
        return model, ema_model

    def test_full_step_finite_loss_and_gradients_only_on_student(self):
        model, ema_model = self._fresh_models()
        image = torch.randn(2, 1, 32, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32, 32))
        target[:, 0, 0, 0] = 4  # a couple of ignored voxels

        loss, components = effdnet_step(model, ema_model, image, target, ignore_index=4, args=self._args())
        self.assertTrue(torch.isfinite(loss))
        for key in ("scribble", "pseudo", "labeled_voxels", "fbsl", "scribble_aug", "pseudo_aug"):
            self.assertIn(key, components)

        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)
        self.assertTrue(all(p.grad is None for p in ema_model.parameters()))

    def test_step_without_fbsl_or_fadc(self):
        model, ema_model = self._fresh_models()
        image = torch.randn(1, 1, 32, 32, 32)
        target = torch.randint(0, 4, (1, 32, 32, 32))

        loss, components = effdnet_step(
            model, ema_model, image, target, ignore_index=4, args=self._args(use_fbsl=False, use_fadc=False)
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("fbsl", components)
        self.assertNotIn("scribble_aug", components)
        loss.backward()


    def test_2d_step_produces_finite_loss_and_gradients(self):
        # effdnet_step reused verbatim by train_effdnet_2d.py: same function,
        # UNet2D models and 2D-shaped tensors.
        torch.manual_seed(1)
        model = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        ema_model = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        for p in ema_model.parameters():
            p.detach_()
        image = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        target[:, 0, 0] = 4

        loss, components = effdnet_step(model, ema_model, image, target, ignore_index=4, args=self._args())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)


class ValidateArgsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="WORD", patch_size=None, batch_size=None, n_filters=16, max_iterations=100,
            early_interval=10, late_interval=10, late_phase_start=50, num_workers=0, val_overlap=0.5,
            foreground_crop_prob=0.0, sw_batch_size=1, max_accumulator_mb=1024, temp_dir=None,
            ema_alpha=0.99, lambda_value=0.6, delta=0.3, num_regions=8,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_defaults_are_filled_in(self):
        args = validate_args(self._args())
        self.assertEqual(args.patch_size, (64, 96, 96))
        self.assertEqual(args.batch_size, 1)

    def test_rejects_bad_ema_alpha(self):
        with self.assertRaises(ValueError):
            validate_args(self._args(ema_alpha=1.5))


if __name__ == "__main__":
    unittest.main()
