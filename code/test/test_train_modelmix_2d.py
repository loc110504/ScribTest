"""Regression tests for train_modelmix_2d.py's step function and wiring."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from train.train_modelmix_2d import (
    _RestartingLoader,
    checkpoint_payload,
    modelmix_task_step,
    restore_checkpoint,
    validate_args,
)


class ModelMixTaskStepTests(unittest.TestCase):
    def _fresh_models(self):
        torch.manual_seed(0)
        model_a = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        torch.manual_seed(1)
        model_b = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        return model_a, model_b

    def test_finite_loss_and_all_components_present(self):
        model_a, model_b = self._fresh_models()
        image = torch.randn(2, 1, 32, 32)
        label = torch.randint(0, 4, (2, 32, 32))
        label[:, 0, 0] = 4  # a couple of ignored pixels

        total, components = modelmix_task_step(model_a, model_b, image, label, ignore_index=4, num_classes=4)
        self.assertTrue(torch.isfinite(total))
        for key in ("own", "image_mix_sup", "image_mix_consistency", "model_mix_sup", "vicinal_reg", "labeled_pixels"):
            self.assertIn(key, components)

    def test_gradients_flow_to_model_self_but_not_model_other(self):
        # Verified-against-source property: the model-mixup branch builds its
        # virtual encoder under torch.no_grad() (a disposable deep copy), so
        # a single modelmix_task_step(self, other, ...) call never updates
        # `other`'s real parameters -- only the caller's second, swapped call
        # (for the partner task) does that.
        model_a, model_b = self._fresh_models()
        image = torch.randn(2, 1, 32, 32)
        label = torch.randint(0, 4, (2, 32, 32))

        total, _ = modelmix_task_step(model_a, model_b, image, label, ignore_index=4, num_classes=4)
        total.backward()

        grad_a = sum(p.grad.abs().sum().item() for p in model_a.parameters() if p.grad is not None)
        self.assertGreater(grad_a, 0.0)
        self.assertTrue(all(p.grad is None for p in model_b.parameters()))

    def test_model_other_is_not_mutated(self):
        model_a, model_b = self._fresh_models()
        weight_before = next(model_b.encoder.parameters()).clone()
        image = torch.randn(2, 1, 32, 32)
        label = torch.randint(0, 4, (2, 32, 32))
        modelmix_task_step(model_a, model_b, image, label, ignore_index=4, num_classes=4)
        torch.testing.assert_close(next(model_b.encoder.parameters()), weight_before)


class ValidateArgsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            patch_size=None, batch_size=None, feature_channels=(16, 32, 64, 128, 256),
            max_iterations=100, early_interval=10, late_interval=10, late_phase_start=50, num_workers=0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_defaults_are_filled_in(self):
        args = validate_args(self._args())
        self.assertEqual(args.patch_size, (256, 256))
        self.assertEqual(args.batch_size, 24)

    def test_batch_size_one_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be >= 2"):
            validate_args(self._args(batch_size=1))

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))


class CheckpointRoundTripTests(unittest.TestCase):
    def test_restore_recovers_step_score_and_weights(self):
        args = SimpleNamespace(feature_channels=(4, 8, 16, 24, 32), root_path=None, patch_size=(64, 64))
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        split = {"train_cases": ["patient001_ED"], "val_cases": ["patient002_ED"]}

        payload = checkpoint_payload(model, args, "ACDC", split, step=42, best_score=0.5)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pth"
            torch.save(payload, path)
            restored_model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
            step, best_score = restore_checkpoint(path, restored_model, args, "ACDC", split)

        self.assertEqual(step, 42)
        self.assertAlmostEqual(best_score, 0.5)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_mismatched_dataset_is_rejected(self):
        args = SimpleNamespace(feature_channels=(4, 8, 16, 24, 32), root_path=None, patch_size=(64, 64))
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        split = {"train_cases": [], "val_cases": []}
        payload = checkpoint_payload(model, args, "ACDC", split, step=1, best_score=0.1)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "best.pth"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "dataset"):
                restore_checkpoint(path, model, args, "MSCMR", split)


class RestartingLoaderTests(unittest.TestCase):
    def test_restarts_after_exhaustion(self):
        dataset = TensorDataset(torch.arange(4))
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        restarting = _RestartingLoader(loader)

        first = restarting.next_batch()
        second = restarting.next_batch()
        third = restarting.next_batch()  # loader exhausted after 2 batches -> restarts
        self.assertEqual(tuple(first[0].tolist()), (0, 1))
        self.assertEqual(tuple(second[0].tolist()), (2, 3))
        self.assertEqual(tuple(third[0].tolist()), (0, 1))


if __name__ == "__main__":
    unittest.main()
