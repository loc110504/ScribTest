"""Regression tests for train_voxtrust3d_2d.py's dataset/step wiring."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from train.train_voxtrust3d_2d import VoxTrustSlice2DDataset, validate_args
from train.train_voxtrust3d_3d import voxtrust_step
from utils.ramps import sigmoid_rampup
from utils.voxtrust3d import RollingCalibrationBuffer, fit_distance_bins, stable_seed, strong_intensity_augment_2d


class FakeScribbleBench2DDataset:
    """Minimal stand-in exposing exactly what VoxTrustSlice2DDataset reads
    from ScribbleBench2DDataset."""

    def __init__(self, num_classes=3, ignore_index=3, height=16, width=16):
        rng = np.random.default_rng(0)
        self.images = [rng.random((2, height, width)).astype(np.float32)]
        label = np.full((2, height, width), ignore_index, dtype=np.int64)
        label[0, 2:5, 2:5] = 1  # a scribble stroke on slice 0
        label[1, 8:11, 8:11] = 1  # a scribble stroke on slice 1
        self.labels = [label]
        self.cases = ["patient001_ED"]
        self.spacings = [np.array([5.0, 1.5, 1.5], dtype=np.float32)]
        self.is_scribble = [True]
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.slice_index = [(0, 0), (0, 1)]

    def slice_positions_for_volumes(self, volume_indices):
        wanted = set(volume_indices)
        return [pos for pos, (v, _) in enumerate(self.slice_index) if v in wanted]


class VoxTrustSlice2DDatasetTests(unittest.TestCase):
    def setUp(self):
        self.base = FakeScribbleBench2DDataset()
        self.dataset = VoxTrustSlice2DDataset(
            self.base, [0, 1], holdout_fraction=0.15, seed=7, patch_size=(16, 16)
        )

    def test_length_and_slice_ids(self):
        self.assertEqual(len(self.dataset), 2)
        self.assertEqual(self.dataset.slice_ids, ["patient001_ED_0", "patient001_ED_1"])

    def test_getitem_shapes(self):
        sample = self.dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16))
        self.assertEqual(tuple(sample["sup_label"].shape), (16, 16))
        self.assertEqual(tuple(sample["cal_label"].shape), (16, 16))
        self.assertEqual(tuple(sample["cal_block"].shape), (16, 16))
        self.assertEqual(tuple(sample["coord"].shape), (2, 16, 16))
        self.assertEqual(sample["spacing"].shape, (2,))  # (H, W), through-plane axis dropped

    def test_spacing_drops_through_plane_axis(self):
        np.testing.assert_allclose(self.dataset.spacings["patient001_ED_0"], [1.5, 1.5])

    def test_case_trees_and_per_case_partitions(self):
        trees = self.dataset.case_trees("patient001_ED_0")
        self.assertIsInstance(trees, dict)
        partitions = self.dataset.per_case_partitions()
        self.assertEqual(len(partitions), 2)


class VoxtrustStep2DIntegrationTests(unittest.TestCase):
    def test_full_step_produces_finite_loss_and_only_updates_student(self):
        torch.manual_seed(0)
        base = FakeScribbleBench2DDataset()
        dataset = VoxTrustSlice2DDataset(base, [0, 1], holdout_fraction=0.15, seed=7, patch_size=(16, 16))
        case_trees = {sid: dataset.case_trees(sid) for sid in dataset.slice_ids}
        edges_np, d_max_np = fit_distance_bins(dataset.per_case_partitions(), base.num_classes, num_strata=3)
        device = torch.device("cpu")
        edges_t = torch.from_numpy(edges_np).float().to(device)
        d_max_t = torch.from_numpy(d_max_np).float().to(device)

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema.load_state_dict(model.state_dict())
        for p in model_ema.parameters():
            p.requires_grad_(False)

        calibrator = RollingCalibrationBuffer(
            num_classes=base.num_classes, num_strata=3, buffer_size=64, block_cap=8,
            rng=np.random.default_rng(stable_seed(7, "calibrator")),
        )
        grid = np.linspace(0.0, 1.0, 21)

        class Args:
            use_strong_aug = True
            strong_brightness = 0.2
            strong_brightness_prob = 1.0
            strong_contrast = 0.2
            strong_contrast_prob = 1.0
            strong_gamma = 0.3
            strong_gamma_prob = 1.0
            strong_noise_std = 0.05
            strong_noise_prob = 1.0
            strong_blur_prob = 0.0
            strong_blur_sigma_min = 0.2
            strong_blur_sigma_max = 1.0
            calibration_min_samples = 1
            target_precision = 0.5
            wilson_delta = 0.05

        loss_scrib, loss_pl, diagnostics = voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
        )
        loss = loss_scrib + 8.0 * sigmoid_rampup(0, 10) * loss_pl
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reliability_mean", diagnostics)

        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)
        self.assertTrue(all(p.grad is None for p in model_ema.parameters()))


class ValidateArgsPatchDivisibilityTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC", patch_size=None, batch_size=None, feature_channels=(16, 32, 64, 128, 256),
            max_iterations=10, early_interval=5, late_interval=5, late_phase_start=5, num_workers=0,
            ema_decay=0.99, holdout_fraction=0.15, distance_strata=3, target_precision=0.95, wilson_delta=0.05,
            calibration_min_samples=32, calibration_buffer_size=4096, calibration_block_cap=64,
            score_grid_points=101, warmup_frac=0.1, rampup_frac=0.2, pseudo_loss_weight=8.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))


if __name__ == "__main__":
    unittest.main()
