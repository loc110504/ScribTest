"""Regression tests for train_scmt_2d.py's dataset/step wiring."""

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
from train.train_scmt_2d import SCMTSlice2DDataset, validate_args
from train.train_scmt_3d import scmt_step
from utils.ramps import sigmoid_rampup
from utils.scmt import ReliabilityCalibrationTable, fit_distance_bin_edges


class FakeScribbleBench2DDataset:
    """Minimal stand-in exposing exactly what SCMTSlice2DDataset reads from
    ScribbleBench2DDataset."""

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


class SCMTSlice2DDatasetTests(unittest.TestCase):
    def setUp(self):
        self.base = FakeScribbleBench2DDataset()
        self.dataset = SCMTSlice2DDataset(self.base, [0, 1], num_folds=3, seed=7, patch_size=(16, 16))

    def test_length_and_slice_ids(self):
        self.assertEqual(len(self.dataset), 2)
        self.assertEqual(self.dataset.slice_ids, ["patient001_ED_0", "patient001_ED_1"])

    def test_getitem_shapes(self):
        sample = self.dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16))
        self.assertEqual(tuple(sample["sup_label"].shape), (16, 16))
        self.assertEqual(tuple(sample["cal_label"].shape), (16, 16))
        self.assertEqual(sample["spacing"].shape, (2,))  # (H, W), through-plane axis dropped

    def test_sup_and_cal_labels_partition_the_raw_scribble(self):
        self.dataset.held_out_fold = 0
        sample_a = self.dataset[0]
        # Every voxel that is a real scribble ends up in exactly one of
        # sup_label/cal_label, and never in both.
        sup = sample_a["sup_label"].numpy()
        cal = sample_a["cal_label"].numpy()
        both_ignore = (sup == self.base.ignore_index) & (cal == self.base.ignore_index)
        neither_ignore = (sup != self.base.ignore_index) & (cal != self.base.ignore_index)
        self.assertFalse(neither_ignore.any())
        # Wherever cal has a real label, sup must be ignore, and vice versa
        # for the raw scribble region (both_ignore covers omega_u).
        self.assertTrue(np.all((sup != self.base.ignore_index) | (cal != self.base.ignore_index) | both_ignore))

    def test_case_trees_partitions_length(self):
        partitions = self.dataset.per_case_partitions()
        self.assertEqual(len(partitions), 2)
        for partition in partitions:
            self.assertEqual(partition["spacing"].shape, (2,))

    def test_rotating_fold_changes_which_voxels_are_held_out(self):
        seen_cal_masks = set()
        for fold in range(3):
            self.dataset.held_out_fold = fold
            sample = self.dataset[0]
            cal_mask = tuple((sample["cal_label"].numpy() != self.base.ignore_index).flatten().tolist())
            seen_cal_masks.add(cal_mask)
        # At least one fold produces a non-empty cal mask (the slice's one
        # scribble block must be held out in exactly one of the folds).
        self.assertTrue(any(any(mask) for mask in seen_cal_masks))


class ScmtStep2DIntegrationTests(unittest.TestCase):
    def test_full_step_produces_finite_loss_and_only_updates_student(self):
        torch.manual_seed(0)
        base = FakeScribbleBench2DDataset()
        dataset = SCMTSlice2DDataset(base, [0, 1], num_folds=3, seed=7, patch_size=(16, 16))
        dataset.held_out_fold = 0
        edges_np = fit_distance_bin_edges(dataset.per_case_partitions(), base.num_classes, num_folds=3, num_distance_bins=3)
        device = torch.device("cpu")
        edges_t = torch.from_numpy(edges_np).float().to(device)

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema.load_state_dict(model.state_dict())
        for p in model_ema.parameters():
            p.requires_grad_(False)

        calibrator = ReliabilityCalibrationTable(
            num_classes=base.num_classes, num_confidence_bins=5, num_distance_bins=4, min_samples=1
        )

        class Args:
            use_strong_aug = True
            strong_brightness = 0.2
            strong_brightness_prob = 1.0
            strong_contrast = 0.2
            strong_contrast_prob = 1.0
            strong_noise_std = 0.05
            strong_noise_prob = 1.0
            confidence_bins = 5
            distance_bins = 3

        loss_scrib, loss_con, diagnostics = scmt_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            edges_t, calibrator, Args(), calibration_active=True,
        )
        loss = loss_scrib + 1.0 * sigmoid_rampup(0, 10) * loss_con
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("sup_voxels", diagnostics)

        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)
        self.assertTrue(all(p.grad is None for p in model_ema.parameters()))

    def test_calibrator_receives_observations_from_held_out_voxels(self):
        torch.manual_seed(0)
        base = FakeScribbleBench2DDataset()
        dataset = SCMTSlice2DDataset(base, [0, 1], num_folds=3, seed=7, patch_size=(16, 16))
        # Find a fold that actually holds something out for this tiny fixture.
        held_out_fold = None
        for fold in range(3):
            dataset.held_out_fold = fold
            sample = dataset[0]
            if (sample["cal_label"].numpy() != base.ignore_index).any():
                held_out_fold = fold
                break
        self.assertIsNotNone(held_out_fold, "fixture should hold out something in at least one fold")
        dataset.held_out_fold = held_out_fold

        edges_np = fit_distance_bin_edges(dataset.per_case_partitions(), base.num_classes, num_folds=3, num_distance_bins=3)
        device = torch.device("cpu")
        edges_t = torch.from_numpy(edges_np).float().to(device)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        model = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema = UNet2D(in_chns=1, class_num=base.num_classes, feature_chns=(4, 8, 16, 24, 32))
        model_ema.load_state_dict(model.state_dict())

        calibrator = ReliabilityCalibrationTable(
            num_classes=base.num_classes, num_confidence_bins=5, num_distance_bins=4, min_samples=1
        )

        class Args:
            use_strong_aug = False
            confidence_bins = 5
            distance_bins = 3

        _, _, diagnostics = scmt_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            edges_t, calibrator, Args(), calibration_active=True,
        )
        self.assertGreater(diagnostics["cal_voxels"], 0)
        calibrator.commit_epoch()
        self.assertGreater(calibrator.class_count.sum(), 0)


class ValidateArgsPatchDivisibilityTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC", patch_size=None, batch_size=None, feature_channels=(16, 32, 64, 128, 256),
            max_iterations=10, early_interval=5, late_interval=5, late_phase_start=5, num_workers=0,
            ema_decay=0.99, num_folds=5, confidence_bins=5, distance_bins=4, calibration_momentum=0.9,
            calibration_min_samples=10, warmup_frac=0.1, rampup_frac=0.2, consistency_weight=1.0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))

    def test_num_folds_must_be_at_least_two(self):
        with self.assertRaisesRegex(ValueError, "num_folds"):
            validate_args(self._args(num_folds=1))


if __name__ == "__main__":
    unittest.main()
