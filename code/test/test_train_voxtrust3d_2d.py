"""Regression tests for train_voxtrust3d_2d.py's dataset/step wiring."""

import copy
import math
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
from utils.voxtrust3d import (
    RollingAccuracyBuffer,
    RollingCalibrationBuffer,
    fit_distance_bins,
    stable_seed,
    strong_intensity_augment_2d,
)


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
            ema_decay = 0.99
            ta_ema_min_scale = 0.1

        loss_scrib, loss_pl, diagnostics = voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
        )
        loss = loss_scrib + 8.0 * sigmoid_rampup(0, 10) * loss_pl
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("reliability_mean", diagnostics)
        # student_quality/teacher_quality default to None (TA-EMA disabled
        # at this call site) -> ema_alpha must fall back to the fixed rate.
        self.assertEqual(diagnostics["ema_alpha"], Args.ema_decay)

        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)
        self.assertTrue(all(p.grad is None for p in model_ema.parameters()))


class VoxtrustStep2DAblationTests(unittest.TestCase):
    """Table 2 ("Ablating Trust Calibration", paper_icassp2027/main.tex)
    component ablation arms, exercised through the shared voxtrust_step."""

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
        ema_decay = 0.99
        ta_ema_min_scale = 0.1
        global_confidence_threshold = 0.75

    def _run_step(self, holdout_fraction, ablation, global_confidence_threshold=None):
        torch.manual_seed(0)
        base = FakeScribbleBench2DDataset()
        dataset = VoxTrustSlice2DDataset(
            base, [0, 1], holdout_fraction=holdout_fraction, seed=7, patch_size=(16, 16)
        )
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

        args = self.Args()
        if global_confidence_threshold is not None:
            args.global_confidence_threshold = global_confidence_threshold

        return voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, args,
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
            ablation=ablation,
        )

    def test_default_ablation_matches_full(self):
        # Backward compatibility: a caller that never passes `ablation=`
        # (e.g. train_voxtrust3d_2d_expert.py today) must still get "full".
        loss_scrib, loss_pl, diagnostics = self._run_step(0.15, "full")
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertIn("distance_branch_ratio", diagnostics)
        self.assertIn("finite_class_only_thresholds", diagnostics)

    def test_all_pseudo_labels_accepts_every_unlabeled_pixel(self):
        loss_scrib, loss_pl, diagnostics = self._run_step(0.0, "all_pseudo_labels")
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertAlmostEqual(diagnostics["accepted_ratio"], 1.0, places=6)
        # No Omega_cal evidence was ever used, so calibration diagnostics
        # from the "full"/"class_only" branch must not appear.
        self.assertNotIn("finite_thresholds", diagnostics)
        self.assertNotIn("distance_branch_ratio", diagnostics)

    def test_global_confidence_threshold_above_one_rejects_everything(self):
        # R_i in [0, 1] by construction (reliability_score), so a threshold
        # above 1 deterministically rejects every candidate regardless of
        # the (randomly initialized) network's actual outputs.
        loss_scrib, loss_pl, diagnostics = self._run_step(
            0.0, "global_confidence", global_confidence_threshold=1.1
        )
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertEqual(loss_pl.item(), 0.0)
        self.assertAlmostEqual(diagnostics["accepted_ratio"], 0.0, places=6)

    def test_global_confidence_threshold_zero_accepts_everything(self):
        loss_scrib, loss_pl, diagnostics = self._run_step(
            0.0, "global_confidence", global_confidence_threshold=0.0
        )
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertAlmostEqual(diagnostics["accepted_ratio"], 1.0, places=6)

    def test_class_only_ablation_never_uses_distance_branch(self):
        loss_scrib, loss_pl, diagnostics = self._run_step(0.15, "class_only")
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertIn("distance_branch_ratio", diagnostics)
        self.assertAlmostEqual(diagnostics["distance_branch_ratio"], 0.0, places=6)


class FakeScribbleBench2DDatasetWithManyBlocks:
    """Like FakeScribbleBench2DDataset, but class 1 has several disjoint
    single-voxel blocks per slice instead of one -- with only one block,
    spatially_blocked_partition's "never hold out a class's last block" rule
    leaves Omega_cal permanently empty, which the plain FakeScribbleBench2DDataset
    fixture above relies on being irrelevant (it never inspects Omega_cal).
    TA-EMA needs real Omega_cal voxels to have anything to gate on."""

    def __init__(self, num_classes=3, ignore_index=3, height=16, width=16):
        rng = np.random.default_rng(0)
        self.images = [rng.random((2, height, width)).astype(np.float32)]
        label = np.full((2, height, width), ignore_index, dtype=np.int64)
        # 6 disjoint (8-connectivity) single-voxel class-1 blocks per slice.
        positions = [(1, 1), (1, 6), (1, 11), (6, 1), (6, 6), (6, 11)]
        for h, w in positions:
            label[0, h, w] = 1
            label[1, h, w] = 1
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


class VoxtrustStep2DTrustAdvantageTests(unittest.TestCase):
    """Trust-Advantage EMA gating, wired through the same 2D voxtrust_step
    call path as VoxtrustStep2DIntegrationTests above."""

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
        ema_decay = 0.99
        ta_ema_min_scale = 0.1

    def _fixture(self):
        torch.manual_seed(0)
        base = FakeScribbleBench2DDatasetWithManyBlocks()
        # holdout_fraction=0.3 over 6 blocks -> round(6*0.3)=2 held out per
        # slice (deterministic regardless of which 2 blocks the seeded rng
        # picks), so Omega_cal is guaranteed non-empty for this fixture.
        dataset = VoxTrustSlice2DDataset(base, [0, 1], holdout_fraction=0.3, seed=7, patch_size=(16, 16))
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
        return model, model_ema, batch, device, base, case_trees, edges_t, d_max_t, calibrator, grid

    def test_ema_alpha_falls_back_to_fixed_when_ta_ema_disabled(self):
        model, model_ema, batch, device, base, case_trees, edges_t, d_max_t, calibrator, grid = self._fixture()
        _, _, diagnostics = voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, self.Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
            student_quality=None, teacher_quality=None,
        )
        self.assertEqual(diagnostics["ema_alpha"], self.Args.ema_decay)
        self.assertNotIn("ta_ema_q_student", diagnostics)

    def test_ta_ema_buffers_are_untouched_before_warmup_ends(self):
        model, model_ema, batch, device, base, case_trees, edges_t, d_max_t, calibrator, grid = self._fixture()
        student_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        teacher_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        _, _, diagnostics = voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, self.Args(),
            calibration_active=False, augment_fn=strong_intensity_augment_2d,
            student_quality=student_quality, teacher_quality=teacher_quality,
        )
        self.assertEqual(diagnostics["ema_alpha"], self.Args.ema_decay)
        self.assertEqual(len(student_quality.cells), 0)
        self.assertEqual(len(teacher_quality.cells), 0)

    def test_ema_alpha_is_gated_once_calibration_is_active_with_buffers(self):
        model, model_ema, batch, device, base, case_trees, edges_t, d_max_t, calibrator, grid = self._fixture()
        student_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        teacher_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        _, _, diagnostics = voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, self.Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
            student_quality=student_quality, teacher_quality=teacher_quality,
        )
        self.assertGreater(len(student_quality.cells), 0)
        self.assertGreater(len(teacher_quality.cells), 0)
        # TA-EMA can only slow, never accelerate, the fixed baseline rate.
        self.assertGreaterEqual(diagnostics["ema_alpha"], self.Args.ema_decay - 1e-9)
        self.assertLessEqual(diagnostics["ema_alpha"], 1.0 + 1e-9)
        self.assertTrue(math.isfinite(diagnostics["ta_ema_q_student"]))
        self.assertTrue(math.isfinite(diagnostics["ta_ema_q_teacher"]))

    def test_diagnostic_forward_does_not_perturb_batchnorm_running_stats(self):
        """The student-on-weak-view forward used to score TA-EMA fairly must
        run in eval() mode -- if it ran in train() mode it would apply a
        second BatchNorm running-stat update on top of the real training
        forward's, changing the trained model's BN buffers as a side effect
        of a purely diagnostic computation."""
        model, model_ema, batch, device, base, case_trees, edges_t, d_max_t, calibrator, grid = self._fixture()
        model_ta = copy.deepcopy(model)
        model_ema_ta = copy.deepcopy(model_ema)
        calibrator_ta = copy.deepcopy(calibrator)

        torch.manual_seed(123)
        voxtrust_step(
            model, model_ema, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator, grid, self.Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
            student_quality=None, teacher_quality=None,
        )

        torch.manual_seed(123)
        student_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        teacher_quality = RollingAccuracyBuffer(base.num_classes, 3, 64)
        voxtrust_step(
            model_ta, model_ema_ta, batch, device, base.ignore_index, base.num_classes,
            case_trees, edges_t, d_max_t, calibrator_ta, grid, self.Args(),
            calibration_active=True, augment_fn=strong_intensity_augment_2d,
            student_quality=student_quality, teacher_quality=teacher_quality,
        )

        bn_pairs = [
            (module, dict(model_ta.named_modules())[name])
            for name, module in model.named_modules()
            if hasattr(module, "running_mean") and module.running_mean is not None
        ]
        self.assertGreater(len(bn_pairs), 0)
        for module, other in bn_pairs:
            torch.testing.assert_close(module.running_mean, other.running_mean)
            torch.testing.assert_close(module.running_var, other.running_var)
        self.assertTrue(model.training)
        self.assertTrue(model_ta.training)


class ValidateArgsPatchDivisibilityTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC", patch_size=None, batch_size=None, feature_channels=(16, 32, 64, 128, 256),
            max_iterations=10, early_interval=5, late_interval=5, late_phase_start=5, num_workers=0,
            ema_decay=0.99, ta_ema=1, ta_ema_min_scale=0.1,
            holdout_fraction=0.15, distance_strata=3, target_precision=0.95, wilson_delta=0.05,
            calibration_min_samples=32, calibration_buffer_size=4096, calibration_block_cap=64,
            score_grid_points=101, warmup_frac=0.1, rampup_frac=0.2, pseudo_loss_weight=8.0,
            ablation="full", global_confidence_threshold=0.75,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))

    def test_ta_ema_min_scale_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "ta_ema_min_scale"):
            validate_args(self._args(ta_ema_min_scale=1.5))
        with self.assertRaisesRegex(ValueError, "ta_ema_min_scale"):
            validate_args(self._args(ta_ema_min_scale=-0.1))

    def test_zero_holdout_fraction_is_allowed(self):
        # global_confidence/all_pseudo_labels ablations run with eta=0.
        args = validate_args(self._args(holdout_fraction=0.0, ablation="all_pseudo_labels"))
        self.assertEqual(args.holdout_fraction, 0.0)

    def test_negative_holdout_fraction_rejected(self):
        with self.assertRaisesRegex(ValueError, "holdout_fraction"):
            validate_args(self._args(holdout_fraction=-0.1))

    def test_global_confidence_threshold_out_of_range_rejected(self):
        with self.assertRaisesRegex(ValueError, "global_confidence_threshold"):
            validate_args(self._args(global_confidence_threshold=1.5))
        with self.assertRaisesRegex(ValueError, "global_confidence_threshold"):
            validate_args(self._args(global_confidence_threshold=-0.1))


if __name__ == "__main__":
    unittest.main()
