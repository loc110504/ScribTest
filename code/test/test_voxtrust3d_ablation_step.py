"""Regression tests for utils/voxtrust3d_ablation_step.py's
voxtrust_step_knockout, the shared step function behind the three Table 2
"w/o ..." design-choice ablation train scripts."""

import os
import sys
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from train.train_voxtrust3d_2d import VoxTrustSlice2DDataset
from utils.voxtrust3d import RollingCalibrationBuffer, fit_distance_bins, stable_seed, strong_intensity_augment_2d
from utils.voxtrust3d_ablation_step import voxtrust_step_knockout


class FakeScribbleBench2DDataset:
    """Minimal stand-in exposing exactly what VoxTrustSlice2DDataset reads
    from ScribbleBench2DDataset (mirrors test_train_voxtrust3d_2d.py's
    fixture of the same name -- redefined here, not imported, so this file
    stays runnable standalone)."""

    def __init__(self, num_classes=3, ignore_index=3, height=16, width=16):
        rng = np.random.default_rng(0)
        self.images = [rng.random((2, height, width)).astype(np.float32)]
        label = np.full((2, height, width), ignore_index, dtype=np.int64)
        label[0, 2:5, 2:5] = 1
        label[1, 8:11, 8:11] = 1
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


class VoxtrustStepKnockoutTests(unittest.TestCase):
    def _fixture(self, holdout_fraction=0.15):
        base = FakeScribbleBench2DDataset()
        dataset = VoxTrustSlice2DDataset(base, [0, 1], holdout_fraction=holdout_fraction, seed=7, patch_size=(16, 16))
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
        return dict(
            model=model, model_ema=model_ema, batch=batch, device=device, ignore_index=base.ignore_index,
            case_trees=case_trees, edges_t=edges_t, d_max_t=d_max_t, calibrator=calibrator, grid=grid,
        )

    def test_default_config_matches_full_method_and_backprops(self):
        torch.manual_seed(0)
        fixture = self._fixture()
        loss_scrib, loss_pl, diagnostics = voxtrust_step_knockout(
            fixture["model"], fixture["model_ema"], fixture["batch"], fixture["device"], fixture["ignore_index"],
            fixture["case_trees"], fixture["edges_t"], fixture["d_max_t"], fixture["calibrator"], fixture["grid"],
            Args(), calibration_active=True, augment_fn=strong_intensity_augment_2d,
        )
        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertIn("distance_branch_ratio", diagnostics)
        loss = loss_scrib + loss_pl
        loss.backward()
        student_grad = sum(p.grad.abs().sum().item() for p in fixture["model"].parameters() if p.grad is not None)
        self.assertGreater(student_grad, 0.0)

    def test_calibration_inactive_skips_pseudo_label_branch(self):
        torch.manual_seed(0)
        fixture = self._fixture()
        loss_scrib, loss_pl, diagnostics = voxtrust_step_knockout(
            fixture["model"], fixture["model_ema"], fixture["batch"], fixture["device"], fixture["ignore_index"],
            fixture["case_trees"], fixture["edges_t"], fixture["d_max_t"], fixture["calibrator"], fixture["grid"],
            Args(), calibration_active=False, augment_fn=strong_intensity_augment_2d,
        )
        self.assertEqual(loss_pl.item(), 0.0)
        self.assertNotIn("reliability_mean", diagnostics)

    def test_raw_estimator_runs_and_reports_estimator_specific_diagnostics(self):
        torch.manual_seed(0)
        fixture = self._fixture()
        loss_scrib, loss_pl, diagnostics = voxtrust_step_knockout(
            fixture["model"], fixture["model_ema"], fixture["batch"], fixture["device"], fixture["ignore_index"],
            fixture["case_trees"], fixture["edges_t"], fixture["d_max_t"], fixture["calibrator"], fixture["grid"],
            Args(), calibration_active=True, augment_fn=strong_intensity_augment_2d,
            calibration_estimator="raw",
        )
        self.assertTrue(torch.isfinite(loss_pl))
        self.assertIn("finite_thresholds", diagnostics)

    def test_extrapolate_policy_runs(self):
        torch.manual_seed(0)
        fixture = self._fixture()
        loss_scrib, loss_pl, diagnostics = voxtrust_step_knockout(
            fixture["model"], fixture["model_ema"], fixture["batch"], fixture["device"], fixture["ignore_index"],
            fixture["case_trees"], fixture["edges_t"], fixture["d_max_t"], fixture["calibrator"], fixture["grid"],
            Args(), calibration_active=True, augment_fn=strong_intensity_augment_2d,
            abstain_policy="extrapolate",
        )
        self.assertTrue(torch.isfinite(loss_pl))

    def test_top1_confidence_signal_differs_from_margin_agreement(self):
        torch.manual_seed(0)
        fixture_a = self._fixture()
        torch.manual_seed(0)
        fixture_b = self._fixture()
        _, _, diag_margin = voxtrust_step_knockout(
            fixture_a["model"], fixture_a["model_ema"], fixture_a["batch"], fixture_a["device"],
            fixture_a["ignore_index"], fixture_a["case_trees"], fixture_a["edges_t"], fixture_a["d_max_t"],
            fixture_a["calibrator"], fixture_a["grid"], Args(), calibration_active=True,
            augment_fn=strong_intensity_augment_2d, reliability_signal="margin_agreement",
        )
        _, _, diag_top1 = voxtrust_step_knockout(
            fixture_b["model"], fixture_b["model_ema"], fixture_b["batch"], fixture_b["device"],
            fixture_b["ignore_index"], fixture_b["case_trees"], fixture_b["edges_t"], fixture_b["d_max_t"],
            fixture_b["calibrator"], fixture_b["grid"], Args(), calibration_active=True,
            augment_fn=strong_intensity_augment_2d, reliability_signal="top1_confidence",
        )
        self.assertNotAlmostEqual(diag_margin["reliability_mean"], diag_top1["reliability_mean"], places=6)


if __name__ == "__main__":
    unittest.main()
