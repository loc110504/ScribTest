"""CPU regression checks for the DMSPS 3D utilities and training step."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_cct_3d import UNetCCT3D
from train.train_dmsps_3d import dmsps_step
from utils.dmsps import (
    dual_branch_volume_probs,
    dynamic_mixed_pseudo_label,
    expand_labels,
    soft_pseudo_supervision_loss,
)


class DynamicMixedPseudoLabelTests(unittest.TestCase):
    def test_alpha_one_selects_main_branch(self):
        p1 = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        p2 = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        mixed = dynamic_mixed_pseudo_label(p1, p2, alpha=1.0)
        self.assertTrue(torch.allclose(mixed, p1))

    def test_alpha_zero_selects_aux_branch(self):
        p1 = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        p2 = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        mixed = dynamic_mixed_pseudo_label(p1, p2, alpha=0.0)
        self.assertTrue(torch.allclose(mixed, p2))

    def test_result_is_detached(self):
        p1 = torch.rand(1, 3, 2, 2, 2, requires_grad=True).softmax(dim=1)
        p2 = torch.rand(1, 3, 2, 2, 2, requires_grad=True).softmax(dim=1)
        mixed = dynamic_mixed_pseudo_label(p1, p2, alpha=0.5)
        self.assertFalse(mixed.requires_grad)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dynamic_mixed_pseudo_label(torch.rand(1, 2, 2, 2, 2), torch.rand(1, 3, 2, 2, 2), 0.5)


class SoftPseudoSupervisionLossTests(unittest.TestCase):
    def test_matches_double_softmax_cross_entropy_by_construction(self):
        """The official DMSPS quirk: CE is applied to already-softmaxed inputs."""
        torch.manual_seed(0)
        probs_main = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        probs_aux = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        pseudo = dynamic_mixed_pseudo_label(probs_main, probs_aux, alpha=0.5)

        actual = soft_pseudo_supervision_loss(probs_main, probs_aux, pseudo)

        def manual_soft_ce(probs, target):
            log_probs = F.log_softmax(probs, dim=1)  # log_softmax applied to already-softmaxed input
            return -(target * log_probs).sum(dim=1).mean()

        expected = 0.5 * (manual_soft_ce(probs_main, pseudo) + manual_soft_ce(probs_aux, pseudo))
        self.assertAlmostEqual(actual.item(), expected.item(), places=5)

    def test_differs_from_naive_soft_cross_entropy(self):
        """Guards against silently "fixing" the quirk to log(p) instead of log_softmax(p)."""
        torch.manual_seed(1)
        probs_main = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        probs_aux = torch.rand(1, 3, 2, 2, 2).softmax(dim=1)
        pseudo = dynamic_mixed_pseudo_label(probs_main, probs_aux, alpha=0.5)

        actual = soft_pseudo_supervision_loss(probs_main, probs_aux, pseudo)
        naive = -(pseudo * torch.log(probs_main.clamp_min(1e-8))).sum(dim=1).mean()
        self.assertGreater(abs(actual.item() - naive.item()), 1e-4)


class ExpandLabelsTests(unittest.TestCase):
    def test_keeps_only_largest_confident_component_per_class(self):
        shape = (1, 6, 6)
        scribble = np.full(shape, 4, dtype=np.int64)  # ignore_index=4, nothing scribbled
        num_classes = 4
        mean_probs = np.zeros((num_classes,) + shape, dtype=np.float32)
        mean_probs[0] = 1.0  # background everywhere by default

        # A confident, size-8 class-1 blob.
        mean_probs[:, 0, 0:2, 0:2] = 0.0
        mean_probs[1, 0, 0:2, 0:2] = 0.99
        mean_probs[0, 0, 0:2, 0:2] = 0.01
        # A confident but isolated single-voxel class-1 blob elsewhere.
        mean_probs[:, 0, 5, 5] = 0.0
        mean_probs[1, 0, 5, 5] = 0.99
        mean_probs[0, 0, 5, 5] = 0.01

        expanded = expand_labels(scribble, mean_probs, ignore_index=4, tau=0.5)
        self.assertTrue(np.all(expanded[0, 0:2, 0:2] == 1))
        self.assertEqual(expanded[0, 5, 5], 4)  # smaller component dropped, stays unlabeled

    def test_original_scribble_is_never_overwritten(self):
        shape = (1, 4, 4)
        scribble = np.full(shape, 4, dtype=np.int64)
        scribble[0, 0, 0] = 2  # an original scribble annotation
        mean_probs = np.zeros((3,) + shape, dtype=np.float32)
        mean_probs[0] = 0.01
        mean_probs[1] = 0.98  # model confidently disagrees with the scribble here
        mean_probs[2] = 0.01
        expanded = expand_labels(scribble, mean_probs, ignore_index=4, tau=0.9)
        self.assertEqual(expanded[0, 0, 0], 2)

    def test_low_confidence_stays_unlabeled(self):
        shape = (1, 4, 4)
        scribble = np.full(shape, 4, dtype=np.int64)
        mean_probs = np.full((3,) + shape, 1.0 / 3, dtype=np.float32)  # maximally uncertain
        expanded = expand_labels(scribble, mean_probs, ignore_index=4, tau=0.1)
        self.assertTrue(np.all(expanded == 4))


class DualBranchVolumeProbsTests(unittest.TestCase):
    def test_output_shape_and_probability_simplex(self):
        torch.manual_seed(0)
        model = UNetCCT3D(in_chns=1, class_num=3, feature_chns=(4, 8, 16, 24, 32))
        model.eval()
        image = torch.randn(1, 1, 16, 32, 32)
        probs = dual_branch_volume_probs(
            model, image, num_classes=3, patch_size=(16, 32, 32), device=torch.device("cpu")
        )
        self.assertEqual(probs.shape, (3, 16, 32, 32))
        sums = probs.sum(axis=0)
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-4)


class DMSPSStepTests(unittest.TestCase):
    def test_step_produces_finite_loss_and_gradients(self):
        torch.manual_seed(3)
        np.random.seed(3)

        class Args:
            lambda_sps = 8.0

        model = UNetCCT3D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        image = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 4, (2, 16, 32, 32))
        target[:, 0, 0, 0] = 4

        loss, components = dmsps_step(model, image, target, ignore_index=4, args=Args())
        self.assertTrue(torch.isfinite(loss))
        for key in ("pce", "sps", "alpha", "labeled_voxels"):
            self.assertIn(key, components)
        self.assertGreaterEqual(components["alpha"], 0.0)
        self.assertLessEqual(components["alpha"], 1.0)

        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
