"""CPU regression checks for Bayes-WSS's CVAE network and losses."""

import argparse
import os
import sys
import unittest

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.bayes_wss_2d import BayesCVAE2D
from utils.bayes_wss import (
    bayes_kl_loss,
    bayes_mean_softmax,
    bayes_pce_loss,
    bayes_reconstruction_loss,
    bayes_wss_cvae_step,
    local_dense_crf_loss,
    merge_pseudo_labels,
)


def _small_cvae(sample_time=1):
    return BayesCVAE2D(
        in_chns=1, class_num=4, patch_size=(32, 32), feature_chns=(4, 8, 16, 32, 64), latent_dim=8
    )


class BayesCVAE2DTests(unittest.TestCase):
    def test_forward_shapes(self):
        torch.manual_seed(0)
        model = _small_cvae()
        image = torch.rand(2, 1, 32, 32)
        mu, log_var, x_gen, y_logits = model(image, sample_time=3)
        self.assertEqual(tuple(mu.shape), (2, 8))
        self.assertEqual(tuple(log_var.shape), (2, 8))
        self.assertEqual(tuple(x_gen.shape), (3, 2, 1, 32, 32))
        self.assertEqual(tuple(y_logits.shape), (3, 2, 4, 32, 32))

    def test_sample_time_one_matches_batch(self):
        torch.manual_seed(0)
        model = _small_cvae()
        image = torch.rand(2, 1, 32, 32)
        mu, log_var, x_gen, y_logits = model(image, sample_time=1)
        self.assertEqual(tuple(x_gen.shape), (1, 2, 1, 32, 32))
        self.assertEqual(tuple(y_logits.shape), (1, 2, 4, 32, 32))

    def test_rejects_wrong_input_channels(self):
        model = _small_cvae()
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 32, 32), sample_time=1)


class BayesLossTests(unittest.TestCase):
    def test_kl_loss_zero_for_standard_normal(self):
        mu = torch.zeros(2, 8)
        log_var = torch.zeros(2, 8)
        self.assertAlmostEqual(bayes_kl_loss(mu, log_var).item(), 0.0, places=5)

    def test_reconstruction_loss_zero_when_matching(self):
        image = torch.rand(2, 1, 8, 8)
        x_gen = image.unsqueeze(0).repeat(3, 1, 1, 1, 1)
        self.assertAlmostEqual(bayes_reconstruction_loss(x_gen, image).item(), 0.0, places=6)

    def test_mean_softmax_and_pce_finite(self):
        torch.manual_seed(0)
        y_logits = torch.randn(3, 2, 4, 8, 8)
        mean_probs = bayes_mean_softmax(y_logits)
        self.assertEqual(tuple(mean_probs.shape), (2, 4, 8, 8))
        target = torch.randint(0, 4, (2, 8, 8))
        target[:, :2, :2] = 4
        loss = bayes_pce_loss(mean_probs, target, ignore_index=4)
        self.assertTrue(torch.isfinite(loss))

    def test_local_dense_crf_loss_finite_and_scales_with_weight(self):
        torch.manual_seed(0)
        image = torch.rand(2, 1, 12, 12)
        probs = torch.rand(2, 4, 12, 12).softmax(dim=1)
        loss_1 = local_dense_crf_loss(image, probs, weight=1.0, radius=3)
        loss_2 = local_dense_crf_loss(image, probs, weight=2.0, radius=3)
        self.assertTrue(torch.isfinite(loss_1))
        self.assertAlmostEqual((2 * loss_1).item(), loss_2.item(), places=5)

    def test_merge_pseudo_labels_keeps_scribble_and_fills_unlabeled(self):
        ignore_index = 4
        scribble = torch.tensor([[0, ignore_index], [ignore_index, 2]])
        pseudo = torch.tensor([[3, 1], [2, 3]])
        merged = merge_pseudo_labels(scribble, pseudo, ignore_index)
        expected = torch.tensor([[0, 1], [2, 2]])
        self.assertTrue(torch.equal(merged, expected))

    def test_cvae_step_backprops(self):
        torch.manual_seed(0)
        model = _small_cvae()
        image = torch.rand(2, 1, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        args = argparse.Namespace(
            sample_time=2, crf=1e-6, crf_sigma_rgb=15.0, crf_sigma_xy=5.0, crf_radius=3, recon=0.1, kl=1e-3
        )
        loss, components = bayes_wss_cvae_step(model, image, target, ignore_index=4, args=args)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(next(model.parameters()).grad)
        for key in ("pce", "crf", "recon", "kl"):
            self.assertIn(key, components)


if __name__ == "__main__":
    unittest.main()
