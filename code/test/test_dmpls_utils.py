"""CPU regression checks for DMPLS's training step (utils/dmpls.py)."""

import argparse
import os
import sys
import unittest

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNetCCT2D
from utils.dmpls import dmpls_step


class DmplsStepTests(unittest.TestCase):
    def _model_and_args(self):
        model = UNetCCT2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 32, 64))
        args = argparse.Namespace(lambda_pls=0.5)
        return model, args

    def test_returns_finite_scalar_loss_with_gradient(self):
        torch.manual_seed(0)
        model, args = self._model_and_args()
        image = torch.rand(2, 1, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16))
        target[:, :8, :8] = 4  # ignore_index
        loss, components = dmpls_step(model, image, target, ignore_index=4, args=args)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(next(model.parameters()).grad)
        for key in ("pce", "pls", "alpha", "labeled_voxels"):
            self.assertIn(key, components)

    def test_lambda_pls_zero_drops_pseudo_label_term(self):
        torch.manual_seed(0)
        model, args = self._model_and_args()
        args.lambda_pls = 0.0
        image = torch.rand(2, 1, 16, 16)
        target = torch.randint(0, 4, (2, 16, 16))
        loss, components = dmpls_step(model, image, target, ignore_index=4, args=args)
        self.assertAlmostEqual(loss.item(), components["pce"], places=5)


if __name__ == "__main__":
    unittest.main()
