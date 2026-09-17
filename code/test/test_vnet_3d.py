"""Small CPU regression checks for the VNet3D/VNetCCT3D backbones."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F


CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.net_factory import net_factory


class VNet3DShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        # VNet3D downsamples 4 times (16x total): spatial dims must be
        # divisible by 16, and the bottleneck extent must exceed 1 per axis
        # so BatchNorm3d has more than one value per channel at batch_size=1.
        self.image = torch.randn(1, 1, 32, 32, 32)
        self.target = torch.randint(0, 4, (1, 32, 32, 32))
        self.target[:, 0] = 4
        self.network_kwargs = {"n_filters": 4}

    def test_vnet_3d(self):
        model = net_factory("vnet_3d", in_chns=1, class_num=4, device="cpu", **self.network_kwargs)
        logits = model(self.image)
        self.assertEqual(tuple(logits.shape), (1, 4, 32, 32, 32))
        loss = F.cross_entropy(logits, self.target, ignore_index=4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_vnet_3d_no_normalization_no_dropout(self):
        model = net_factory(
            "vnet_3d",
            in_chns=1,
            class_num=4,
            device="cpu",
            n_filters=4,
            normalization="none",
            has_dropout=False,
        )
        model.eval()
        logits = model(self.image)
        self.assertEqual(tuple(logits.shape), (1, 4, 32, 32, 32))

    def test_vnet_cct_3d(self):
        model = net_factory("vnet_cct_3d", in_chns=1, class_num=4, device="cpu", **self.network_kwargs)
        outputs = model(self.image)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (1, 4, 32, 32, 32))
        self.assertEqual(tuple(outputs[1].shape), (1, 4, 32, 32, 32))
        logits = model(self.image, return_auxiliary=False)
        self.assertEqual(tuple(logits.shape), (1, 4, 32, 32, 32))

    def test_vnet_cct_3d_multiple_perturbations(self):
        model = net_factory(
            "vnet_cct_3d",
            in_chns=1,
            class_num=4,
            device="cpu",
            n_filters=4,
            perturbations=("dropout", "feature_noise", "feature_dropout"),
        )
        model.train()
        outputs = model(self.image)
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (1, 4, 32, 32, 32))

    def test_vnet_3d_return_features(self):
        model = net_factory("vnet_3d", in_chns=1, class_num=4, device="cpu", **self.network_kwargs)
        logits, features = model(self.image, return_features=True)
        self.assertEqual(tuple(logits.shape), (1, 4, 32, 32, 32))
        self.assertEqual(len(features["encoder"]), 5)
        self.assertEqual(len(features["decoder"]), 4)
        # decoder[0] is the coarsest (bottleneck-adjacent) stage, decoder[-1]
        # is full input resolution -- required by SDT-Net's HiCo loss.
        self.assertEqual(tuple(features["decoder"][-1].shape[2:]), (32, 32, 32))

    def test_rejects_2d_input(self):
        model = net_factory("vnet_3d", in_chns=1, class_num=4, device="cpu", **self.network_kwargs)
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(1, 1, 24, 24))


if __name__ == "__main__":
    unittest.main()
