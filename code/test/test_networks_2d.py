"""Small CPU regression checks for the 2D segmentation networks."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F


CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.net_factory import net_factory


class Network2DShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        # UNet2D is a faithful HiLab port (ConvTranspose2d upsampling, no skip
        # resize-alignment), so spatial dims must be divisible by 16 (4 maxpools).
        self.image = torch.randn(2, 1, 32, 48)
        self.target = torch.randint(0, 4, (2, 32, 48))
        self.target[:, 0] = 4

    def test_unet_2d(self):
        model = net_factory("unet_2d", in_chns=1, class_num=4, device="cpu")
        logits = model(self.image)
        self.assertEqual(tuple(logits.shape), (2, 4, 32, 48))
        loss = F.cross_entropy(logits, self.target, ignore_index=4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_unet_cct_2d(self):
        model = net_factory("unet_cct_2d", in_chns=1, class_num=4, device="cpu")
        outputs = model(self.image)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (2, 4, 32, 48))
        self.assertEqual(tuple(outputs[1].shape), (2, 4, 32, 48))
        logits = model(self.image, return_auxiliary=False)
        self.assertEqual(tuple(logits.shape), (2, 4, 32, 48))

    def test_unet_cct_2d_multiple_perturbations(self):
        model = net_factory(
            "unet_cct_2d",
            in_chns=1,
            class_num=4,
            device="cpu",
            perturbations=("dropout", "feature_noise", "feature_dropout"),
        )
        model.train()
        outputs = model(self.image)
        self.assertEqual(len(outputs), 4)
        for output in outputs:
            self.assertEqual(tuple(output.shape), (2, 4, 32, 48))

    def test_unet_2d_return_features(self):
        model = net_factory("unet_2d", in_chns=1, class_num=4, device="cpu")
        logits, features = model(self.image, return_features=True)
        self.assertEqual(tuple(logits.shape), (2, 4, 32, 48))
        self.assertEqual(len(features["encoder"]), 5)
        self.assertEqual(len(features["decoder"]), 4)
        # decoder[0] is the coarsest (bottleneck-adjacent) stage, decoder[-1]
        # is full input resolution -- required by SDT-Net's HiCo loss, which
        # reads feats["decoder"][0]/[-1] as the "high"/"low"-level features.
        self.assertEqual(tuple(features["decoder"][-1].shape[2:]), (32, 48))

    def test_rejects_3d_input(self):
        model = net_factory("unet_2d", in_chns=1, class_num=4, device="cpu")
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(1, 1, 4, 32, 32))


if __name__ == "__main__":
    unittest.main()
