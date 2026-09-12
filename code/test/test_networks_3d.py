"""Small CPU regression checks for the volumetric segmentation networks."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F


CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.net_factory import net_factory


class NetworkShapeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.image = torch.randn(1, 1, 7, 33, 35)
        self.target = torch.randint(0, 4, (1, 7, 33, 35))
        self.target[:, 0] = 4
        self.network_kwargs = {"feature_chns": (4, 8, 16, 24, 32)}

    def _check_single_output(self, net_type):
        model = net_factory(
            net_type,
            in_chns=1,
            class_num=4,
            device="cpu",
            **self.network_kwargs
        )
        logits = model(self.image)
        self.assertEqual(tuple(logits.shape), (1, 4, 7, 33, 35))
        loss = F.cross_entropy(logits, self.target, ignore_index=4)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_unet_3d(self):
        self._check_single_output("unet_3d")

    def test_resunet_3d(self):
        self._check_single_output("resunet_3d")

    def test_nnunet_3d(self):
        self._check_single_output("nnunet_3d")

    def test_unet_cct_3d(self):
        model = net_factory(
            "unet_ctt_3d",
            in_chns=1,
            class_num=4,
            device="cpu",
            **self.network_kwargs
        )
        outputs = model(self.image)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(tuple(outputs[0].shape), (1, 4, 7, 33, 35))
        self.assertEqual(tuple(outputs[1].shape), (1, 4, 7, 33, 35))
        logits = model(self.image, return_auxiliary=False)
        self.assertEqual(tuple(logits.shape), (1, 4, 7, 33, 35))

    def test_nnunet_deep_supervision(self):
        model = net_factory(
            "nnunet_3d",
            in_chns=1,
            class_num=4,
            device="cpu",
            deep_supervision=True,
            **self.network_kwargs
        )
        outputs = model(self.image)
        self.assertEqual(len(outputs), 4)
        self.assertEqual(tuple(outputs[0].shape), (1, 4, 7, 33, 35))

    def test_rejects_2d_input(self):
        model = net_factory(
            "unet_3d",
            in_chns=1,
            class_num=4,
            device="cpu",
            **self.network_kwargs
        )
        with self.assertRaisesRegex(ValueError, "expects"):
            model(torch.randn(1, 1, 32, 32))


if __name__ == "__main__":
    unittest.main()
