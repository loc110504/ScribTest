"""CPU regression checks for the ModelMix utilities."""

import os
import sys
import unittest
from unittest import mock

import torch
import torch.nn as nn

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import Encoder2D
from utils.modelmix import (
    encoder_conv_layer_names,
    mix_invariance_loss,
    mix_one_encoder_layer,
    one_hot_scribble,
    random_rotate_image_and_label,
    rotate_back,
    sample_mix_ratio,
    soft_partial_cross_entropy,
    vicinal_regularization_loss,
)


class SampleMixRatioTests(unittest.TestCase):
    def test_shape_and_range(self):
        ratio = sample_mix_ratio(4, torch.device("cpu"))
        self.assertEqual(tuple(ratio.shape), (4, 1, 1, 1))
        self.assertTrue(torch.all((ratio >= 0) & (ratio <= 1)))


class OneHotScribbleTests(unittest.TestCase):
    def test_ignore_pixels_are_all_zero(self):
        label = torch.tensor([[0, 1], [4, 2]])
        onehot = one_hot_scribble(label.unsqueeze(0), num_classes=4, ignore_index=4)
        self.assertEqual(tuple(onehot.shape), (1, 4, 2, 2))
        self.assertTrue(torch.all(onehot[0, :, 1, 0] == 0))  # label==4 -> all-zero
        self.assertEqual(onehot[0, 0, 0, 0].item(), 1.0)  # label==0 -> one-hot class 0
        self.assertEqual(onehot[0, 1, 0, 1].item(), 1.0)  # label==1 -> one-hot class 1


class SoftPartialCrossEntropyTests(unittest.TestCase):
    def test_perfect_prediction_gives_near_zero_loss(self):
        label = torch.zeros(1, 4, 4, dtype=torch.long)
        label[0, 0:2, 0:2] = 1
        target = one_hot_scribble(label, num_classes=3, ignore_index=3)
        logits = (target * 20.0 - 10.0)  # very confident correct logits
        loss = soft_partial_cross_entropy(logits, target)
        self.assertLess(loss.item(), 1e-2)

    def test_all_unannotated_target_gives_zero_loss_not_nan(self):
        target = torch.zeros(1, 3, 4, 4)
        logits = torch.randn(1, 3, 4, 4, requires_grad=True)
        loss = soft_partial_cross_entropy(logits, target)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))


class MixInvarianceLossTests(unittest.TestCase):
    def test_zero_when_mixed_prediction_matches_the_blend_exactly(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(2, 3, 4, 4), dim=1)
        mix_ratio = torch.full((2, 1, 1, 1), 0.4)
        blended = mix_ratio * probs + (1.0 - mix_ratio) * torch.flip(probs, dims=[0])
        loss = mix_invariance_loss(blended, probs, mix_ratio)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)


class VicinalRegularizationLossTests(unittest.TestCase):
    def test_identical_predictions_give_minus_one(self):
        probs = torch.softmax(torch.randn(2, 3, 4, 4), dim=1).clamp_min(1e-3)
        loss = vicinal_regularization_loss(probs, probs.clone())
        self.assertAlmostEqual(loss.item(), -1.0, places=5)


class EncoderMixingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.encoder_a = Encoder2D(in_chns=1, feature_chns=(4, 8, 16, 24, 32))
        self.encoder_b = Encoder2D(in_chns=1, feature_chns=(4, 8, 16, 24, 32))

    def test_layer_names_are_conv_modules_and_resolvable(self):
        names = encoder_conv_layer_names(self.encoder_a)
        self.assertGreater(len(names), 0)
        for name in names:
            module = self.encoder_a.get_submodule(name)
            self.assertIsInstance(module, nn.Conv2d)

    def test_mix_only_changes_the_selected_layer(self):
        names = encoder_conv_layer_names(self.encoder_a)
        layer_name = names[0]
        mixed = mix_one_encoder_layer(self.encoder_a, self.encoder_b, layer_name, mix_ratio=0.3)

        mixed_layer = mixed.get_submodule(layer_name)
        layer_a = self.encoder_a.get_submodule(layer_name)
        layer_b = self.encoder_b.get_submodule(layer_name)
        expected_weight = 0.3 * layer_a.weight + 0.7 * layer_b.weight
        torch.testing.assert_close(mixed_layer.weight, expected_weight)
        if mixed_layer.bias is not None:
            expected_bias = 0.3 * layer_a.bias + 0.7 * layer_b.bias
            torch.testing.assert_close(mixed_layer.bias, expected_bias)

        for other_name in names[1:]:
            mixed_other = mixed.get_submodule(other_name)
            original_other = self.encoder_a.get_submodule(other_name)
            torch.testing.assert_close(mixed_other.weight, original_other.weight)

    def test_original_encoders_are_not_mutated(self):
        names = encoder_conv_layer_names(self.encoder_a)
        layer_name = names[0]
        weight_before = self.encoder_a.get_submodule(layer_name).weight.clone()
        mix_one_encoder_layer(self.encoder_a, self.encoder_b, layer_name, mix_ratio=0.3)
        torch.testing.assert_close(self.encoder_a.get_submodule(layer_name).weight, weight_before)


class RotationTests(unittest.TestCase):
    def test_shapes_are_preserved(self):
        image = torch.randn(2, 1, 16, 16)
        label = torch.randint(0, 4, (2, 16, 16))
        with mock.patch("numpy.random.uniform", return_value=37.0):
            rotated_image, rotated_label, angle = random_rotate_image_and_label(image, label)
        self.assertEqual(tuple(rotated_image.shape), tuple(image.shape))
        self.assertEqual(tuple(rotated_label.shape), tuple(label.shape))
        self.assertAlmostEqual(angle, 37.0)

    def test_90_degree_round_trip_recovers_original_exactly(self):
        # A 90-degree multiple rotation is a pure nearest-neighbor
        # permutation with no interpolation artifacts, so rotating forward
        # then back must reproduce the original tensor exactly.
        image = torch.randn(1, 1, 16, 16)
        label = torch.randint(0, 4, (1, 16, 16))
        with mock.patch("numpy.random.uniform", return_value=90.0):
            rotated_image, rotated_label, angle = random_rotate_image_and_label(image, label)
        restored = rotate_back(rotated_image, angle)
        torch.testing.assert_close(restored, image)


if __name__ == "__main__":
    unittest.main()
