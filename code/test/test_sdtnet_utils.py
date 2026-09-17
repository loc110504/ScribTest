"""CPU regression checks for the SDT-Net 3D utilities and training step."""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from networks.unet_3d import UNet3D
from train.train_sdtnet_3d import sdtnet_step
from utils.sdtnet import TeacherEMA, feature_consistency_loss, pick_reliable_pixels, soft_dice_loss


def _reference_refine_high_confidence(probs, threshold):
    """Direct transcription of utils/pick_reliable_pixels.py::refine_high_confidence
    for a 4-class map, used only to prove pick_reliable_pixels generalizes it
    correctly (not reused in the implementation itself)."""
    pred_class = torch.argmax(probs, dim=1)
    masks = []
    for class_id in range(4):
        prob_c = probs[:, class_id]
        fill = 5 if class_id == 0 else class_id
        masks.append(torch.where((prob_c > threshold) & (pred_class == class_id), fill, 0))
    stacked = sum(masks)
    high_conf = (stacked > 0).int()
    low_conf = 1 - high_conf
    refined = stacked + 4 * low_conf
    refined[refined == 5] = 0
    return refined


class PickReliablePixelsTests(unittest.TestCase):
    def test_matches_hardcoded_4class_reference(self):
        torch.manual_seed(0)
        probs = torch.softmax(torch.randn(3, 4, 5, 6, 6), dim=1)
        expected = _reference_refine_high_confidence(probs, threshold=0.5)
        actual = pick_reliable_pixels(probs, threshold=0.5, ignore_index=4)
        self.assertTrue(torch.equal(actual, expected))

    def test_uses_strict_greater_than(self):
        probs = torch.zeros(1, 2, 1, 1, 1)
        probs[0, 1, 0, 0, 0] = 0.5  # exactly at threshold
        probs[0, 0, 0, 0, 0] = 0.5
        result = pick_reliable_pixels(probs, threshold=0.5, ignore_index=2)
        self.assertEqual(result.item(), 2)  # not strictly greater -> ignored

    def test_confident_pixel_kept(self):
        probs = torch.zeros(1, 3, 1, 1, 1)
        probs[0, 1, 0, 0, 0] = 0.9
        probs[0, 0, 0, 0, 0] = 0.05
        probs[0, 2, 0, 0, 0] = 0.05
        result = pick_reliable_pixels(probs, threshold=0.5, ignore_index=3)
        self.assertEqual(result.item(), 1)


class FeatureConsistencyLossTests(unittest.TestCase):
    def test_identical_features_give_zero_loss(self):
        feat = torch.randn(2, 8, 4, 4, 4)
        loss = feature_consistency_loss(feat, feat.clone())
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            feature_consistency_loss(torch.randn(1, 2, 2, 2, 2), torch.randn(1, 3, 2, 2, 2))

    def test_opposite_features_give_positive_loss(self):
        feat = torch.randn(2, 8, 4, 4, 4)
        loss = feature_consistency_loss(feat, -feat)
        self.assertGreater(loss.item(), 0.5)


class SoftDiceLossTests(unittest.TestCase):
    def test_perfect_prediction_gives_near_zero_loss(self):
        target = torch.zeros(2, 4, 4, 4, dtype=torch.long)
        target[:, 0:2, 0:2, 0:2] = 1
        probs = F.one_hot(target, num_classes=3).permute(0, 4, 1, 2, 3).float()
        loss = soft_dice_loss(probs, target, num_classes=3, ignore_index=3)
        self.assertLess(loss.item(), 1e-3)

    def test_ignored_voxels_are_excluded_not_mixed_across_batch(self):
        # Two very different per-sample label maps; loss must equal the mean
        # of two INDEPENDENT per-sample dice losses (no cross-batch mixing).
        target = torch.zeros(2, 4, 4, 4, dtype=torch.long)
        target[0, 0:2, 0:2, 0:2] = 1
        target[1, :, :, :] = 2
        probs = F.one_hot(target, num_classes=3).permute(0, 4, 1, 2, 3).float()
        batched_loss = soft_dice_loss(probs, target, num_classes=3, ignore_index=3).item()

        per_sample_losses = [
            soft_dice_loss(probs[i : i + 1], target[i : i + 1], num_classes=3, ignore_index=3).item()
            for i in range(2)
        ]
        self.assertAlmostEqual(batched_loss, sum(per_sample_losses) / 2, places=4)

    def test_all_ignored_gives_finite_loss(self):
        target = torch.full((1, 4, 4, 4), fill_value=3, dtype=torch.long)
        probs = torch.softmax(torch.randn(1, 3, 4, 4, 4), dim=1)
        loss = soft_dice_loss(probs, target, num_classes=3, ignore_index=3)
        self.assertTrue(torch.isfinite(loss))

    def test_2d_perfect_prediction_gives_near_zero_loss(self):
        # Same scenario as the 3D case, one spatial rank down: probs/target
        # are [B,C,H,W]/[B,H,W], the ACDC/MSCMR 2D slice pipeline's shape.
        target = torch.zeros(2, 4, 4, dtype=torch.long)
        target[:, 0:2, 0:2] = 1
        probs = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
        loss = soft_dice_loss(probs, target, num_classes=3, ignore_index=3)
        self.assertLess(loss.item(), 1e-3)

    def test_rejects_unsupported_rank(self):
        with self.assertRaises(ValueError):
            soft_dice_loss(torch.rand(1, 3, 4), torch.zeros(1, 4, dtype=torch.long), 3, 3)


class TeacherEMATests(unittest.TestCase):
    def test_step_moves_teacher_toward_student_and_decays_student(self):
        student = UNet3D(in_chns=1, class_num=3, feature_chns=(4, 8, 16, 24, 32))
        teacher = UNet3D(in_chns=1, class_num=3, feature_chns=(4, 8, 16, 24, 32))
        student_before = {k: v.clone() for k, v in student.state_dict().items()}
        teacher_before = {k: v.clone() for k, v in teacher.state_dict().items()}

        ema = TeacherEMA(student, teacher, alpha=0.9, student_weight_decay=0.1)
        ema.step()

        for key, teacher_param in teacher.state_dict().items():
            expected = 0.9 * teacher_before[key] + 0.1 * student_before[key]
            torch.testing.assert_close(teacher_param, expected)
        for key, student_param in student.state_dict().items():
            expected = student_before[key] * 0.9
            torch.testing.assert_close(student_param, expected)


class SDTNetStepTests(unittest.TestCase):
    def test_step_produces_finite_loss_and_gradients(self):
        torch.manual_seed(2)

        class Args:
            confidence_threshold = 0.5

        student = UNet3D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        teacher1 = UNet3D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        teacher2 = UNet3D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        image = torch.randn(2, 1, 16, 32, 32)
        target = torch.randint(0, 4, (2, 16, 32, 32))
        target[:, 0, 0, 0] = 4

        loss, selected, components = sdtnet_step(
            student, teacher1, teacher2, image, target, ignore_index=4, num_classes=4, args=Args()
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn(selected, (1, 2))
        for key in ("scribble", "pseudo", "hico_low", "hico_high", "labeled_voxels", "pseudo_voxels"):
            self.assertIn(key, components)

        loss.backward()
        student_grad_norm = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
        self.assertGreater(student_grad_norm, 0.0)
        for teacher in (teacher1, teacher2):
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_2d_step_produces_finite_loss_and_gradients(self):
        # sdtnet_step reused verbatim by train_sdtnet_2d.py: same function,
        # UNet2D models (now supporting return_features=True) and 2D tensors.
        torch.manual_seed(2)

        class Args:
            confidence_threshold = 0.5

        student = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        teacher1 = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        teacher2 = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        image = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32))
        target[:, 0, 0] = 4

        loss, selected, components = sdtnet_step(
            student, teacher1, teacher2, image, target, ignore_index=4, num_classes=4, args=Args()
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn(selected, (1, 2))

        loss.backward()
        student_grad_norm = sum(p.grad.abs().sum().item() for p in student.parameters() if p.grad is not None)
        self.assertGreater(student_grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
