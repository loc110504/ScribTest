"""CPU regression checks for the EFFDNet utilities."""

import os
import random
import sys
import unittest

import torch
import torch.nn as nn

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.effdnet import (
    foreground_augmentation_diverse_context,
    foreground_background_region_labels,
    foreground_background_separation_loss,
    update_ema_variables,
)


class UpdateEmaVariablesTests(unittest.TestCase):
    def test_step_zero_copies_student_into_teacher(self):
        student = nn.Linear(2, 2)
        teacher = nn.Linear(2, 2)
        with torch.no_grad():
            student.weight.fill_(1.0)
            teacher.weight.fill_(0.0)
        update_ema_variables(student, teacher, alpha=0.99, global_step=0)
        # alpha_0 = min(1 - 1/1, 0.99) = 0 -> teacher fully replaced by student.
        torch.testing.assert_close(teacher.weight, student.weight)

    def test_large_step_uses_the_target_alpha(self):
        student = nn.Linear(2, 2)
        teacher = nn.Linear(2, 2)
        with torch.no_grad():
            student.weight.fill_(1.0)
            teacher.weight.fill_(0.0)
        update_ema_variables(student, teacher, alpha=0.9, global_step=10_000)
        expected = 0.9 * 0.0 + 0.1 * 1.0
        torch.testing.assert_close(teacher.weight, torch.full_like(teacher.weight, expected))


class ForegroundBackgroundRegionLabelsTests(unittest.TestCase):
    def test_2d_cell_with_foreground_scribble_is_true(self):
        label = torch.full((1, 8, 8), 4, dtype=torch.long)  # ignore_index=4 everywhere
        label[0, 0, 0] = 1  # a foreground scribble pixel in the top-left cell
        region = foreground_background_region_labels(label, ignore_index=4, num_regions=2)
        self.assertEqual(tuple(region.shape), (1, 2, 2))
        self.assertTrue(region[0, 0, 0].item())
        self.assertFalse(region[0, 0, 1].item())
        self.assertFalse(region[0, 1, 0].item())
        self.assertFalse(region[0, 1, 1].item())

    def test_background_only_cell_is_false(self):
        label = torch.full((1, 8, 8), 4, dtype=torch.long)
        label[0, 0, 0] = 0  # background scribble only
        region = foreground_background_region_labels(label, ignore_index=4, num_regions=2)
        self.assertFalse(region[0, 0, 0].item())

    def test_3d_cell_with_foreground_scribble_is_true(self):
        label = torch.full((1, 4, 8, 8), 8, dtype=torch.long)
        label[0, 0, 0, 0] = 2
        region = foreground_background_region_labels(label, ignore_index=8, num_regions=2)
        self.assertEqual(tuple(region.shape), (1, 2, 2, 2))
        self.assertTrue(region[0, 0, 0, 0].item())


class ForegroundBackgroundSeparationLossTests(unittest.TestCase):
    def test_finite_and_lower_for_well_separated_features(self):
        torch.manual_seed(0)
        label = torch.zeros(2, 16, 16, dtype=torch.long)
        label[:, :8, :8] = 1  # top-left quadrant of every sample is foreground

        # Well-separated: foreground region features point one way, background another.
        good_feature = torch.zeros(2, 4, 16, 16)
        good_feature[:, 0, :8, :8] = 5.0
        good_feature[:, 1, :, :] = 5.0
        good_feature[:, 1, :8, :8] = 0.0
        good_loss = foreground_background_separation_loss(good_feature, label, ignore_index=4, num_regions=2)

        random_feature = torch.randn(2, 4, 16, 16)
        random_loss = foreground_background_separation_loss(random_feature, label, ignore_index=4, num_regions=2)

        self.assertTrue(torch.isfinite(good_loss))
        self.assertTrue(torch.isfinite(random_loss))
        self.assertLess(good_loss.item(), random_loss.item())

    def test_3d_shape_runs_and_is_finite(self):
        torch.manual_seed(1)
        label = torch.zeros(2, 8, 16, 16, dtype=torch.long)
        label[:, :4, :8, :8] = 1
        feature = torch.randn(2, 4, 8, 16, 16)
        loss = foreground_background_separation_loss(feature, label, ignore_index=4, num_regions=2)
        self.assertTrue(torch.isfinite(loss))

    def test_gradients_flow_to_feature(self):
        label = torch.zeros(2, 16, 16, dtype=torch.long)
        label[:, :8, :8] = 1
        feature = torch.randn(2, 4, 16, 16, requires_grad=True)
        loss = foreground_background_separation_loss(feature, label, ignore_index=4, num_regions=2)
        loss.backward()
        self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertGreater(feature.grad.abs().sum().item(), 0.0)


class ForegroundAugmentationDiverseContextTests(unittest.TestCase):
    def test_swaps_in_a_donor_crop_resized_to_the_receivers_bbox(self):
        image = torch.zeros(2, 1, 8, 8)
        label = torch.full((2, 8, 8), 4, dtype=torch.long)
        pseudo_label = torch.zeros(2, 8, 8, dtype=torch.long)

        # Sample 0: small annotated box in the corner, filled with 1.0 / class 2.
        label[0, 0:2, 0:2] = 2
        image[0, 0, 0:2, 0:2] = 1.0
        # Sample 1: larger annotated box elsewhere, filled with 9.0 / class 3.
        label[1, 4:8, 4:8] = 3
        image[1, 0, 4:8, 4:8] = 9.0

        random_state = random.getstate()
        try:
            random.seed(0)
            out_image, out_label, out_pseudo = foreground_augmentation_diverse_context(
                image, label, pseudo_label, ignore_index=4
            )
        finally:
            random.setstate(random_state)

        # Sample 0's own bbox region must now hold a resized donor crop (either
        # its own content or sample 1's), still shaped like its own 2x2 box,
        # and everywhere outside the box stays background/ignore_index.
        self.assertEqual(tuple(out_label[0, 0:2, 0:2].shape), (2, 2))
        self.assertTrue(torch.all(out_label[0, 2:, :] == 4))
        self.assertTrue(torch.all(out_label[0, :, 2:] == 4))

    def test_sample_with_no_annotation_is_left_unchanged(self):
        image = torch.randn(1, 1, 8, 8)
        label = torch.full((1, 8, 8), 4, dtype=torch.long)  # nothing annotated anywhere
        pseudo_label = torch.zeros(1, 8, 8, dtype=torch.long)

        out_image, out_label, out_pseudo = foreground_augmentation_diverse_context(
            image, label, pseudo_label, ignore_index=4
        )
        torch.testing.assert_close(out_image, image)
        torch.testing.assert_close(out_label, label)

    def test_3d_shape_runs(self):
        image = torch.randn(2, 1, 4, 8, 8)
        label = torch.full((2, 4, 8, 8), 4, dtype=torch.long)
        label[0, 0:2, 0:2, 0:2] = 1
        label[1, 2:4, 4:8, 4:8] = 2
        pseudo_label = torch.zeros(2, 4, 8, 8, dtype=torch.long)
        out_image, out_label, out_pseudo = foreground_augmentation_diverse_context(
            image, label, pseudo_label, ignore_index=4
        )
        self.assertEqual(tuple(out_image.shape), tuple(image.shape))


if __name__ == "__main__":
    unittest.main()
