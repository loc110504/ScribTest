"""CPU regression checks for the CycleMix 3D utilities and training step."""

import os
import sys
import unittest

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_3d import UNet3D
from train.train_cyclemix_3d import cyclemix_step
from utils.cyclemix import (
    largest_component_targets,
    mix_images,
    mix_labels,
    negative_cosine_similarity,
    occlude,
    sample_batch_cuboid_masks,
    sample_cuboid_mask,
)


class CuboidMaskTests(unittest.TestCase):
    def test_mask_respects_fraction_bounds(self):
        np.random.seed(0)
        patch_size = (8, 32, 32)
        for _ in range(20):
            mask = sample_cuboid_mask(patch_size, (0.25, 0.5))
            self.assertEqual(mask.shape, patch_size)
            self.assertTrue(mask.any())
            box_extents = [np.count_nonzero(mask.any(axis=tuple(a for a in range(3) if a != axis))) for axis in range(3)]
            for extent, axis_size in zip(box_extents, patch_size):
                self.assertLessEqual(extent, int(round(axis_size * 0.5)) + 1)
                self.assertGreaterEqual(extent, 1)

    def test_batch_masks_are_independent(self):
        torch.manual_seed(0)
        np.random.seed(0)
        masks = sample_batch_cuboid_masks(4, (8, 16, 16), (0.3, 0.6), torch.device("cpu"))
        self.assertEqual(tuple(masks.shape), (4, 1, 8, 16, 16))
        self.assertEqual(masks.dtype, torch.bool)
        # Extremely unlikely that two independently sampled boxes are identical.
        self.assertFalse(torch.equal(masks[0], masks[1]))


class MixAndOccludeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.patch_size = (4, 8, 8)
        self.image_a = torch.zeros(2, 1, *self.patch_size)
        self.image_b = torch.ones(2, 1, *self.patch_size)
        self.label_a = torch.zeros(2, *self.patch_size, dtype=torch.long)
        self.label_b = torch.full((2, *self.patch_size), 2, dtype=torch.long)
        self.mask = torch.zeros(2, 1, *self.patch_size, dtype=torch.bool)
        self.mask[:, :, :2] = True

    def test_mix_images_takes_b_inside_mask(self):
        mixed = mix_images(self.image_a, self.image_b, self.mask)
        self.assertTrue(torch.all(mixed[self.mask] == 1.0))
        self.assertTrue(torch.all(mixed[~self.mask] == 0.0))

    def test_mix_labels_takes_b_inside_mask(self):
        mixed = mix_labels(self.label_a, self.label_b, self.mask)
        mask_dhw = self.mask.squeeze(1)
        self.assertTrue(torch.all(mixed[mask_dhw] == 2))
        self.assertTrue(torch.all(mixed[~mask_dhw] == 0))

    def test_occlude_blanks_image_and_ignores_label(self):
        occluded_image, occluded_label = occlude(self.image_b, self.label_b, self.mask, ignore_index=4)
        self.assertTrue(torch.all(occluded_image[self.mask] == 0.0))
        mask_dhw = self.mask.squeeze(1)
        self.assertTrue(torch.all(occluded_label[mask_dhw] == 4))
        self.assertTrue(torch.all(occluded_label[~mask_dhw] == 2))


class NegativeCosineSimilarityTests(unittest.TestCase):
    def test_identical_vectors_give_minus_one(self):
        p = torch.rand(2, 3, 4, 4, 4).clamp_min(1e-3)
        loss = negative_cosine_similarity(p, p.clone())
        self.assertAlmostEqual(loss.item(), -1.0, places=5)

    def test_orthogonal_channels_give_zero(self):
        p = torch.zeros(1, 2, 1, 1, 1)
        q = torch.zeros(1, 2, 1, 1, 1)
        p[0, 0] = 1.0
        q[0, 1] = 1.0
        loss = negative_cosine_similarity(p, q)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            negative_cosine_similarity(torch.rand(1, 2, 2, 2, 2), torch.rand(1, 3, 2, 2, 2))


class LargestComponentTargetsTests(unittest.TestCase):
    def test_keeps_only_largest_component_per_class(self):
        # Two disjoint class-1 blobs (sizes 8 and 1) plus background elsewhere.
        argmax = np.zeros((1, 6, 6, 6), dtype=np.int64)
        argmax[0, 0:2, 0:2, 0:2] = 1  # size-8 component
        argmax[0, 5, 5, 5] = 1  # size-1 component, same class
        probs = torch.nn.functional.one_hot(torch.from_numpy(argmax), num_classes=2)
        probs = probs.permute(0, 4, 1, 2, 3).float()
        cleaned = largest_component_targets(probs)
        cleaned_argmax = cleaned.argmax(dim=1)[0].numpy()
        self.assertTrue(np.all(cleaned_argmax[0:2, 0:2, 0:2] == 1))
        self.assertEqual(cleaned_argmax[5, 5, 5], 0)  # smaller component dropped
        self.assertEqual(cleaned_argmax.sum(), 8)

    def test_no_foreground_returns_all_background(self):
        probs = torch.zeros(1, 3, 4, 4, 4)
        probs[:, 0] = 1.0
        cleaned = largest_component_targets(probs)
        self.assertTrue(torch.all(cleaned.argmax(dim=1) == 0))


class CycleMixStepTests(unittest.TestCase):
    def test_step_produces_finite_loss_and_gradients(self):
        torch.manual_seed(2)
        np.random.seed(2)

        class Args:
            patch_size = (16, 32, 32)
            mix_frac_low = 0.3
            mix_frac_high = 0.6
            occlusion_frac_low = 0.1
            occlusion_frac_high = 0.2
            lambda_unmix = 1.0
            lambda_mix = 1.0
            lambda_con_global = 0.05
            lambda_con_local = 1.0

        model = UNet3D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        device = torch.device("cpu")
        image = torch.randn(2, 1, *Args.patch_size)
        target = torch.randint(0, 4, (2, *Args.patch_size))
        target[:, 0, 0, 0] = 4  # a couple of ignored (unlabeled) voxels

        loss, components = cyclemix_step(model, image, target, ignore_index=4, args=Args(), device=device)
        self.assertTrue(torch.isfinite(loss))
        for key in ("unmix", "mix", "con_global", "con_local", "labeled_voxels"):
            self.assertIn(key, components)

        loss.backward()
        grad_norm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(grad_norm, 0.0)

    def test_batch_size_one_is_rejected_by_validate_args(self):
        from train.train_cyclemix_3d import validate_args

        class Args:
            dataset = "ACDC"
            batch_size = 1
            patch_size = None
            feature_channels = [16, 32, 64, 128, 256]
            max_iterations = 100
            eval_every = 10
            save_every = 10
            num_workers = 0
            val_overlap = 0.5
            foreground_crop_prob = 1.0
            sw_batch_size = 1
            max_accumulator_mb = 1024
            temp_dir = None
            mix_frac_low = 0.3
            mix_frac_high = 0.6
            occlusion_frac_low = 0.1
            occlusion_frac_high = 0.2

        with self.assertRaisesRegex(ValueError, "batch_size must be >= 2"):
            validate_args(Args())


if __name__ == "__main__":
    unittest.main()
