"""Regression tests for the ACDC/MSCMR 2D slice dataset and augmentation."""

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataloader.scribblebench_2d import (
    RandomGenerator2D,
    ScribbleBench2DDataset,
    ToTensor2D,
)


def _write_case(root, case_id, shape_xyz=(6, 8, 3)):
    """Write a tiny synthetic ACDC-style image/scribble/dense-label triple.

    ``shape_xyz`` matches NIfTI's on-disk (X, Y, Z) axis order; after
    ScribbleBench3DDataset's (Z, Y, X) transpose this becomes a (D, H, W) =
    (3, 8, 6) volume -- 3 slices, matching a small ACDC short-axis stack.
    """
    rng = np.random.default_rng(hash(case_id) % (2**32))
    image = rng.normal(size=shape_xyz).astype(np.float32)
    dense = rng.integers(0, 4, size=shape_xyz).astype(np.int16)
    scribble = np.full(shape_xyz, 4, dtype=np.int16)  # 4 = ignore_index
    scribble[1, 1, :] = dense[1, 1, :]  # a thin scribble stroke, one per slice

    affine = np.eye(4, dtype=np.float64)
    nib.save(nib.Nifti1Image(image, affine), root / "imagesTr" / "{}_0000.nii.gz".format(case_id))
    nib.save(nib.Nifti1Image(scribble, affine), root / "labelsTr" / "{}.nii.gz".format(case_id))
    nib.save(nib.Nifti1Image(dense, affine), root / "labelsTr_dense" / "{}.nii.gz".format(case_id))
    return shape_xyz


class ScribbleBench2DDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        for sub in ("imagesTr", "labelsTr", "labelsTr_dense"):
            (self.root / sub).mkdir(parents=True)
        self.shapes = [
            _write_case(self.root, "patient001_ED"),
            _write_case(self.root, "patient001_ES"),
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_rejects_word(self):
        with self.assertRaisesRegex(ValueError, "ACDC.*MSCMR|MSCMR.*ACDC"):
            ScribbleBench2DDataset(dataset_name="WORD", base_dir=self.root)

    def test_flat_index_covers_every_slice(self):
        dataset = ScribbleBench2DDataset(dataset_name="ACDC", base_dir=self.root, split="train")
        total_slices = sum(shape[2] for shape in self.shapes)  # Z axis -> D after transpose
        self.assertEqual(len(dataset), total_slices)

    def test_sample_has_2d_shape_and_scribble_ignore_index(self):
        dataset = ScribbleBench2DDataset(dataset_name="ACDC", base_dir=self.root, split="train")
        sample = dataset[0]
        self.assertEqual(sample["image"].ndim, 2)
        self.assertEqual(sample["image"].shape, sample["label"].shape)
        self.assertEqual(sample["ignore_index"], 4)
        self.assertTrue(sample["is_scribble"])
        self.assertIn(4, np.unique(sample["label"]))  # most of a scribble slice is unlabeled

    def test_random_generator_2d_resizes_and_tensorizes(self):
        dataset = ScribbleBench2DDataset(
            dataset_name="ACDC",
            base_dir=self.root,
            split="train",
            transform=RandomGenerator2D((16, 16)),
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16))
        self.assertEqual(tuple(sample["label"].shape), (16, 16))
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertEqual(sample["label"].dtype, torch.int64)

    def test_dense_split_has_no_ignore_index(self):
        dataset = ScribbleBench2DDataset(
            dataset_name="ACDC", base_dir=self.root, split="train", sup_type="dense"
        )
        sample = dataset[0]
        self.assertFalse(sample["is_scribble"])
        self.assertNotIn(4, np.unique(sample["label"]))

    def test_slice_positions_for_volumes_selects_only_requested_cases(self):
        dataset = ScribbleBench2DDataset(dataset_name="ACDC", base_dir=self.root, split="train")
        self.assertEqual(dataset.cases, ["patient001_ED", "patient001_ES"])
        positions = dataset.slice_positions_for_volumes([0])
        depth_of_first_case = self.shapes[0][2]
        self.assertEqual(len(positions), depth_of_first_case)
        for position in positions:
            volume_index, _ = dataset.slice_index[position]
            self.assertEqual(volume_index, 0)

    def test_to_tensor_2d_preserves_native_resolution(self):
        dataset = ScribbleBench2DDataset(
            dataset_name="ACDC", base_dir=self.root, split="train", transform=ToTensor2D()
        )
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 8, 6))
        self.assertEqual(tuple(sample["label"].shape), (8, 6))


class RandomGenerator2DAugmentationTests(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_rotate_branch_pads_scribble_label_with_ignore_index_not_background(self):
        transform = RandomGenerator2D((10, 10))
        image = np.random.randn(10, 10).astype(np.float32)
        label = np.zeros((10, 10), dtype=np.int64)
        label[:] = 4  # ignore_index everywhere except one stroke
        label[4:6, 4:6] = 2
        sample = {"image": image, "label": label, "ignore_index": 4}

        # Force the "elif random.random() > 0.5" (rotate) branch deterministically:
        # first draw <= 0.5 skips rot_flip, second draw > 0.5 takes rotate.
        calls = iter([0.1, 0.9])
        original_random = random.random
        random.random = lambda: next(calls, 0.9)
        try:
            out = transform(sample)
        finally:
            random.random = original_random
        # Corners introduced by rotation must be ignore_index, not 0 (background),
        # since 4 already appears in the label -- matches the verified-against-
        # source class-conditional cval in _random_rotate_2d.
        self.assertIn(4, np.unique(out["label"].numpy()))


if __name__ == "__main__":
    unittest.main()
