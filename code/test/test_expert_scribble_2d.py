"""CPU regression checks for the expert-scribble ACDC/MSCMR h5 dataset.

Builds tiny synthetic h5 fixtures (matching the real
``data/{ACDC,MSCMR}/*_training_slices|*_training_volumes|*_validation_volumes|
*_testing_volumes`` layout) in a temp dir, mirroring this repo's convention
of never touching ``data/``/``dataset/`` directly in tests.
"""

import os
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataloader.expert_scribble_2d import ExpertScribble2DDataset, ExpertScribbleVolumeDataset
from dataloader.scribblebench_2d import RandomGenerator2D


def _write_h5(path, **arrays):
    with h5py.File(path, "w") as handle:
        for name, array in arrays.items():
            handle.create_dataset(name, data=array)


class ExpertScribbleFixtureMixin:
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _build_acdc_archive(self, root):
        slices_dir = os.path.join(root, "ACDC_training_slices")
        volumes_dir = os.path.join(root, "ACDC_training_volumes")
        os.makedirs(slices_dir)
        os.makedirs(volumes_dir)
        rng = np.random.default_rng(0)
        # Two cases (frames) for patient001, three slices each.
        for case, n_slices in (("patient001_frame01", 3), ("patient001_frame12", 2)):
            for slice_index in range(n_slices):
                image = rng.random((8, 8)).astype(np.float32)
                scribble = np.full((8, 8), 4, dtype=np.uint16)
                scribble[1:3, 1:3] = 1
                label = np.zeros((8, 8), dtype=np.uint8)
                label[1:4, 1:4] = 1
                _write_h5(
                    os.path.join(slices_dir, "{}_slice_{}.h5".format(case, slice_index)),
                    image=image, label=label, scribble=scribble,
                )
        for case, depth in (("patient001_frame01", 3), ("patient001_frame12", 2)):
            image = rng.random((depth, 8, 8)).astype(np.float32)
            label = np.zeros((depth, 8, 8), dtype=np.uint8)
            label[:, 1:4, 1:4] = 1
            _write_h5(os.path.join(volumes_dir, "{}.h5".format(case)), image=image, label=label)
        return root

    def _build_mscmr_archive(self, root):
        slices_dir = os.path.join(root, "MSCMR_training_slices")
        val_dir = os.path.join(root, "MSCMR_validation_volumes")
        test_dir = os.path.join(root, "MSCMR_testing_volumes")
        os.makedirs(slices_dir)
        os.makedirs(val_dir)
        os.makedirs(test_dir)
        rng = np.random.default_rng(1)
        for slice_index in range(2):
            image = rng.random((10, 10)).astype(np.float32)
            scribble = np.full((10, 10), 4, dtype=np.uint16)
            scribble[2:4, 2:4] = 2
            label = np.zeros((10, 10), dtype=np.int16)
            label[2:5, 2:5] = 2
            _write_h5(
                os.path.join(slices_dir, "subject13_DE_slice_{}.h5".format(slice_index)),
                image=image, label=label, scribble=scribble,
            )
        for folder, case in ((val_dir, "subject1_DE"), (test_dir, "subject3_DE")):
            image = rng.random((4, 10, 10)).astype(np.float32)
            label = np.zeros((4, 10, 10), dtype=np.int16)
            label[:, 2:5, 2:5] = 2
            _write_h5(os.path.join(folder, "{}.h5".format(case)), image=image, label=label)
        return root


class ExpertScribble2DDatasetTests(ExpertScribbleFixtureMixin, unittest.TestCase):
    def test_acdc_groups_slices_by_case_in_order(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        dataset = ExpertScribble2DDataset("ACDC", base_dir=root, split="train", sup_type="scribble")
        self.assertEqual(dataset.cases, ["patient001_frame01", "patient001_frame12"])
        self.assertEqual(len(dataset), 5)  # 3 + 2 slices
        self.assertEqual(dataset.num_classes, 4)
        self.assertEqual(dataset.ignore_index, 4)
        self.assertEqual(dataset.images[0].shape, (3, 8, 8))
        self.assertEqual(dataset.labels[0].shape, (3, 8, 8))

    def test_mscmr_case_name_keeps_de_suffix(self):
        root = self._build_mscmr_archive(os.path.join(self.tmp_dir, "MSCMR"))
        dataset = ExpertScribble2DDataset("MSCMR", base_dir=root, split="train", sup_type="scribble")
        self.assertEqual(dataset.cases, ["subject13_DE"])
        self.assertEqual(len(dataset), 2)

    def test_sup_type_scribble_selects_scribble_key(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        dataset = ExpertScribble2DDataset("ACDC", base_dir=root, split="train", sup_type="scribble")
        sample = dataset[0]
        self.assertIn(4, np.unique(sample["label"]))  # ignore_index present -> came from "scribble"

    def test_sup_type_dense_selects_label_key(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        dataset = ExpertScribble2DDataset("ACDC", base_dir=root, split="train", sup_type="dense")
        sample = dataset[0]
        self.assertNotIn(4, np.unique(sample["label"]))  # dense label never contains ignore_index=4

    def test_slice_positions_for_volumes(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        dataset = ExpertScribble2DDataset("ACDC", base_dir=root, split="train", sup_type="scribble")
        positions = dataset.slice_positions_for_volumes([1])
        self.assertEqual(positions, [3, 4])  # patient001_frame12's 2 slices

    def test_getitem_matches_scribblebench_contract_for_randomgenerator2d(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        transform = RandomGenerator2D((16, 16))
        dataset = ExpertScribble2DDataset("ACDC", base_dir=root, split="train", sup_type="scribble", transform=transform)
        sample = dataset[0]
        self.assertEqual(tuple(sample["image"].shape), (1, 16, 16))
        self.assertEqual(tuple(sample["label"].shape), (16, 16))

    def test_rejects_unsupported_dataset(self):
        with self.assertRaises(ValueError):
            ExpertScribble2DDataset("WORD", base_dir=self.tmp_dir)

    def test_rejects_non_train_split(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        with self.assertRaises(ValueError):
            ExpertScribble2DDataset("ACDC", base_dir=root, split="val")

    def test_missing_archive_raises(self):
        with self.assertRaises(FileNotFoundError):
            ExpertScribble2DDataset("ACDC", base_dir=os.path.join(self.tmp_dir, "nope"))

    def test_non_contiguous_slice_indices_raise(self):
        root = os.path.join(self.tmp_dir, "ACDC")
        slices_dir = os.path.join(root, "ACDC_training_slices")
        os.makedirs(slices_dir)
        _write_h5(
            os.path.join(slices_dir, "patient001_frame01_slice_0.h5"),
            image=np.zeros((4, 4), dtype=np.float32), label=np.zeros((4, 4), dtype=np.uint8),
            scribble=np.zeros((4, 4), dtype=np.uint16),
        )
        _write_h5(
            os.path.join(slices_dir, "patient001_frame01_slice_2.h5"),  # gap: no slice_1
            image=np.zeros((4, 4), dtype=np.float32), label=np.zeros((4, 4), dtype=np.uint8),
            scribble=np.zeros((4, 4), dtype=np.uint16),
        )
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            ExpertScribble2DDataset("ACDC", base_dir=root, split="train")


class ExpertScribbleVolumeDatasetTests(ExpertScribbleFixtureMixin, unittest.TestCase):
    def test_acdc_reads_training_volumes(self):
        root = self._build_acdc_archive(os.path.join(self.tmp_dir, "ACDC"))
        dataset = ExpertScribbleVolumeDataset("ACDC", base_dir=root, split="val")
        self.assertEqual(sorted(dataset.cases), ["patient001_frame01", "patient001_frame12"])
        sample = dataset[dataset.cases.index("patient001_frame01")]
        self.assertEqual(sample["image"].shape, (3, 8, 8))
        self.assertEqual(sample["gt_label"].shape, (3, 8, 8))

    def test_mscmr_val_and_test_use_separate_folders(self):
        root = self._build_mscmr_archive(os.path.join(self.tmp_dir, "MSCMR"))
        val_dataset = ExpertScribbleVolumeDataset("MSCMR", base_dir=root, split="val")
        test_dataset = ExpertScribbleVolumeDataset("MSCMR", base_dir=root, split="test")
        self.assertEqual(val_dataset.cases, ["subject1_DE"])
        self.assertEqual(test_dataset.cases, ["subject3_DE"])

    def test_missing_folder_raises(self):
        root = os.path.join(self.tmp_dir, "MSCMR")
        os.makedirs(root)
        with self.assertRaises(FileNotFoundError):
            ExpertScribbleVolumeDataset("MSCMR", base_dir=root, split="val")


if __name__ == "__main__":
    unittest.main()
