"""Regression tests for WORD's 16-organ -> DMSPS 7-organ label remapping."""

import os
import sys
import unittest

import numpy as np

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from dataloader.scribblebench_3d import DATASET_CONFIGS, _remap_word_label

IGNORE_INDEX = DATASET_CONFIGS["WORD"]["ignore_index"]

# WORD-v0.1.0's original fixed label ids, for building synthetic inputs.
ORIGINAL_BACKGROUND = 0
ORIGINAL_LIVER = 1
ORIGINAL_PANCREAS = 8  # kept, but esophagus=7 sits between spleen-family ids and it
ORIGINAL_ESOPHAGUS = 7  # dropped
ORIGINAL_DUODENUM = 9  # dropped
ORIGINAL_FEMUR_R = 16  # dropped
ORIGINAL_IGNORE = 17


class DatasetConfigTests(unittest.TestCase):
    def test_word_config_has_seven_organs_plus_background(self):
        config = DATASET_CONFIGS["WORD"]
        self.assertEqual(len(config["class_names"]), 8)
        self.assertEqual(
            config["class_names"],
            (
                "background",
                "liver",
                "spleen",
                "left_kidney",
                "right_kidney",
                "stomach",
                "gallbladder",
                "pancreas",
            ),
        )
        self.assertEqual(config["ignore_index"], 8)


class RemapScribbleModeTests(unittest.TestCase):
    def test_kept_organs_map_to_contiguous_ids(self):
        label = np.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=np.int64)
        remapped = _remap_word_label(label, is_scribble=True, ignore_index=IGNORE_INDEX)
        np.testing.assert_array_equal(remapped, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_dropped_organs_and_original_ignore_become_new_ignore_index(self):
        label = np.array(
            [ORIGINAL_ESOPHAGUS, ORIGINAL_DUODENUM, ORIGINAL_FEMUR_R, ORIGINAL_IGNORE],
            dtype=np.int64,
        )
        remapped = _remap_word_label(label, is_scribble=True, ignore_index=IGNORE_INDEX)
        self.assertTrue(np.all(remapped == IGNORE_INDEX))

    def test_dropped_organ_scribble_is_not_forced_to_background(self):
        # A real (non-background) scribble point on a dropped organ must not
        # silently become "background" supervision -- that would be false.
        label = np.array([ORIGINAL_DUODENUM], dtype=np.int64)
        remapped = _remap_word_label(label, is_scribble=True, ignore_index=IGNORE_INDEX)
        self.assertNotEqual(remapped[0], 0)
        self.assertEqual(remapped[0], IGNORE_INDEX)


class RemapDenseModeTests(unittest.TestCase):
    def test_kept_organs_map_to_contiguous_ids(self):
        label = np.array([0, 1, 2, 3, 4, 5, 6, 8], dtype=np.int64)
        remapped = _remap_word_label(label, is_scribble=False, ignore_index=IGNORE_INDEX)
        np.testing.assert_array_equal(remapped, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_dropped_organs_become_background(self):
        label = np.array([ORIGINAL_ESOPHAGUS, ORIGINAL_DUODENUM, ORIGINAL_FEMUR_R], dtype=np.int64)
        remapped = _remap_word_label(label, is_scribble=False, ignore_index=IGNORE_INDEX)
        self.assertTrue(np.all(remapped == 0))

    def test_voxel_counts_are_conserved(self):
        rng = np.random.default_rng(0)
        label = rng.integers(0, 18, size=(20, 20, 20)).astype(np.int64)
        remapped = _remap_word_label(label, is_scribble=False, ignore_index=IGNORE_INDEX)
        self.assertEqual(remapped.shape, label.shape)
        self.assertEqual(remapped.size, label.size)
        # Every kept *non-background* id's voxel count must be preserved
        # exactly (id 0 is skipped here: it is both the original background
        # id and the fallback destination for dropped organs in dense mode,
        # so it is checked separately below instead of one-to-one).
        for original_id, new_id in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (8, 7)]:
            self.assertEqual(
                int(np.count_nonzero(label == original_id)),
                int(np.count_nonzero(remapped == new_id)),
            )
        dropped_count = sum(
            int(np.count_nonzero(label == original_id)) for original_id in (7, 9, 10, 11, 12, 13, 14, 15, 16, 17)
        )
        self.assertEqual(int(np.count_nonzero(remapped == 0)) - int(np.count_nonzero(label == 0)), dropped_count)


if __name__ == "__main__":
    unittest.main()
