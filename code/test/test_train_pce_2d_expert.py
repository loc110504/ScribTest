"""Regression tests for train_pce_2d_expert.py's non-dataset-touching pieces
(split resolution against the published protocol, checkpoint round-trip) --
mirrors test_train_pce_2d.py, adapted for this script's different
``resolve_case_split`` contract (see that function's docstring for why it no
longer also returns val indices into the same listing).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from networks.unet_2d import UNet2D
from train.legacy_splits import ACDC_TEST, ACDC_TRAIN, ACDC_VAL, MSCMR_TRAIN, MSCMR_VAL
from train.train_pce_2d_expert import (
    checkpoint_payload,
    resolve_case_split,
    resolve_test_indices,
    resolve_val_indices,
    restore_checkpoint,
    validate_args,
)


class ValidateArgsTests(unittest.TestCase):
    def _args(self, **overrides):
        defaults = dict(
            dataset="ACDC",
            patch_size=None,
            batch_size=None,
            feature_channels=(16, 32, 64, 128, 256),
            max_iterations=100,
            early_interval=10,
            late_interval=10,
            late_phase_start=50,
            num_workers=0,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_defaults_are_filled_in_per_dataset(self):
        args = validate_args(self._args())
        self.assertEqual(args.patch_size, (256, 256))
        self.assertEqual(args.batch_size, 24)

    def test_patch_size_must_be_divisible_by_16(self):
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            validate_args(self._args(patch_size=[250, 250]))


class ResolveCaseSplitTests(unittest.TestCase):
    def test_train_only_needs_train_groups_present(self):
        # Unlike ScribbleBench's single shared imagesTr directory, this
        # archive's training-slice cases never include the val patients at
        # all -- resolve_case_split must not require them.
        cases = ["patient{:03d}_frame01".format(patient) for patient in ACDC_TRAIN]
        train_indices, train_groups, val_groups, protocol = resolve_case_split(cases, "ACDC")
        self.assertEqual(len(train_groups), len(ACDC_TRAIN))
        self.assertEqual(len(val_groups), len(ACDC_VAL))
        self.assertEqual(len(train_indices), len(ACDC_TRAIN))

    def test_extra_on_disk_groups_are_tolerated_not_fatal(self):
        # Mirrors this archive's real MSCMR_training_slices, which also
        # contains subject2/subject4 (excluded from the published split).
        cases = ["subject{}_DE".format(subject) for subject in MSCMR_TRAIN] + ["subject2_DE", "subject4_DE"]
        train_indices, train_groups, val_groups, protocol = resolve_case_split(cases, "MSCMR")
        self.assertEqual(len(train_groups), len(MSCMR_TRAIN))
        self.assertEqual(len(train_indices), len(MSCMR_TRAIN))  # subject2/4 excluded

    def test_missing_train_group_raises(self):
        cases = ["patient{:03d}_frame01".format(patient) for patient in ACDC_TRAIN[1:]]  # missing one
        with self.assertRaises(RuntimeError):
            resolve_case_split(cases, "ACDC")


class ResolveValTestIndicesTests(unittest.TestCase):
    def test_resolve_val_indices_matches_published_val_groups(self):
        cases = ["patient{:03d}_frame01".format(patient) for patient in ACDC_VAL]
        indices = resolve_val_indices(cases, "ACDC")
        self.assertEqual(len(indices), len(ACDC_VAL))

    def test_resolve_val_indices_missing_group_raises(self):
        cases = ["patient{:03d}_frame01".format(patient) for patient in ACDC_VAL[1:]]
        with self.assertRaises(RuntimeError):
            resolve_val_indices(cases, "ACDC")

    def test_resolve_test_indices_matches_published_test_groups(self):
        cases = ["patient{:03d}_frame01".format(patient) for patient in ACDC_TEST]
        indices = resolve_test_indices(cases, "ACDC")
        self.assertEqual(len(indices), len(ACDC_TEST))


class CheckpointRoundTripTests(unittest.TestCase):
    def test_restore_recovers_step_score_and_weights(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
        )
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        split = {"train_cases": ["patient001_frame01"], "val_cases": ["patient002_frame01"]}

        payload = checkpoint_payload(model, optimizer, scaler, args, split, step=123, best_score=0.42)
        self.assertEqual(payload["data_config"]["data_source"], "expert_scribble")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)

            restored_model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
            restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.01, momentum=0.9)
            restored_scaler = torch.cuda.amp.GradScaler(enabled=False)
            step, best_score = restore_checkpoint(path, restored_model, restored_optimizer, restored_scaler, args, split)

        self.assertEqual(step, 123)
        self.assertAlmostEqual(best_score, 0.42)
        for expected, actual in zip(model.parameters(), restored_model.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_mismatched_dataset_is_rejected(self):
        args = SimpleNamespace(
            dataset="ACDC", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
        )
        model = UNet2D(in_chns=1, class_num=4, feature_chns=args.feature_channels)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        split = {"train_cases": [], "val_cases": []}
        payload = checkpoint_payload(model, optimizer, scaler, args, split, step=1, best_score=0.1)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "last.pth"
            torch.save(payload, path)
            mismatched_args = SimpleNamespace(
                dataset="MSCMR", root_path=None, patch_size=(64, 64), feature_channels=(4, 8, 16, 24, 32)
            )
            with self.assertRaisesRegex(ValueError, "dataset"):
                restore_checkpoint(path, model, optimizer, scaler, mismatched_args, split)


if __name__ == "__main__":
    unittest.main()
