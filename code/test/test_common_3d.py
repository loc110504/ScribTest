"""Regression tests for shared training infrastructure in common_3d.py."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from train.common_3d import guard_fresh_output_dir, partial_cross_entropy


class PartialCrossEntropyTests(unittest.TestCase):
    def test_all_unlabeled_patch_returns_zero_loss_not_error(self):
        """Uniform random cropping can yield a patch with zero annotated
        voxels; this must contribute zero loss/gradient rather than raise,
        so a long unattended training run does not abort on a rare patch."""
        logits = torch.randn(2, 4, 4, 8, 8, requires_grad=True)
        target = torch.full((2, 4, 8, 8), fill_value=4, dtype=torch.long)  # all ignore_index
        loss, valid_count = partial_cross_entropy(logits, target, ignore_index=4)
        self.assertEqual(valid_count.item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_partially_labeled_patch_still_computes_normal_loss(self):
        logits = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
        target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        target[0, 0, 0, 0] = 1
        loss, valid_count = partial_cross_entropy(logits, target, ignore_index=3)
        self.assertEqual(valid_count.item(), 8)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.any(logits.grad != 0))

    def test_out_of_range_class_raises(self):
        logits = torch.randn(1, 3, 2, 2, 2)
        target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
        target[0, 0, 0, 0] = 5  # not a valid class and not ignore_index(3)
        with self.assertRaises(ValueError):
            partial_cross_entropy(logits, target, ignore_index=3)


class GuardFreshOutputDirTests(unittest.TestCase):
    def test_empty_directory_allows_fresh_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard_fresh_output_dir(Path(tmp), resume=None)  # must not raise

    def test_fresh_start_into_existing_best_pth_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "best.pth").touch()
            with self.assertRaisesRegex(ValueError, "best.pth"):
                guard_fresh_output_dir(Path(tmp), resume=None)

    def test_fresh_start_into_existing_last_pth_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "last.pth").touch()
            with self.assertRaisesRegex(ValueError, "last.pth"):
                guard_fresh_output_dir(Path(tmp), resume=None)

    def test_resume_bypasses_the_guard_even_with_existing_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "best.pth").touch()
            guard_fresh_output_dir(Path(tmp), resume=str(Path(tmp) / "best.pth"))  # must not raise

    def test_custom_checkpoint_names_are_honored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ACDC" / "best.pth").parent.mkdir(parents=True)
            (Path(tmp) / "ACDC" / "best.pth").touch()
            with self.assertRaises(ValueError):
                guard_fresh_output_dir(Path(tmp) / "ACDC", resume=None, checkpoint_names=("best.pth",))
            guard_fresh_output_dir(Path(tmp) / "MSCMR", resume=None, checkpoint_names=("best.pth",))


if __name__ == "__main__":
    unittest.main()
