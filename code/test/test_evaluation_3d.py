"""Regression tests for 3D sliding-window evaluation utilities."""

import importlib.util
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn


EVALUATOR_PATH = Path(__file__).with_name("test.py")
SPEC = importlib.util.spec_from_file_location("scribblebench_test_3d", EVALUATOR_PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class PointwiseModel(nn.Module):
    def forward(self, x):
        return torch.cat((-x, x), dim=1)


class TuplePointwiseModel(nn.Module):
    def forward(self, x):
        logits = torch.cat((-x, x), dim=1)
        return logits, torch.zeros_like(logits)


class Evaluation3DTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.image = torch.randn(1, 1, 7, 17, 19)

    def _run_sliding(self, model, max_accumulator_mb=1024):
        return evaluation.sliding_window_predict(
            model=model,
            image=self.image,
            class_num=2,
            patch_size=(4, 8, 9),
            device=torch.device("cpu"),
            overlap=0.5,
            sw_batch_size=3,
            max_accumulator_mb=max_accumulator_mb,
        )

    def test_sliding_window_matches_full_prediction(self):
        expected = (self.image[0, 0] > 0).numpy().astype(np.uint8)
        prediction = self._run_sliding(PointwiseModel())
        np.testing.assert_array_equal(prediction, expected)

    def test_tuple_output_and_memmap_path(self):
        expected = (self.image[0, 0] > 0).numpy().astype(np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            prediction = evaluation.sliding_window_predict(
                model=TuplePointwiseModel(),
                image=self.image,
                class_num=2,
                patch_size=(4, 8, 9),
                device=torch.device("cpu"),
                overlap=0.25,
                max_accumulator_mb=0,
                temp_dir=temp_dir,
            )
        np.testing.assert_array_equal(prediction, expected)

    def test_padding_is_removed(self):
        image = torch.randn(1, 1, 3, 5, 7)
        prediction = evaluation.sliding_window_predict(
            PointwiseModel(), image, 2, (4, 8, 9), torch.device("cpu")
        )
        self.assertEqual(prediction.shape, (3, 5, 7))

    def test_physical_metrics(self):
        target = np.zeros((4, 5, 6), dtype=np.uint8)
        target[1:3, 1:4, 2:5] = 1
        result = evaluation.calculate_metric_per_class(target, target, 1, (2.0, 1.0, 1.0))
        self.assertAlmostEqual(result["dice"], 1.0)
        self.assertAlmostEqual(result["asd"], 0.0)
        self.assertAlmostEqual(result["hd95"], 0.0)

    def test_empty_metric_policy(self):
        empty = np.zeros((4, 5, 6), dtype=np.uint8)
        both_empty = evaluation.calculate_metric_per_class(empty, empty, 1, (1, 1, 1))
        self.assertEqual(both_empty["status"], "both_empty")
        self.assertTrue(np.isnan(both_empty["dice"]))
        prediction = empty.copy()
        prediction[1, 1, 1] = 1
        one_empty = evaluation.calculate_metric_per_class(prediction, empty, 1, (1, 1, 1))
        self.assertEqual(one_empty["dice"], 0.0)
        self.assertTrue(np.isnan(one_empty["hd95"]))

    def test_gaussian_and_scan_coverage(self):
        gaussian = evaluation.compute_gaussian_importance_map((5, 7, 9))
        self.assertEqual(gaussian.shape, (5, 7, 9))
        self.assertGreater(float(gaussian.min()), 0.0)
        starts = evaluation.compute_scan_starts((7, 17, 19), (4, 8, 9), 0.5)
        coverage = np.zeros((7, 17, 19), dtype=np.uint8)
        for d_start, h_start, w_start in starts:
            coverage[d_start : d_start + 4, h_start : h_start + 8, w_start : w_start + 9] = 1
        self.assertTrue(np.all(coverage))

    def test_checkpoint_prefixes(self):
        source = nn.Conv3d(1, 2, kernel_size=1)
        checkpoint = {
            "model_state_dict": {
                "module.model." + key: value.clone()
                for key, value in source.state_dict().items()
            }
        }
        restored = nn.Conv3d(1, 2, kernel_size=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pth"
            torch.save(checkpoint, checkpoint_path)
            missing, unexpected = evaluation.load_checkpoint(restored, checkpoint_path)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        for expected, actual in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(actual, expected)

    def test_prediction_nifti_preserves_geometry(self):
        prediction_dhw = np.arange(3 * 4 * 5, dtype=np.uint8).reshape(3, 4, 5)
        affine = np.asarray(
            [[1.2, 0, 0, 4], [0, 1.4, 0, 5], [0, 0, 3.0, 6], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.nii.gz"
            output_path = Path(temp_dir) / "prediction.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((5, 4, 3), np.float32), affine), image_path)
            evaluation.save_prediction_nifti(
                prediction_dhw, {"image_path": str(image_path)}, output_path
            )
            saved = nib.load(output_path)
            np.testing.assert_array_equal(
                np.asanyarray(saved.dataobj), prediction_dhw.transpose(2, 1, 0)
            )
            np.testing.assert_allclose(saved.affine, affine)

    def test_argument_validation(self):
        valid = Namespace(
            in_chns=1,
            num_classes=None,
            patch_size=(4, 8, 9),
            overlap=0.5,
            sw_batch_size=1,
            sigma_scale=0.125,
            mirror_axes=(2, 0, 2),
            max_accumulator_mb=1024,
            case_limit=None,
            temp_dir=None,
        )
        self.assertEqual(evaluation.validate_args(valid).mirror_axes, (0, 2))
        valid.overlap = 1.0
        with self.assertRaisesRegex(ValueError, "overlap"):
            evaluation.validate_args(valid)

    def test_scribblebench_and_macro_averages_are_both_reported(self):
        cases = [
            {
                "foreground_mean": {"dice": 0.5, "asd": 2.0, "hd95": 4.0},
                "classes": {
                    "1": {"dice": 1.0, "asd": 1.0, "hd95": 2.0},
                    "2": {"dice": 0.0, "asd": 3.0, "hd95": 6.0},
                },
            },
            {
                "foreground_mean": {"dice": 1.0, "asd": 2.0, "hd95": 3.0},
                "classes": {
                    "1": {"dice": 1.0, "asd": 2.0, "hd95": 3.0},
                    "2": {"dice": np.nan, "asd": np.nan, "hd95": np.nan},
                },
            },
        ]
        summary = evaluation.summarize_results(cases, ("background", "one", "two"))
        self.assertAlmostEqual(summary["foreground_macro_mean"]["dice"], 0.5)
        self.assertAlmostEqual(summary["scribblebench_mean_dice"], 0.75)


if __name__ == "__main__":
    unittest.main()
