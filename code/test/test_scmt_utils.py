"""CPU regression checks for the SC-MT (Scribble-Calibrated Mean Teacher) utilities."""

import math
import os
import sys
import unittest

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.scmt import (
    ReliabilityCalibrationTable,
    assign_block_folds,
    assign_confidence_bin,
    assign_distance_bin,
    assign_scribble_blocks,
    batch_transfer_distance_from_labels,
    build_class_trees,
    build_fold_map,
    fit_distance_bin_edges,
    partition_for_held_out_fold,
    query_transfer_distance,
    random_crop_3d,
    random_flip_rotate_3d,
    random_flip_rotate_resize_2d,
    reliability_weighted_consistency_loss,
    stable_seed,
    strong_intensity_augment,
    teacher_confidence,
)


class AssignScribbleBlocksTests(unittest.TestCase):
    def test_two_slices_two_disjoint_blocks(self):
        label = np.zeros((3, 6, 6), dtype=np.int64)
        label[0, 0:2, 0:2] = 1
        label[1, 4:6, 4:6] = 1
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertEqual(len(blocks[1]), 2)

    def test_2d_single_slice_two_disjoint_components(self):
        label = np.zeros((10, 10), dtype=np.int64)
        label[0:2, 0:2] = 1
        label[8:10, 8:10] = 1
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertEqual(len(blocks[1]), 2)
        self.assertEqual(blocks[1][0].shape[1], 2)

    def test_background_class_included(self):
        label = np.zeros((2, 4, 4), dtype=np.int64)
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertIn(0, blocks)

    def test_rejects_unsupported_rank(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            assign_scribble_blocks(np.zeros((2, 2, 2, 2)), num_classes=2)


class AssignBlockFoldsTests(unittest.TestCase):
    def _label_with_n_blocks(self, n_blocks, class_id=1, ignore_index=2, spacing_stride=3):
        label = np.full((1, n_blocks * spacing_stride), ignore_index, dtype=np.int64)
        for i in range(n_blocks):
            label[0, i * spacing_stride] = class_id
        return label

    def test_every_voxel_gets_a_fold_in_range(self):
        label = self._label_with_n_blocks(20)
        rng = np.random.default_rng(0)
        coords, folds = assign_block_folds(label, num_classes=2, num_folds=5, rng=rng)
        self.assertEqual(len(coords[1]), 20)
        self.assertTrue(np.all(folds[1] >= 0) and np.all(folds[1] < 5))

    def test_rejects_num_folds_below_two(self):
        label = self._label_with_n_blocks(3)
        with self.assertRaises(ValueError):
            assign_block_folds(label, num_classes=2, num_folds=1, rng=np.random.default_rng(0))

    def test_deterministic_given_same_seed(self):
        label = self._label_with_n_blocks(20)
        rng_a = np.random.default_rng(stable_seed(7, "caseA"))
        rng_b = np.random.default_rng(stable_seed(7, "caseA"))
        coords_a, folds_a = assign_block_folds(label, 2, 5, rng_a)
        coords_b, folds_b = assign_block_folds(label, 2, 5, rng_b)
        np.testing.assert_array_equal(coords_a[1], coords_b[1])
        np.testing.assert_array_equal(folds_a[1], folds_b[1])

    def test_absent_class_has_empty_arrays(self):
        label = np.zeros((1, 6), dtype=np.int64)
        coords, folds = assign_block_folds(label, num_classes=3, num_folds=4, rng=np.random.default_rng(0))
        self.assertEqual(len(coords[2]), 0)
        self.assertEqual(len(folds[2]), 0)

    def test_full_rotation_covers_every_voxel_exactly_once_as_held_out(self):
        # Many single-voxel blocks -> over a full num_folds cycle, every
        # voxel is held out in exactly one fold and supervised in the rest.
        label = self._label_with_n_blocks(30)
        rng = np.random.default_rng(1)
        coords, folds = assign_block_folds(label, num_classes=2, num_folds=5, rng=rng)
        all_coords = [tuple(row) for row in coords[1]]
        held_out_counts = {coord: 0 for coord in all_coords}
        for fold in range(5):
            _, cal = partition_for_held_out_fold(coords, folds, fold)
            for row in cal[1]:
                held_out_counts[tuple(row)] += 1
        self.assertTrue(all(count == 1 for count in held_out_counts.values()))


class PartitionForHeldOutFoldTests(unittest.TestCase):
    def test_sup_and_cal_disjoint_and_cover_all(self):
        coords = {1: np.array([[0, 0], [1, 1], [2, 2], [3, 3]])}
        folds = {1: np.array([0, 1, 0, 2])}
        sup, cal = partition_for_held_out_fold(coords, folds, held_out_fold=0)
        self.assertEqual(len(cal[1]), 2)
        self.assertEqual(len(sup[1]), 2)
        sup_set = {tuple(row) for row in sup[1]}
        cal_set = {tuple(row) for row in cal[1]}
        self.assertEqual(sup_set & cal_set, set())
        expected = {tuple(row) for row in coords[1]}
        self.assertEqual(sup_set | cal_set, expected)


class BuildFoldMapTests(unittest.TestCase):
    def test_scribble_voxels_get_their_fold_others_get_negative_one(self):
        label = np.full((1, 12), 2, dtype=np.int64)
        label[0, 0] = 1
        label[0, 6] = 1
        rng = np.random.default_rng(0)
        fold_map = build_fold_map(label, num_classes=2, num_folds=3, rng=rng)
        self.assertEqual(fold_map.shape, label.shape)
        self.assertTrue(np.all(fold_map[label == 2] == -1))
        self.assertTrue(np.all(fold_map[label != 2] >= 0))


class TransferDistanceTests(unittest.TestCase):
    def test_matches_brute_force_with_anisotropic_spacing(self):
        rng = np.random.default_rng(1)
        sup_points = rng.integers(0, 20, size=(15, 3))
        spacing = np.array([5.0, 1.0, 1.0])
        trees = build_class_trees({0: sup_points}, spacing)

        query_points = rng.integers(0, 20, size=(8, 3))
        class_ids = np.zeros(8, dtype=np.int64)
        distance = query_transfer_distance(trees, class_ids, query_points, spacing)

        physical_sup = sup_points * spacing[None, :]
        physical_query = query_points * spacing[None, :]
        expected = np.array([np.min(np.linalg.norm(physical_sup - q, axis=1)) for q in physical_query])
        np.testing.assert_allclose(distance, expected, atol=1e-6)

    def test_missing_class_tree_gives_nan(self):
        trees = build_class_trees({0: np.zeros((0, 3), dtype=np.int64)}, np.ones(3))
        distance = query_transfer_distance(trees, np.array([0]), np.array([[1, 1, 1]]), np.ones(3))
        self.assertTrue(np.isnan(distance[0]))

    def test_batch_transfer_distance_from_labels_only_touches_valid_voxels(self):
        sup_label = np.full((1, 2, 2, 2), 3, dtype=np.int64)  # 3 = "ignore" (outside num_classes)
        sup_label[0, 0, 0, 0] = 0  # the only class-0 supervised voxel, at the origin
        predicted_class = np.zeros((1, 2, 2, 2), dtype=np.int64)
        valid = np.zeros((1, 2, 2, 2), dtype=bool)
        valid[0, 1, 1, 1] = True
        out = batch_transfer_distance_from_labels(
            sup_label, predicted_class, valid, np.ones((1, 3)), num_classes=3
        )
        self.assertTrue(np.isnan(out[0, 0, 0, 0]))  # not requested
        self.assertAlmostEqual(out[0, 1, 1, 1], math.sqrt(3), places=5)

    def test_no_support_for_predicted_class_is_nan(self):
        sup_label = np.full((1, 2, 2, 2), 3, dtype=np.int64)  # no supervised voxel of any class
        predicted_class = np.zeros((1, 2, 2, 2), dtype=np.int64)
        valid = np.ones((1, 2, 2, 2), dtype=bool)
        out = batch_transfer_distance_from_labels(sup_label, predicted_class, valid, np.ones((1, 3)), num_classes=3)
        self.assertTrue(np.all(np.isnan(out)))


class FitDistanceBinEdgesTests(unittest.TestCase):
    def test_edges_shape_and_monotonic(self):
        coords = {1: np.array([[0, 0, k] for k in range(10)])}
        folds = {1: np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])}  # one sup voxel (fold 0), rest cal (fold 1)
        case = {"coords_by_class": coords, "fold_by_class": folds, "spacing": np.ones(3)}
        edges = fit_distance_bin_edges([case], num_classes=2, num_folds=2, num_distance_bins=3)
        self.assertEqual(edges.shape, (2, 2))
        self.assertTrue(np.all(np.diff(edges[1]) >= 0))

    def test_single_bin_returns_no_edges(self):
        coords = {1: np.array([[0, 0, 0]])}
        folds = {1: np.array([0])}
        case = {"coords_by_class": coords, "fold_by_class": folds, "spacing": np.ones(3)}
        edges = fit_distance_bin_edges([case], num_classes=2, num_folds=2, num_distance_bins=1)
        self.assertEqual(edges.shape, (2, 0))

    def test_class_never_observed_is_nan_row(self):
        coords = {1: np.array([[0, 0, 0]])}
        folds = {1: np.array([0])}
        case = {"coords_by_class": coords, "fold_by_class": folds, "spacing": np.ones(3)}
        edges = fit_distance_bin_edges([case], num_classes=2, num_folds=2, num_distance_bins=3)
        self.assertTrue(np.all(np.isnan(edges[0])))  # class 0 never appears


class AssignDistanceBinTests(unittest.TestCase):
    def test_nan_distance_gets_no_support_bin(self):
        distance = torch.tensor([float("nan"), 1.0])
        predicted_class = torch.tensor([0, 0])
        edges = torch.zeros(1, 0)
        out = assign_distance_bin(distance, predicted_class, edges, num_finite_bins=1)
        self.assertEqual(out[0].item(), 1)  # no-support sentinel == num_finite_bins
        self.assertEqual(out[1].item(), 0)

    def test_finite_bins_follow_edges(self):
        distance = torch.tensor([0.5, 5.0, 50.0])
        predicted_class = torch.tensor([0, 0, 0])
        edges = torch.tensor([[1.0, 10.0]])  # class 0: bin0<1, bin1 in [1,10), bin2>=10
        out = assign_distance_bin(distance, predicted_class, edges, num_finite_bins=3)
        self.assertEqual(out.tolist(), [0, 1, 2])

    def test_nan_edges_row_falls_back_to_bin_zero(self):
        distance = torch.tensor([5.0])
        predicted_class = torch.tensor([0])
        edges = torch.tensor([[float("nan"), float("nan")]])
        out = assign_distance_bin(distance, predicted_class, edges, num_finite_bins=3)
        self.assertEqual(out.item(), 0)


class TeacherConfidenceTests(unittest.TestCase):
    def test_matches_max_and_argmax(self):
        prob = torch.tensor([[[0.1, 0.7], [0.2, 0.1]], [[0.6, 0.2], [0.7, 0.8]]]).unsqueeze(0)
        # shape [1, 2, 2, 2] -> C=2
        conf, pred = teacher_confidence(prob)
        expected_conf, expected_pred = prob.max(dim=1)
        torch.testing.assert_close(conf, expected_conf)
        torch.testing.assert_close(pred, expected_pred)

    def test_confidence_bin_edges(self):
        confidence = torch.tensor([0.0, 0.19, 0.2, 0.99, 1.0])
        bins = assign_confidence_bin(confidence, num_confidence_bins=5)
        self.assertEqual(bins.tolist(), [0, 0, 1, 4, 4])


class ReliabilityCalibrationTableTests(unittest.TestCase):
    def test_default_used_before_any_evidence(self):
        table = ReliabilityCalibrationTable(num_classes=2, num_confidence_bins=2, num_distance_bins=2, min_samples=5)
        out = table.query(np.array([0]), np.array([0]), np.array([0]), default_reliability=np.array([0.42]))
        self.assertAlmostEqual(out[0], 0.42)

    def test_cell_used_once_min_samples_reached(self):
        table = ReliabilityCalibrationTable(num_classes=1, num_confidence_bins=1, num_distance_bins=1, min_samples=3)
        table.observe(
            class_ids=np.zeros(3, dtype=np.int64), confidence_bins=np.zeros(3, dtype=np.int64),
            distance_bins=np.zeros(3, dtype=np.int64), corrects=np.array([1.0, 1.0, 0.0]),
        )
        table.commit_epoch()
        out = table.query(np.array([0]), np.array([0]), np.array([0]), default_reliability=np.array([0.0]))
        self.assertAlmostEqual(out[0], 2.0 / 3.0)

    def test_below_min_samples_falls_back_to_class_then_global_then_default(self):
        table = ReliabilityCalibrationTable(num_classes=1, num_confidence_bins=2, num_distance_bins=2, min_samples=10)
        # 4 observations spread over 4 different cells: none reaches the cell
        # min_samples, but pooled at the class level there are 4 -- still
        # short of min_samples=10, so class-level also fails -> falls to
        # global, which also has only 4 -> falls to caller default.
        table.observe(
            class_ids=np.array([0, 0, 0, 0]), confidence_bins=np.array([0, 0, 1, 1]),
            distance_bins=np.array([0, 1, 0, 1]), corrects=np.array([1.0, 1.0, 1.0, 1.0]),
        )
        table.commit_epoch()
        out = table.query(np.array([0]), np.array([0]), np.array([0]), default_reliability=np.array([0.33]))
        self.assertAlmostEqual(out[0], 0.33)

    def test_ema_blends_across_epochs(self):
        table = ReliabilityCalibrationTable(
            num_classes=1, num_confidence_bins=1, num_distance_bins=1, momentum=0.5, min_samples=1
        )
        table.observe(np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.ones(10))
        table.commit_epoch()  # first commit: value = 1.0 (no prior -> direct assign)
        table.observe(np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.zeros(10))
        table.commit_epoch()  # second commit: EMA(1.0, 0.0, momentum=0.5) = 0.5
        out = table.query(np.array([0]), np.array([0]), np.array([0]), default_reliability=np.array([0.0]))
        self.assertAlmostEqual(out[0], 0.5)

    def test_state_dict_round_trip(self):
        table = ReliabilityCalibrationTable(num_classes=2, num_confidence_bins=2, num_distance_bins=2, min_samples=1)
        table.observe(np.array([0, 1]), np.array([0, 1]), np.array([0, 1]), np.array([1.0, 0.0]))
        table.commit_epoch()
        state = table.state_dict()
        restored = ReliabilityCalibrationTable(num_classes=2, num_confidence_bins=2, num_distance_bins=2, min_samples=1)
        restored.load_state_dict(state)
        np.testing.assert_array_equal(restored.cell_value, table.cell_value)
        np.testing.assert_array_equal(restored.cell_count, table.cell_count)

    def test_query_output_is_clipped_to_unit_interval(self):
        table = ReliabilityCalibrationTable(num_classes=1, num_confidence_bins=1, num_distance_bins=1, min_samples=1)
        out = table.query(np.array([0]), np.array([0]), np.array([0]), default_reliability=np.array([1.5]))
        self.assertLessEqual(out[0], 1.0)


class ReliabilityWeightedConsistencyLossTests(unittest.TestCase):
    def test_zero_when_omega_u_empty(self):
        student = torch.softmax(torch.randn(1, 3, 2, 2), dim=1)
        teacher = torch.softmax(torch.randn(1, 3, 2, 2), dim=1)
        reliability = torch.ones(1, 2, 2)
        omega_u = torch.zeros(1, 2, 2, dtype=torch.bool)
        loss = reliability_weighted_consistency_loss(student, teacher, reliability, omega_u)
        self.assertEqual(loss.item(), 0.0)

    def test_matches_manual_computation_and_backprops_through_student_only(self):
        torch.manual_seed(0)
        logits_s = torch.randn(1, 3, 1, 2, requires_grad=True)
        logits_t = torch.randn(1, 3, 1, 2, requires_grad=True)
        student = torch.softmax(logits_s, dim=1)
        teacher = torch.softmax(logits_t, dim=1)
        reliability = torch.tensor([[[0.5, 0.8]]])
        omega_u = torch.ones(1, 1, 2, dtype=torch.bool)

        loss = reliability_weighted_consistency_loss(student, teacher, reliability, omega_u)
        diff2 = (student - teacher.detach()).pow(2).mean(dim=1)
        expected = (diff2 * reliability).sum() / 2.0
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)

        loss.backward()
        self.assertIsNotNone(logits_s.grad)
        self.assertIsNone(logits_t.grad)

    def test_higher_reliability_voxel_dominates_gradient_magnitude(self):
        torch.manual_seed(0)
        logits_s = torch.zeros(1, 3, 1, 2, requires_grad=True)
        teacher = torch.softmax(torch.randn(1, 3, 1, 2), dim=1)
        student = torch.softmax(logits_s, dim=1)
        omega_u = torch.ones(1, 1, 2, dtype=torch.bool)

        reliability_low_first = torch.tensor([[[0.01, 0.99]]])
        loss = reliability_weighted_consistency_loss(student, teacher, reliability_low_first, omega_u)
        loss.backward()
        grad = logits_s.grad.clone()
        self.assertGreater(grad.abs().sum().item(), 0.0)


class StrongIntensityAugmentTests(unittest.TestCase):
    class Args:
        strong_brightness = 0.2
        strong_brightness_prob = 1.0
        strong_contrast = 0.2
        strong_contrast_prob = 1.0
        strong_noise_std = 0.05
        strong_noise_prob = 1.0

    def test_preserves_shape_and_changes_image_2d(self):
        torch.manual_seed(0)
        image = torch.rand(2, 1, 8, 8)
        augmented = strong_intensity_augment(image, self.Args())
        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(torch.allclose(augmented, image))

    def test_preserves_shape_and_changes_image_3d(self):
        torch.manual_seed(0)
        image = torch.rand(2, 1, 4, 8, 8)
        augmented = strong_intensity_augment(image, self.Args())
        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(torch.allclose(augmented, image))

    def test_all_probabilities_zero_is_identity(self):
        class NoAugArgs(self.Args):
            strong_brightness_prob = 0.0
            strong_contrast_prob = 0.0
            strong_noise_prob = 0.0

        image = torch.rand(1, 1, 8, 8)
        augmented = strong_intensity_augment(image, NoAugArgs())
        torch.testing.assert_close(augmented, image)


class RandomCrop3DTests(unittest.TestCase):
    def test_output_shape_matches_patch_size_for_every_key(self):
        rng = np.random.default_rng(0)
        arrays = {
            "image": rng.random((6, 6, 6)).astype(np.float32),
            "label": np.zeros((6, 6, 6), dtype=np.int64),
            "fold": np.full((6, 6, 6), -1, dtype=np.int64),
        }
        cval = {"image": 0.0, "label": 3, "fold": -1}
        out = random_crop_3d(arrays, cval, patch_size=(4, 4, 4), foreground_prob=0.0, num_classes=3)
        for value in out.values():
            self.assertEqual(value.shape, (4, 4, 4))

    def test_foreground_bias_includes_the_foreground_voxel(self):
        arrays = {
            "image": np.zeros((10, 10, 10), dtype=np.float32),
            "label": np.zeros((10, 10, 10), dtype=np.int64),
        }
        arrays["label"][5, 5, 5] = 1
        cval = {"image": 0.0, "label": 3}
        found = False
        for _ in range(20):
            out = random_crop_3d(arrays, cval, patch_size=(4, 4, 4), foreground_prob=1.0, num_classes=3)
            if (out["label"] == 1).any():
                found = True
        self.assertTrue(found)

    def test_pads_smaller_than_patch_volumes(self):
        arrays = {"image": np.zeros((2, 2, 2), dtype=np.float32), "label": np.zeros((2, 2, 2), dtype=np.int64)}
        cval = {"image": 0.0, "label": 3}
        out = random_crop_3d(arrays, cval, patch_size=(4, 4, 4), foreground_prob=0.0, num_classes=3)
        self.assertEqual(out["image"].shape, (4, 4, 4))
        self.assertTrue(np.any(out["label"] == 3))  # padding used the cval


class RandomFlipRotate3DTests(unittest.TestCase):
    def test_arrays_stay_pixel_aligned(self):
        rng = np.random.default_rng(0)
        image = rng.random((4, 6, 6)).astype(np.float32)
        marker = np.zeros((4, 6, 6), dtype=np.int64)
        marker[2, 3, 1] = 1
        out = random_flip_rotate_3d({"image": image, "marker": marker})
        marked = np.argwhere(out["marker"] == 1)
        self.assertEqual(len(marked), 1)
        d, h, w = marked[0]
        self.assertEqual(out["image"][d, h, w], image[2, 3, 1])


class RandomFlipRotateResize2DTests(unittest.TestCase):
    def test_resizes_every_key_to_output_size(self):
        image = np.random.default_rng(0).random((12, 16)).astype(np.float32)
        label = np.zeros((12, 16), dtype=np.int64)
        fold = np.full((12, 16), -1, dtype=np.int64)
        arrays = {"image": image, "label": label, "fold": fold}
        cval = {"image": 0.0, "label": 4, "fold": -1}
        for _ in range(10):
            out = random_flip_rotate_resize_2d(dict(arrays), cval, output_size=(20, 24))
            for value in out.values():
                self.assertEqual(value.shape, (20, 24))


if __name__ == "__main__":
    unittest.main()
