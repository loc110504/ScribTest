"""CPU regression checks for the VoxTrust-3D utilities."""

import math
import os
import random
import sys
import unittest

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from utils.voxtrust3d import (
    RollingAccuracyBuffer,
    RollingCalibrationBuffer,
    assign_scribble_blocks,
    batch_transfer_distance,
    bin_distance_by_class,
    build_class_trees,
    build_patch_coordinates,
    build_pseudo_targets,
    choose_patch_origin,
    extrapolate_thresholds,
    fit_distance_bins,
    gather_patch,
    global_threshold_pseudo_targets,
    masked_soft_ce_loss,
    query_transfer_distance,
    random_flip_rotate,
    random_flip_rotate_resize_2d,
    reliability_score,
    scatter_points_into_patch,
    select_reliability,
    spatially_blocked_partition,
    stable_seed,
    strong_intensity_augment_2d,
    strong_intensity_augment_3d,
    trust_advantage_alpha,
    unconditional_pseudo_targets,
    wilson_lower_bound,
)


class AssignScribbleBlocksTests(unittest.TestCase):
    def test_two_slices_two_disjoint_blocks(self):
        label = np.zeros((3, 6, 6), dtype=np.int64)
        label[0, 0:2, 0:2] = 1
        label[1, 4:6, 4:6] = 1
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertEqual(len(blocks[1]), 2)
        sizes = sorted(len(block) for block in blocks[1])
        self.assertEqual(sizes, [4, 4])

    def test_two_disjoint_components_same_slice_are_separate_blocks(self):
        label = np.zeros((1, 10, 10), dtype=np.int64)
        label[0, 0:2, 0:2] = 1
        label[0, 8:10, 8:10] = 1
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertEqual(len(blocks[1]), 2)

    def test_background_class_is_included(self):
        label = np.zeros((2, 4, 4), dtype=np.int64)
        label[0, 0, 0] = 0
        label[1, 3, 3] = 0
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertIn(0, blocks)

    def test_absent_class_has_no_blocks(self):
        label = np.zeros((2, 4, 4), dtype=np.int64)
        blocks = assign_scribble_blocks(label, num_classes=3)
        self.assertEqual(blocks[2], [])

    def test_2d_single_slice_two_disjoint_components_are_separate_blocks(self):
        # A (H, W) label -- one ACDC/MSCMR training slice -- is the 2D
        # pipeline's "volume" for this same block-assignment function.
        label = np.zeros((10, 10), dtype=np.int64)
        label[0:2, 0:2] = 1
        label[8:10, 8:10] = 1
        blocks = assign_scribble_blocks(label, num_classes=2)
        self.assertEqual(len(blocks[1]), 2)
        sizes = sorted(len(block) for block in blocks[1])
        self.assertEqual(sizes, [4, 4])
        self.assertEqual(blocks[1][0].shape[1], 2)  # (h, w) coords, not (d, h, w)

    def test_rejects_unsupported_rank(self):
        with self.assertRaisesRegex(ValueError, "must be"):
            assign_scribble_blocks(np.zeros((2, 2, 2, 2)), num_classes=2)


class SpatiallyBlockedPartitionTests(unittest.TestCase):
    def _label_with_n_blocks(self, n_blocks, class_id=1, ignore_index=2):
        label = np.full((n_blocks, 4, 4), ignore_index, dtype=np.int64)
        for d in range(n_blocks):
            label[d, 0, 0] = class_id
        return label

    def test_sup_and_cal_are_disjoint_and_cover_all_scribble_voxels(self):
        label = self._label_with_n_blocks(10)
        rng = np.random.default_rng(0)
        sup, cal, cal_block = spatially_blocked_partition(label, ignore_index=2, num_classes=2, holdout_fraction=0.3, rng=rng)
        sup_set = {tuple(row) for row in sup[1]}
        cal_set = {tuple(row) for row in cal[1]}
        self.assertEqual(sup_set & cal_set, set())
        expected = {tuple(row) for row in np.argwhere(label == 1)}
        self.assertEqual(sup_set | cal_set, expected)
        self.assertEqual(len(cal[1]), len(cal_block[1]))

    def test_last_block_is_never_held_out(self):
        label = self._label_with_n_blocks(1)
        rng = np.random.default_rng(0)
        sup, cal, _ = spatially_blocked_partition(label, ignore_index=2, num_classes=2, holdout_fraction=0.9, rng=rng)
        self.assertEqual(len(sup[1]), 1)
        self.assertEqual(len(cal[1]), 0)

    def test_deterministic_given_same_seed(self):
        label = self._label_with_n_blocks(20)
        rng_a = np.random.default_rng(stable_seed(7, "caseA"))
        rng_b = np.random.default_rng(stable_seed(7, "caseA"))
        sup_a, cal_a, _ = spatially_blocked_partition(label, 2, 2, 0.3, rng_a)
        sup_b, cal_b, _ = spatially_blocked_partition(label, 2, 2, 0.3, rng_b)
        np.testing.assert_array_equal(sup_a[1], sup_b[1])
        np.testing.assert_array_equal(cal_a[1], cal_b[1])

    def test_holdout_fraction_out_of_range_raises(self):
        label = self._label_with_n_blocks(3)
        with self.assertRaises(ValueError):
            spatially_blocked_partition(label, 2, 2, -0.1, np.random.default_rng(0))
        with self.assertRaises(ValueError):
            spatially_blocked_partition(label, 2, 2, 1.0, np.random.default_rng(0))

    def test_zero_holdout_fraction_holds_out_nothing(self):
        # Used by the global_confidence/all_pseudo_labels ablations (Table 2,
        # paper_icassp2027/main.tex), which never consult Omega_cal and
        # should not waste any scribble on it.
        label = self._label_with_n_blocks(10)
        rng = np.random.default_rng(0)
        sup, cal, cal_block = spatially_blocked_partition(label, ignore_index=2, num_classes=2, holdout_fraction=0.0, rng=rng)
        self.assertEqual(len(cal[1]), 0)
        self.assertEqual(len(cal_block[1]), 0)
        expected = {tuple(row) for row in np.argwhere(label == 1)}
        sup_set = {tuple(row) for row in sup[1]}
        self.assertEqual(sup_set, expected)

    def test_2d_label_produces_two_column_coordinates(self):
        # One ACDC/MSCMR slice: 10 disjoint single-voxel class-1 blocks,
        # spaced 3 columns apart so none are 8-connected to a neighbor.
        label = np.full((1, 30), 2, dtype=np.int64)
        for i in range(10):
            label[0, i * 3] = 1
        rng = np.random.default_rng(0)
        sup, cal, cal_block = spatially_blocked_partition(
            label, ignore_index=2, num_classes=2, holdout_fraction=0.3, rng=rng
        )
        self.assertEqual(sup[1].shape[1], 2)
        self.assertEqual(cal[1].shape[1], 2)
        sup_set = {tuple(row) for row in sup[1]}
        cal_set = {tuple(row) for row in cal[1]}
        self.assertEqual(sup_set & cal_set, set())
        expected = {tuple(row) for row in np.argwhere(label == 1)}
        self.assertEqual(sup_set | cal_set, expected)


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

    def test_2d_matches_brute_force_with_anisotropic_spacing(self):
        # Same scenario, one spatial rank down: (H, W) coordinates and a
        # 2-entry spacing, the ACDC/MSCMR 2D slice pipeline's shape.
        rng = np.random.default_rng(1)
        sup_points = rng.integers(0, 20, size=(15, 2))
        spacing = np.array([1.5, 1.0])
        trees = build_class_trees({0: sup_points}, spacing)

        query_points = rng.integers(0, 20, size=(8, 2))
        class_ids = np.zeros(8, dtype=np.int64)
        distance = query_transfer_distance(trees, class_ids, query_points, spacing)

        physical_sup = sup_points * spacing[None, :]
        physical_query = query_points * spacing[None, :]
        expected = np.array([np.min(np.linalg.norm(physical_sup - q, axis=1)) for q in physical_query])
        np.testing.assert_allclose(distance, expected, atol=1e-6)

    def test_batch_transfer_distance_only_touches_valid_voxels(self):
        sup_points = {0: np.array([[0, 0, 0]])}
        trees = [build_class_trees(sup_points, np.ones(3))]
        predicted_class = np.zeros((1, 2, 2, 2), dtype=np.int64)
        coord = np.stack(np.meshgrid(np.arange(2), np.arange(2), np.arange(2), indexing="ij"), axis=0)[None]
        valid = np.zeros((1, 2, 2, 2), dtype=bool)
        valid[0, 1, 1, 1] = True
        out = batch_transfer_distance(predicted_class, coord, valid, trees, np.ones((1, 3)))
        self.assertTrue(np.isnan(out[0, 0, 0, 0]))
        self.assertAlmostEqual(out[0, 1, 1, 1], math.sqrt(3), places=5)


class FitDistanceBinsTests(unittest.TestCase):
    def test_edges_and_dmax_match_pooled_quantiles(self):
        # class 1: Omega_sup at d=0; Omega_cal voxels at d=1,2,...,9 along one axis.
        sup_coords = {1: np.array([[0, 0, 0]])}
        cal_coords = {1: np.array([[0, 0, k] for k in range(1, 10)])}
        case = {"sup_coords": sup_coords, "cal_coords": cal_coords, "spacing": np.ones(3)}
        edges, d_max = fit_distance_bins([case], num_classes=2, num_strata=3)
        self.assertAlmostEqual(d_max[1], 9.0)
        self.assertEqual(edges.shape, (2, 2))
        self.assertTrue(np.all(np.diff(edges[1]) >= 0))

    def test_class_with_no_evidence_is_nan(self):
        sup_coords = {1: np.array([[0, 0, 0]])}
        cal_coords = {1: np.zeros((0, 3), dtype=np.int64)}
        case = {"sup_coords": sup_coords, "cal_coords": cal_coords, "spacing": np.ones(3)}
        edges, d_max = fit_distance_bins([case], num_classes=2, num_strata=3)
        self.assertTrue(np.isnan(d_max[1]))
        self.assertTrue(np.all(np.isnan(edges[1])))


class ReliabilityScoreTests(unittest.TestCase):
    def test_bounds_and_perfect_agreement_is_maximally_stable(self):
        torch.manual_seed(0)
        teacher_logits = torch.randn(2, 4, 3, 3, 3)
        teacher_prob = torch.softmax(teacher_logits, dim=1)
        result = reliability_score(teacher_prob, teacher_prob)
        self.assertTrue(torch.all(result["stability"] >= 1.0 - 1e-5))
        self.assertTrue(torch.all(result["score"] >= 0.0))
        self.assertTrue(torch.all(result["score"] <= 1.0))

    def test_confident_and_agreeing_prediction_scores_high(self):
        teacher_prob = torch.zeros(1, 3, 1, 1, 1)
        teacher_prob[0, 0] = 1.0
        student_prob = teacher_prob.clone()
        result = reliability_score(student_prob, teacher_prob)
        self.assertGreater(result["score"].item(), 0.99)
        self.assertEqual(result["teacher_pred"].item(), 0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            reliability_score(torch.rand(1, 3, 2, 2, 2), torch.rand(1, 2, 2, 2, 2))


class SelectReliabilityTests(unittest.TestCase):
    def test_margin_agreement_returns_score(self):
        rel = {"score": torch.tensor([0.3]), "teacher_conf": torch.tensor([0.9])}
        torch.testing.assert_close(select_reliability(rel, "margin_agreement"), rel["score"])

    def test_top1_confidence_returns_teacher_conf(self):
        rel = {"score": torch.tensor([0.3]), "teacher_conf": torch.tensor([0.9])}
        torch.testing.assert_close(select_reliability(rel, "top1_confidence"), rel["teacher_conf"])

    def test_unknown_signal_raises(self):
        rel = {"score": torch.tensor([0.3]), "teacher_conf": torch.tensor([0.9])}
        with self.assertRaises(ValueError):
            select_reliability(rel, "bogus")


class WilsonLowerBoundTests(unittest.TestCase):
    def test_matches_hand_computed_value(self):
        # p_hat=0.9, n=100, delta=0.05 (z=1.959964) -> LCB ~= 0.826 (standard reference value).
        lcb = wilson_lower_bound(0.9, 100, delta=0.05)
        self.assertAlmostEqual(float(lcb), 0.826, places=3)

    def test_more_samples_raises_the_bound_at_fixed_precision(self):
        low_n = wilson_lower_bound(0.9, 20, delta=0.05)
        high_n = wilson_lower_bound(0.9, 2000, delta=0.05)
        self.assertGreater(high_n, low_n)

    def test_zero_samples_gives_zero(self):
        self.assertEqual(wilson_lower_bound(0.5, 0, delta=0.05), 0.0)


class RollingCalibrationBufferTests(unittest.TestCase):
    def test_fifo_eviction_at_buffer_size(self):
        buffer = RollingCalibrationBuffer(num_classes=2, num_strata=2, buffer_size=3, block_cap=100)
        for i in range(5):
            buffer.update(
                class_ids=np.array([1]), bin_ids=np.array([0]), block_ids=np.array([i]),
                reliabilities=np.array([0.5]), corrects=np.array([1.0]),
            )
        self.assertEqual(len(buffer._bin_buffer(1, 0)), 3)
        self.assertEqual(len(buffer._class_buffer(1)), 3)

    def test_block_cap_limits_one_blocks_contribution(self):
        buffer = RollingCalibrationBuffer(num_classes=2, num_strata=2, buffer_size=1000, block_cap=5,
                                           rng=np.random.default_rng(0))
        buffer.update(
            class_ids=np.zeros(50, dtype=np.int64), bin_ids=np.zeros(50, dtype=np.int64),
            block_ids=np.zeros(50, dtype=np.int64), reliabilities=np.linspace(0, 1, 50), corrects=np.ones(50),
        )
        self.assertEqual(len(buffer._bin_buffer(0, 0)), 5)

    def test_negative_bin_id_updates_only_class_only_buffer(self):
        buffer = RollingCalibrationBuffer(num_classes=2, num_strata=2, buffer_size=100, block_cap=100)
        buffer.update(
            class_ids=np.array([0]), bin_ids=np.array([-1]), block_ids=np.array([0]),
            reliabilities=np.array([0.7]), corrects=np.array([1.0]),
        )
        self.assertEqual(len(buffer.per_bin), 0)
        self.assertEqual(len(buffer._class_buffer(0)), 1)

    def test_fit_thresholds_picks_minimal_feasible_grid_point(self):
        buffer = RollingCalibrationBuffer(num_classes=1, num_strata=1, buffer_size=1000, block_cap=1000)
        # 100 records: reliability 0.0..0.99, all correct above 0.5, mixed below.
        reliabilities = np.linspace(0.0, 0.99, 100)
        corrects = (reliabilities >= 0.5).astype(np.float64)
        buffer.update(
            class_ids=np.zeros(100, dtype=np.int64), bin_ids=np.zeros(100, dtype=np.int64),
            block_ids=np.arange(100), reliabilities=reliabilities, corrects=corrects,
        )
        grid = np.linspace(0.0, 1.0, 101)
        n_min, rho, delta = 10, 0.9, 0.05
        thresholds, _ = buffer.fit_thresholds(grid, n_min, rho, delta)
        picked = thresholds[0, 0]
        self.assertTrue(math.isfinite(picked))

        # Recompute feasibility directly from the raw records (Eq. 11-15) and
        # check `picked` really is the *minimum* grid point that qualifies --
        # this is a from-scratch cross-check, not a hardcoded magic number.
        def feasible_at(t):
            n_t = int(np.sum(reliabilities >= t))
            k_t = int(np.sum((reliabilities >= t) & (corrects == 1)))
            p_hat = k_t / max(n_t, 1)
            return n_t >= n_min and wilson_lower_bound(p_hat, n_t, delta) >= rho

        self.assertTrue(feasible_at(picked))
        for t in grid[grid < picked - 1e-9]:
            self.assertFalse(feasible_at(t), "grid point {} should not be feasible before {}".format(t, picked))

    def test_abstains_below_min_samples(self):
        buffer = RollingCalibrationBuffer(num_classes=1, num_strata=1, buffer_size=100, block_cap=100)
        buffer.update(
            class_ids=np.zeros(5, dtype=np.int64), bin_ids=np.zeros(5, dtype=np.int64),
            block_ids=np.arange(5), reliabilities=np.ones(5), corrects=np.ones(5),
        )
        grid = np.linspace(0.0, 1.0, 101)
        thresholds, _ = buffer.fit_thresholds(grid, n_min=32, rho=0.95, delta=0.05)
        self.assertTrue(math.isinf(thresholds[0, 0]))

    def test_state_dict_round_trip(self):
        buffer = RollingCalibrationBuffer(num_classes=2, num_strata=2, buffer_size=10, block_cap=10)
        buffer.update(
            class_ids=np.array([0, 1]), bin_ids=np.array([0, 1]), block_ids=np.array([0, 1]),
            reliabilities=np.array([0.6, 0.7]), corrects=np.array([1.0, 0.0]),
        )
        state = buffer.state_dict()
        restored = RollingCalibrationBuffer(num_classes=2, num_strata=2, buffer_size=10, block_cap=10)
        restored.load_state_dict(state)
        self.assertEqual(list(restored._bin_buffer(0, 0)), list(buffer._bin_buffer(0, 0)))
        self.assertEqual(list(restored._class_buffer(1)), list(buffer._class_buffer(1)))

    def test_raw_estimator_accepts_a_cutoff_wilson_would_reject(self):
        # Table 2's "w/o Wilson bound" row: a small, noisy sample (9/10
        # correct) clears rho=0.85 at face value (p_hat=0.9) but not through
        # the Wilson lower bound, which is more conservative at low n.
        buffer = RollingCalibrationBuffer(num_classes=1, num_strata=1, buffer_size=100, block_cap=100)
        reliabilities = np.full(10, 0.9)
        corrects = np.array([1.0] * 9 + [0.0])
        buffer.update(
            class_ids=np.zeros(10, dtype=np.int64), bin_ids=np.zeros(10, dtype=np.int64),
            block_ids=np.arange(10), reliabilities=reliabilities, corrects=corrects,
        )
        grid = np.linspace(0.0, 1.0, 101)
        wilson_thresholds, _ = buffer.fit_thresholds(grid, n_min=5, rho=0.85, delta=0.05, estimator="wilson")
        raw_thresholds, _ = buffer.fit_thresholds(grid, n_min=5, rho=0.85, delta=0.05, estimator="raw")
        self.assertTrue(math.isinf(wilson_thresholds[0, 0]))
        self.assertTrue(math.isfinite(raw_thresholds[0, 0]))

    def test_unknown_estimator_raises(self):
        buffer = RollingCalibrationBuffer(num_classes=1, num_strata=1, buffer_size=10, block_cap=10)
        buffer.update(
            class_ids=np.array([0]), bin_ids=np.array([0]), block_ids=np.array([0]),
            reliabilities=np.array([0.9]), corrects=np.array([1.0]),
        )
        with self.assertRaises(ValueError):
            buffer.fit_thresholds(np.linspace(0, 1, 11), 1, 0.9, 0.05, estimator="bogus")


class ExtrapolateThresholdsTests(unittest.TestCase):
    def test_fills_abstaining_cell_with_nearest_finite_bin(self):
        thresholds = np.array([[0.3, math.inf, 0.7]])
        filled = extrapolate_thresholds(thresholds)
        # Bin 1 is equidistant from bins 0 and 2; argmin ties keep the first
        # (lowest-index) match, matching np.argmin's own tie-breaking.
        self.assertEqual(filled[0, 1], 0.3)
        self.assertEqual(filled[0, 0], 0.3)
        self.assertEqual(filled[0, 2], 0.7)

    def test_row_entirely_abstaining_is_left_unchanged(self):
        thresholds = np.array([[math.inf, math.inf]])
        filled = extrapolate_thresholds(thresholds)
        self.assertTrue(np.all(np.isinf(filled)))

    def test_row_with_no_abstention_is_unchanged(self):
        thresholds = np.array([[0.2, 0.4]])
        filled = extrapolate_thresholds(thresholds)
        np.testing.assert_array_equal(filled, thresholds)

    def test_does_not_mutate_input(self):
        thresholds = np.array([[0.3, math.inf]])
        original = thresholds.copy()
        extrapolate_thresholds(thresholds)
        np.testing.assert_array_equal(thresholds, original)


class BinDistanceByClassTests(unittest.TestCase):
    def test_matches_per_row_searchsorted(self):
        edges = np.array([[1.0, 2.0], [4.0, 5.0]])  # 2 classes, 3 strata each
        distance = np.array([0.5, 1.5, 2.5, 3.5, 6.0])
        class_ids = np.array([0, 0, 0, 1, 1])
        bins = bin_distance_by_class(distance, class_ids, edges)
        np.testing.assert_array_equal(bins, [0, 1, 2, 0, 2])

    def test_single_stratum_is_always_bin_zero(self):
        edges = np.zeros((2, 0))
        bins = bin_distance_by_class(np.array([0.0, 100.0]), np.array([0, 1]), edges)
        np.testing.assert_array_equal(bins, [0, 0])

    def test_no_dmax_cutoff_extrapolates_into_outermost_stratum(self):
        # Unlike build_pseudo_targets, there is no d_max rejection here.
        edges = np.array([[1.0, 2.0]])
        bins = bin_distance_by_class(np.array([1000.0]), np.array([0]), edges)
        self.assertEqual(bins[0], 2)


class RollingAccuracyBufferTests(unittest.TestCase):
    def test_quality_is_none_below_min_samples(self):
        buffer = RollingAccuracyBuffer(num_classes=2, num_strata=2, buffer_size=100)
        buffer.update(np.array([0]), np.array([0]), np.array([1.0]))
        self.assertIsNone(buffer.quality(n_min=5, delta=0.05))

    def test_quality_matches_wilson_lower_bound_once_supported(self):
        buffer = RollingAccuracyBuffer(num_classes=1, num_strata=1, buffer_size=100)
        corrects = np.array([1.0] * 9 + [0.0])  # 9/10 correct
        buffer.update(np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), corrects)
        expected = wilson_lower_bound(0.9, 10, 0.05)
        self.assertAlmostEqual(buffer.quality(n_min=10, delta=0.05), float(expected), places=6)

    def test_quality_averages_only_supported_strata(self):
        buffer = RollingAccuracyBuffer(num_classes=2, num_strata=1, buffer_size=100)
        # class 0: 10 observations (supported); class 1: 2 observations (not supported).
        buffer.update(np.zeros(10, dtype=np.int64), np.zeros(10, dtype=np.int64), np.ones(10))
        buffer.update(np.array([1, 1]), np.array([0, 0]), np.array([0.0, 0.0]))
        expected = wilson_lower_bound(1.0, 10, 0.05)
        self.assertAlmostEqual(buffer.quality(n_min=5, delta=0.05), float(expected), places=6)

    def test_fifo_eviction_at_buffer_size(self):
        buffer = RollingAccuracyBuffer(num_classes=1, num_strata=1, buffer_size=3)
        for value in [1.0, 1.0, 1.0, 0.0, 0.0]:
            buffer.update(np.array([0]), np.array([0]), np.array([value]))
        self.assertEqual(len(buffer.cells[(0, 0)]), 3)
        self.assertEqual(list(buffer.cells[(0, 0)]), [1.0, 0.0, 0.0])

    def test_state_dict_round_trip(self):
        buffer = RollingAccuracyBuffer(num_classes=2, num_strata=2, buffer_size=10)
        buffer.update(np.array([0, 1]), np.array([0, 1]), np.array([1.0, 0.0]))
        restored = RollingAccuracyBuffer(num_classes=2, num_strata=2, buffer_size=10)
        restored.load_state_dict(buffer.state_dict())
        self.assertEqual(list(restored.cells[(0, 0)]), list(buffer.cells[(0, 0)]))
        self.assertEqual(list(restored.cells[(1, 1)]), list(buffer.cells[(1, 1)]))

    def test_load_state_dict_rejects_shape_mismatch(self):
        buffer = RollingAccuracyBuffer(num_classes=2, num_strata=2, buffer_size=10)
        state = buffer.state_dict()
        mismatched = RollingAccuracyBuffer(num_classes=3, num_strata=2, buffer_size=10)
        with self.assertRaises(ValueError):
            mismatched.load_state_dict(state)


class TrustAdvantageAlphaTests(unittest.TestCase):
    def test_falls_back_to_fixed_ema_when_evidence_missing(self):
        self.assertEqual(trust_advantage_alpha(0.99, None, 0.8), 0.99)
        self.assertEqual(trust_advantage_alpha(0.99, 0.8, None), 0.99)
        self.assertEqual(trust_advantage_alpha(0.99, None, None), 0.99)

    def test_student_advantage_uses_full_rate(self):
        # Q_S=0.95 > Q_T=0.80 -> eta = (1-0.99)*0.95 = 0.0095 -> alpha = 0.9905.
        alpha = trust_advantage_alpha(0.99, q_student=0.95, q_teacher=0.80, min_scale=0.1)
        self.assertAlmostEqual(alpha, 1.0 - 0.01 * 0.95, places=8)

    def test_no_advantage_is_throttled_not_frozen(self):
        # Q_S=0.70 <= Q_T=0.90 -> eta = (1-0.99)*0.70*0.1 = 0.0007 -> alpha = 0.9993.
        alpha = trust_advantage_alpha(0.99, q_student=0.70, q_teacher=0.90, min_scale=0.1)
        self.assertAlmostEqual(alpha, 1.0 - 0.01 * 0.70 * 0.1, places=8)
        self.assertLess(alpha, 1.0)  # never a hard freeze (alpha < 1) as long as min_scale > 0

    def test_tie_is_treated_as_no_advantage(self):
        advantage_alpha = trust_advantage_alpha(0.99, 0.8, 0.8, min_scale=0.1)
        no_advantage_alpha = trust_advantage_alpha(0.99, 0.8, 0.80000001, min_scale=0.1)
        self.assertAlmostEqual(advantage_alpha, no_advantage_alpha, places=6)

    def test_min_scale_zero_reproduces_a_hard_freeze(self):
        alpha = trust_advantage_alpha(0.99, q_student=0.1, q_teacher=0.9, min_scale=0.0)
        self.assertEqual(alpha, 1.0)  # eta = 0 -> teacher frozen this step

    def test_alpha_never_drops_below_ema_decay(self):
        # eta_t in [0, 1-ema_decay] for any q_student in [0, 1] and any scale in [0, 1].
        for q_student in np.linspace(0.0, 1.0, 6):
            for q_teacher in np.linspace(0.0, 1.0, 6):
                alpha = trust_advantage_alpha(0.99, float(q_student), float(q_teacher), min_scale=0.1)
                self.assertGreaterEqual(alpha, 0.99 - 1e-9)
                self.assertLessEqual(alpha, 1.0 + 1e-9)

    def test_rejects_min_scale_out_of_range(self):
        with self.assertRaises(ValueError):
            trust_advantage_alpha(0.99, 0.9, 0.5, min_scale=1.5)


class BuildPseudoTargetsTests(unittest.TestCase):
    def _base_tensors(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 3)
        teacher_prob[0, 0] = 1.0  # every voxel predicted class 0
        teacher_pred = torch.zeros(1, 1, 1, 3, dtype=torch.long)
        reliability = torch.full((1, 1, 1, 3), 0.9)
        omega_u = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        stratum_edges = torch.zeros(2, 0)
        return teacher_prob, teacher_pred, reliability, omega_u, stratum_edges

    def test_in_support_accept_uses_stratum_threshold(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[0.5], [math.inf]])
        class_only = torch.tensor([math.inf, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
        )
        self.assertTrue(torch.all(out["mask"] > 0))

    def test_out_of_support_rejects_even_with_high_reliability(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 5.0)  # > d_max
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([0.0, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
        )
        self.assertTrue(torch.all(out["mask"] == 0))

    def test_undefined_distance_falls_back_to_class_only(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), float("nan"))
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([0.5, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
        )
        self.assertTrue(torch.all(out["mask"] > 0))
        self.assertAlmostEqual(out["fallback_branch_ratio"].item(), 1.0)

    def test_class_only_threshold_unavailable_rejects(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), float("nan"))
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([math.inf, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
        )
        self.assertTrue(torch.all(out["mask"] == 0))

    def test_omega_u_false_always_rejects(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        omega_u = torch.zeros_like(omega_u)
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([0.0, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
        )
        self.assertTrue(torch.all(out["mask"] == 0))

    def test_distance_conditioning_false_ignores_in_support_stratum_threshold(self):
        # Table 2 "MT, class-only calibration" ablation: even a voxel that
        # IS within d_max and would pass the per-stratum threshold must be
        # judged solely against class_only_thresholds when
        # distance_conditioning=False.
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)  # well within support
        thresholds_table = torch.tensor([[0.0], [math.inf]])  # per-stratum: would accept (reliability=0.9 >= 0.0)
        class_only = torch.tensor([math.inf, math.inf])  # class-only: abstains
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            distance_conditioning=False,
        )
        self.assertTrue(torch.all(out["mask"] == 0))
        self.assertAlmostEqual(out["distance_branch_ratio"].item(), 0.0)

    def test_distance_conditioning_false_accepts_via_class_only(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[math.inf], [math.inf]])  # per-stratum would reject
        class_only = torch.tensor([0.5, math.inf])  # class-only accepts (reliability=0.9 >= 0.5)
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            distance_conditioning=False,
        )
        self.assertTrue(torch.all(out["mask"] > 0))
        self.assertAlmostEqual(out["fallback_branch_ratio"].item(), 1.0)

    def test_extrapolate_policy_accepts_beyond_d_max(self):
        # Same setup as test_out_of_support_rejects_even_with_high_reliability
        # (distance=5.0 > d_max=2.0), but abstain_policy="extrapolate" drops
        # the d_max cutoff entirely.
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 5.0)
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([0.0, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            abstain_policy="extrapolate",
        )
        self.assertTrue(torch.all(out["mask"] > 0))

    def test_extrapolate_policy_still_rejects_when_stratum_and_class_only_both_abstain(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[math.inf], [math.inf]])
        class_only = torch.tensor([math.inf, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            abstain_policy="extrapolate",
        )
        self.assertTrue(torch.all(out["mask"] == 0))

    def test_extrapolate_policy_falls_back_to_class_only_when_stratum_abstains(self):
        # Stratum threshold is inf (would abstain) but class_only has
        # evidence; abstain_policy="extrapolate" should accept via the
        # class-only fallback instead of rejecting outright.
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[math.inf], [math.inf]])
        class_only = torch.tensor([0.5, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        out_abstain = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            abstain_policy="abstain",
        )
        out_extrapolate = build_pseudo_targets(
            teacher_prob, omega_u, distance, teacher_pred, reliability,
            stratum_edges, thresholds_table, class_only, d_max,
            abstain_policy="extrapolate",
        )
        self.assertTrue(torch.all(out_abstain["mask"] == 0))
        self.assertTrue(torch.all(out_extrapolate["mask"] > 0))

    def test_unknown_abstain_policy_raises(self):
        teacher_prob, teacher_pred, reliability, omega_u, stratum_edges = self._base_tensors()
        distance = torch.full((1, 1, 1, 3), 1.0)
        thresholds_table = torch.tensor([[0.0], [math.inf]])
        class_only = torch.tensor([0.0, math.inf])
        d_max = torch.tensor([2.0, 2.0])
        with self.assertRaises(ValueError):
            build_pseudo_targets(
                teacher_prob, omega_u, distance, teacher_pred, reliability,
                stratum_edges, thresholds_table, class_only, d_max,
                abstain_policy="bogus",
            )


class UnconditionalPseudoTargetsTests(unittest.TestCase):
    def test_accepts_every_omega_u_candidate(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 3)
        teacher_prob[0, 0] = 1.0
        omega_u = torch.tensor([[[[True, False, True]]]])
        out = unconditional_pseudo_targets(teacher_prob, omega_u)
        expected_mask = omega_u.float().unsqueeze(1)
        torch.testing.assert_close(out["mask"], expected_mask)
        self.assertAlmostEqual(out["accepted_ratio"].item(), 1.0)

    def test_target_matches_normalized_teacher_prob(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 1)
        teacher_prob[0, 0] = 1.0
        omega_u = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        out = unconditional_pseudo_targets(teacher_prob, omega_u)
        torch.testing.assert_close(out["target"], teacher_prob)


class GlobalThresholdPseudoTargetsTests(unittest.TestCase):
    def test_below_threshold_rejects(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 3)
        teacher_prob[0, 0] = 1.0
        reliability = torch.full((1, 1, 1, 3), 0.5)
        omega_u = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        out = global_threshold_pseudo_targets(teacher_prob, omega_u, reliability, threshold=0.75)
        self.assertTrue(torch.all(out["mask"] == 0))
        self.assertAlmostEqual(out["accepted_ratio"].item(), 0.0)

    def test_at_or_above_threshold_accepts(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 3)
        teacher_prob[0, 0] = 1.0
        reliability = torch.full((1, 1, 1, 3), 0.75)
        omega_u = torch.ones(1, 1, 1, 3, dtype=torch.bool)
        out = global_threshold_pseudo_targets(teacher_prob, omega_u, reliability, threshold=0.75)
        self.assertTrue(torch.all(out["mask"] > 0))
        self.assertAlmostEqual(out["accepted_ratio"].item(), 1.0)

    def test_omega_u_false_always_rejects(self):
        teacher_prob = torch.zeros(1, 2, 1, 1, 3)
        teacher_prob[0, 0] = 1.0
        reliability = torch.full((1, 1, 1, 3), 1.0)
        omega_u = torch.zeros(1, 1, 1, 3, dtype=torch.bool)
        out = global_threshold_pseudo_targets(teacher_prob, omega_u, reliability, threshold=0.0)
        self.assertTrue(torch.all(out["mask"] == 0))


class MaskedSoftCELossTests(unittest.TestCase):
    def test_zero_when_mask_empty(self):
        logits = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
        target = torch.softmax(torch.randn(1, 3, 2, 2, 2), dim=1)
        mask = torch.zeros(1, 1, 2, 2, 2)
        loss = masked_soft_ce_loss(logits, target, mask)
        self.assertEqual(loss.item(), 0.0)

    def test_matches_manual_computation_and_backprops_through_logits_only(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 3, 1, 1, 2, requires_grad=True)
        target = torch.softmax(torch.randn(1, 3, 1, 1, 2), dim=1)
        mask = torch.ones(1, 1, 1, 1, 2)
        loss = masked_soft_ce_loss(logits, target, mask)

        log_prob = torch.log_softmax(logits, dim=1)
        expected = -(target * log_prob).sum(dim=1, keepdim=True).mean()
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.any(logits.grad != 0))


class StrongIntensityAugment3DTests(unittest.TestCase):
    class Args:
        strong_brightness = 0.2
        strong_brightness_prob = 1.0
        strong_contrast = 0.2
        strong_contrast_prob = 1.0
        strong_gamma = 0.3
        strong_gamma_prob = 1.0
        strong_noise_std = 0.05
        strong_noise_prob = 1.0
        strong_blur_prob = 1.0
        strong_blur_sigma_min = 0.5
        strong_blur_sigma_max = 0.5

    def test_preserves_shape_and_changes_the_image(self):
        torch.manual_seed(0)
        image = torch.rand(2, 1, 4, 8, 8)
        augmented = strong_intensity_augment_3d(image, self.Args())
        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(torch.allclose(augmented, image))

    def test_all_probabilities_zero_is_near_identity(self):
        class NoAugArgs(self.Args):
            strong_brightness_prob = 0.0
            strong_contrast_prob = 0.0
            strong_gamma_prob = 0.0
            strong_noise_prob = 0.0
            strong_blur_prob = 0.0

        image = torch.rand(1, 1, 4, 8, 8)
        augmented = strong_intensity_augment_3d(image, NoAugArgs())
        torch.testing.assert_close(augmented, image)


class StrongIntensityAugment2DTests(unittest.TestCase):
    Args = StrongIntensityAugment3DTests.Args

    def test_preserves_shape_and_changes_the_image(self):
        torch.manual_seed(0)
        image = torch.rand(2, 1, 8, 8)
        augmented = strong_intensity_augment_2d(image, self.Args())
        self.assertEqual(augmented.shape, image.shape)
        self.assertFalse(torch.allclose(augmented, image))

    def test_all_probabilities_zero_is_near_identity(self):
        class NoAugArgs(self.Args):
            strong_brightness_prob = 0.0
            strong_contrast_prob = 0.0
            strong_gamma_prob = 0.0
            strong_noise_prob = 0.0
            strong_blur_prob = 0.0

        image = torch.rand(1, 1, 8, 8)
        augmented = strong_intensity_augment_2d(image, NoAugArgs())
        torch.testing.assert_close(augmented, image)


class ChoosePatchOriginTests(unittest.TestCase):
    def test_no_padding_needed_origin_in_valid_range(self):
        for _ in range(50):
            origin = choose_patch_origin((10, 20, 20), (4, 4, 4))
            for axis, size in enumerate((10, 20, 20)):
                self.assertGreaterEqual(origin[axis], 0)
                self.assertLessEqual(origin[axis] + 4, size)

    def test_patch_larger_than_volume_forces_symmetric_negative_origin(self):
        # shape[0]=10 < patch[0]=16: pad_before = (16-10)//2 = 3, so the
        # single valid origin is exactly -3 (matches RandomCrop3D's symmetric padding).
        origin = choose_patch_origin((10, 20, 20), (16, 4, 4))
        self.assertEqual(origin[0], -3)

    def test_foreground_bias_centers_on_a_foreground_voxel(self):
        foreground_coords = np.array([[5, 5, 5]])
        seen_offsets = set()
        for _ in range(20):
            origin = choose_patch_origin((10, 10, 10), (4, 4, 4), foreground_coords, foreground_prob=1.0)
            for axis in range(3):
                self.assertLessEqual(origin[axis], 5)
                self.assertGreater(origin[axis] + 4, 5)
            seen_offsets.add(tuple(origin))
        self.assertGreater(len(seen_offsets), 1)  # still randomized within the feasible window


class BuildPatchCoordinatesTests(unittest.TestCase):
    def test_valid_flags_match_in_bounds_original_coordinates(self):
        coord_d, coord_h, coord_w, valid = build_patch_coordinates((10, 10, 10), origin=[-3, 2, 2], patch_size=(6, 4, 4))
        # First 3 D-slices are padding (origin -3, -2, -1 < 0); the rest are real.
        self.assertTrue(np.all(~valid[:3]))
        self.assertTrue(np.all(valid[3:]))
        self.assertTrue(np.all(coord_d[:3] == -1))
        np.testing.assert_array_equal(coord_d[3:, 0, 0], np.array([0, 1, 2]))
        # Padding sentinel is consistent across every coordinate channel.
        np.testing.assert_array_equal(coord_d < 0, coord_h < 0)
        np.testing.assert_array_equal(coord_d < 0, coord_w < 0)


class GatherPatchTests(unittest.TestCase):
    def test_matches_direct_slice_when_fully_in_bounds(self):
        array = np.arange(1000, dtype=np.float32).reshape(10, 10, 10)
        origin = [2, 3, 4]
        patch_size = (3, 3, 3)
        _, _, _, valid = build_patch_coordinates(array.shape, origin, patch_size)
        gathered = gather_patch(array, origin, patch_size, valid, fill_value=-1.0)
        expected = array[2:5, 3:6, 4:7]
        np.testing.assert_array_equal(gathered, expected)

    def test_fills_out_of_range_voxels_without_touching_in_range_ones(self):
        array = np.arange(1000, dtype=np.float64).reshape(10, 10, 10)
        origin = [-2, 0, 0]
        patch_size = (4, 2, 2)
        _, _, _, valid = build_patch_coordinates(array.shape, origin, patch_size)
        gathered = gather_patch(array, origin, patch_size, valid, fill_value=-99.0)
        self.assertTrue(np.all(gathered[:2] == -99.0))
        np.testing.assert_array_equal(gathered[2:], array[0:2, 0:2, 0:2])


class ScatterPointsIntoPatchTests(unittest.TestCase):
    def test_only_points_inside_the_window_are_written(self):
        out = np.zeros((4, 4, 4), dtype=np.int64)
        coords = np.array([[5, 5, 5], [6, 1, 1], [20, 20, 20]])  # origin=[5,0,0]; only the 2nd point lands inside
        values = np.array([11, 22, 33])
        scatter_points_into_patch(coords, values, origin=[5, 0, 0], patch_size=(4, 4, 4), out=out)
        self.assertEqual(out[1, 1, 1], 22)
        self.assertEqual(int(out.sum()), 22)

    def test_empty_coords_is_a_no_op(self):
        out = np.zeros((2, 2, 2), dtype=np.int64)
        scatter_points_into_patch(np.zeros((0, 3), dtype=np.int64), np.zeros(0), [0, 0, 0], (2, 2, 2), out)
        self.assertEqual(out.sum(), 0)


class RandomFlipRotateTests(unittest.TestCase):
    def test_arrays_stay_pixel_aligned(self):
        rng = np.random.default_rng(0)
        image = rng.random((4, 6, 6)).astype(np.float32)
        marker = np.zeros((4, 6, 6), dtype=np.int64)
        marker[2, 3, 1] = 1  # a single distinguishable voxel
        out = random_flip_rotate({"image": image, "marker": marker})
        marked = np.argwhere(out["marker"] == 1)
        self.assertEqual(len(marked), 1)
        d, h, w = marked[0]
        self.assertEqual(out["image"][d, h, w], image[2, 3, 1])


class RandomFlipRotateResize2DTests(unittest.TestCase):
    def test_resizes_every_key_to_output_size(self):
        random_state = random.getstate()
        random.seed(0)
        try:
            image = np.random.default_rng(0).random((12, 16)).astype(np.float32)
            label = np.zeros((12, 16), dtype=np.int64)
            coord_h, coord_w = np.indices((12, 16))
            arrays = {"image": image, "label": label, "coord_h": coord_h, "coord_w": coord_w}
            cval = {"image": 0.0, "label": 4, "coord_h": -1, "coord_w": -1}
            for _ in range(10):
                out = random_flip_rotate_resize_2d(dict(arrays), cval, output_size=(20, 24))
                for key, value in out.items():
                    self.assertEqual(value.shape, (20, 24))
        finally:
            random.setstate(random_state)

    def test_rotate_branch_fills_new_corners_with_the_given_cval(self):
        image = np.ones((10, 10), dtype=np.float32)
        label = np.zeros((10, 10), dtype=np.int64)
        arrays = {"image": image, "label": label}
        cval = {"image": 0.0, "label": 4}
        # First draw <= 0.5 skips rot_flip, second draw > 0.5 takes rotate;
        # force a non-trivial angle so the rotation actually introduces corners.
        calls = iter([0.1, 0.9])
        original_random = random.random
        original_randint = random.randint
        random.random = lambda: next(calls, 0.9)
        random.randint = lambda low, high: 15
        try:
            out = random_flip_rotate_resize_2d(arrays, cval, output_size=(10, 10))
        finally:
            random.random = original_random
            random.randint = original_randint
        self.assertIn(4, np.unique(out["label"]))


class PatchPipelineEndToEndTests(unittest.TestCase):
    """choose_patch_origin -> build_patch_coordinates -> gather_patch -> random_flip_rotate,
    exactly as VoxTrustPatch3DDataset.__getitem__ composes them."""

    def test_coordinate_channel_reproduces_source_intensity_after_full_pipeline(self):
        depth, height, width = 6, 6, 6
        image = np.random.default_rng(1).random((depth, height, width)).astype(np.float32)
        origin = choose_patch_origin((depth, height, width), (4, 4, 4))
        coord_d, coord_h, coord_w, valid = build_patch_coordinates((depth, height, width), origin, (4, 4, 4))
        image_patch = gather_patch(image, origin, (4, 4, 4), valid, 0.0)
        augmented = random_flip_rotate(
            {"image": image_patch, "coord_d": coord_d, "coord_h": coord_h, "coord_w": coord_w}
        )
        for d in range(4):
            for h in range(4):
                for w in range(4):
                    od, oh, ow = augmented["coord_d"][d, h, w], augmented["coord_h"][d, h, w], augmented["coord_w"][d, h, w]
                    self.assertGreaterEqual(od, 0)  # (depth,height,width)=(6,6,6) >= patch, so never padded here
                    self.assertEqual(augmented["image"][d, h, w], image[od, oh, ow])

    def test_padding_case_never_allocates_full_volume_sized_arrays(self):
        # WORD-scale regression guard: patch larger than one axis must still
        # only ever produce patch_size-shaped intermediates.
        shape = (10, 20, 20)
        patch_size = (16, 8, 8)
        origin = choose_patch_origin(shape, patch_size)
        coord_d, coord_h, coord_w, valid = build_patch_coordinates(shape, origin, patch_size)
        for array in (coord_d, coord_h, coord_w, valid):
            self.assertEqual(array.shape, patch_size)
        image = np.zeros(shape, dtype=np.float32)
        gathered = gather_patch(image, origin, patch_size, valid, 0.0)
        self.assertEqual(gathered.shape, patch_size)


if __name__ == "__main__":
    unittest.main()
