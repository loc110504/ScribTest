"""CPU regression checks for the NeSy-Scrib utilities."""

import os
import sys
import unittest

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from train.train_nesyscrib_2d import nesyscrib_step  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from utils.nesyscrib import (  # noqa: E402
    LV,
    MYO,
    RV,
    adjacency_violation,
    enclosure_violation,
    repair_reliability_map,
    scribble_anchored_keep,
    symbolic_repair,
    teacher_confidence,
    weighted_pixel_ce_loss,
)

IGNORE_INDEX = 4


class ScribbleAnchoredKeepTests(unittest.TestCase):
    def test_keeps_scribble_anchored_component_over_larger_one(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:2, 0:2] = True  # small, scribble-anchored component
        mask[5:9, 5:9] = True  # larger, unanchored component
        scribble = np.zeros((10, 10), dtype=bool)
        scribble[0, 0] = True

        kept = scribble_anchored_keep(mask, scribble)
        self.assertTrue(kept[0:2, 0:2].all())
        self.assertFalse(kept[5:9, 5:9].any())

    def test_falls_back_to_largest_component_without_scribble(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0:2, 0:2] = True
        mask[5:9, 5:9] = True
        no_scribble = np.zeros((10, 10), dtype=bool)

        kept = scribble_anchored_keep(mask, no_scribble)
        self.assertFalse(kept[0:2, 0:2].any())
        self.assertTrue(kept[5:9, 5:9].all())

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((5, 5), dtype=bool)
        kept = scribble_anchored_keep(mask, None)
        self.assertFalse(kept.any())


class SymbolicRepairTests(unittest.TestCase):
    def _ring_case(self):
        # A 20x20 slice: MYO forms a ring around a genuine LV cavity, plus a
        # spurious extra hole elsewhere in MYO that does not overlap LV --
        # placed with a 1-pixel MYO buffer on every side so it is a distinct
        # 8-connected hole component from the LV cavity, not merged with it.
        pred = np.zeros((20, 20), dtype=np.int64)
        pred[4:16, 4:16] = MYO
        pred[8:12, 8:12] = LV  # the genuine ring cavity
        pred[13:15, 13:15] = 0  # a spurious hole punched into MYO (not overlapping LV)
        scribble = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)
        return pred, scribble

    def test_myo_ring_keeps_genuine_cavity_but_fills_spurious_hole(self):
        pred, scribble = self._ring_case()
        repaired, changed = symbolic_repair(pred, scribble, IGNORE_INDEX)

        self.assertTrue((repaired[8:12, 8:12] == LV).all())  # genuine cavity preserved
        self.assertTrue((repaired[13:15, 13:15] == MYO).all())  # spurious hole filled
        self.assertTrue(changed[13:15, 13:15].all())

    def test_disconnected_debris_is_dropped_without_scribble_anchor(self):
        pred = np.zeros((20, 20), dtype=np.int64)
        pred[2:6, 2:6] = RV  # the real RV component
        pred[15:17, 15:17] = RV  # small disconnected debris, no scribble
        scribble = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)

        repaired, changed = symbolic_repair(pred, scribble, IGNORE_INDEX)
        self.assertTrue((repaired[2:6, 2:6] == RV).all())
        self.assertTrue((repaired[15:17, 15:17] == 0).all())
        self.assertTrue(changed[15:17, 15:17].all())

    def test_scribble_anchored_debris_is_kept(self):
        pred = np.zeros((20, 20), dtype=np.int64)
        pred[2:6, 2:6] = RV
        pred[15:17, 15:17] = RV
        scribble = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)
        scribble[15, 15] = RV  # human annotation anchors the smaller component

        repaired, _ = symbolic_repair(pred, scribble, IGNORE_INDEX)
        self.assertTrue((repaired[15:17, 15:17] == RV).all())

    def test_scribble_pixels_are_never_overridden(self):
        pred = np.zeros((20, 20), dtype=np.int64)
        pred[5:10, 5:10] = MYO
        scribble = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)
        scribble[7, 7] = LV  # disagrees with the teacher's own prediction there

        repaired, _ = symbolic_repair(pred, scribble, IGNORE_INDEX)
        self.assertEqual(repaired[7, 7], LV)

    def test_no_change_when_already_anatomically_valid(self):
        pred = np.zeros((20, 20), dtype=np.int64)
        pred[4:16, 4:16] = MYO
        pred[8:12, 8:12] = LV
        scribble = np.full((20, 20), IGNORE_INDEX, dtype=np.int64)

        repaired, changed = symbolic_repair(pred, scribble, IGNORE_INDEX)
        np.testing.assert_array_equal(repaired, pred)
        self.assertFalse(changed.any())


class DiagnosticRuleTests(unittest.TestCase):
    def test_enclosure_violation_zero_when_myo_surrounds_lv(self):
        # A small (2x2) LV hole fully bordered by MYO on every side: a single
        # dilation step of MYO reaches every LV pixel, so the "dilate(MYO)
        # covers LV" ratio this diagnostic checks is exactly 1.
        myo = np.zeros((20, 20), dtype=bool)
        myo[4:16, 4:16] = True
        lv = np.zeros((20, 20), dtype=bool)
        lv[9:11, 9:11] = True
        myo[9:11, 9:11] = False
        self.assertEqual(enclosure_violation(myo, lv), 0.0)

    def test_enclosure_violation_positive_when_lv_isolated(self):
        myo = np.zeros((20, 20), dtype=bool)
        myo[0:2, 0:2] = True
        lv = np.zeros((20, 20), dtype=bool)
        lv[15:18, 15:18] = True
        self.assertGreater(enclosure_violation(myo, lv), 0.0)

    def test_enclosure_violation_zero_when_lv_absent(self):
        myo = np.zeros((20, 20), dtype=bool)
        lv = np.zeros((20, 20), dtype=bool)
        self.assertEqual(enclosure_violation(myo, lv), 0.0)

    def test_adjacency_violation_when_both_present_but_far_apart(self):
        a = np.zeros((20, 20), dtype=bool)
        a[0:2, 0:2] = True
        b = np.zeros((20, 20), dtype=bool)
        b[15:17, 15:17] = True
        self.assertEqual(adjacency_violation(a, b), 1.0)

    def test_adjacency_satisfied_when_touching(self):
        a = np.zeros((20, 20), dtype=bool)
        a[0:5, 0:5] = True
        b = np.zeros((20, 20), dtype=bool)
        b[5:10, 0:5] = True
        self.assertEqual(adjacency_violation(a, b), 0.0)

    def test_adjacency_zero_when_one_class_absent(self):
        a = np.zeros((20, 20), dtype=bool)
        b = np.zeros((20, 20), dtype=bool)
        self.assertEqual(adjacency_violation(a, b), 0.0)


class ReliabilityTests(unittest.TestCase):
    def test_full_trust_when_nothing_changed_and_no_violation(self):
        changed = np.zeros((10, 10), dtype=bool)
        reliability = repair_reliability_map(changed, diagnostic_violation=0.0, sigma=3.0)
        np.testing.assert_allclose(reliability, 1.0)

    def test_zero_trust_exactly_at_an_edit(self):
        changed = np.zeros((10, 10), dtype=bool)
        changed[5, 5] = True
        reliability = repair_reliability_map(changed, diagnostic_violation=0.0, sigma=3.0)
        self.assertAlmostEqual(reliability[5, 5], 0.0, places=5)
        self.assertGreater(reliability[0, 0], reliability[5, 5])

    def test_diagnostic_violation_discounts_everywhere(self):
        changed = np.zeros((10, 10), dtype=bool)
        reliability = repair_reliability_map(changed, diagnostic_violation=1.0, sigma=3.0)
        np.testing.assert_allclose(reliability, 0.0)

    def test_sigma_must_be_positive(self):
        with self.assertRaises(ValueError):
            repair_reliability_map(np.zeros((3, 3), dtype=bool), 0.0, sigma=0.0)

    def test_teacher_confidence_bounds_and_ordering(self):
        confident = torch.zeros(1, 4, 2, 2)
        confident[0, 0] = 10.0
        confident = torch.softmax(confident, dim=1)
        uncertain = torch.full((1, 4, 2, 2), 0.25)

        conf_score = teacher_confidence(confident)
        uncertain_score = teacher_confidence(uncertain)
        self.assertTrue(torch.all(conf_score > uncertain_score))
        self.assertTrue(torch.all(uncertain_score >= 0.0))
        self.assertTrue(torch.all(conf_score <= 1.0))


class WeightedPixelCeLossTests(unittest.TestCase):
    def test_zero_weight_gives_zero_loss(self):
        logits = torch.randn(1, 4, 4, 4, requires_grad=True)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        weight = torch.zeros(1, 4, 4)
        loss = weighted_pixel_ce_loss(logits, target, weight)
        self.assertEqual(loss.item(), 0.0)

    def test_gradients_flow_only_where_weighted(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 4, 4, 4, requires_grad=True)
        target = torch.zeros(1, 4, 4, dtype=torch.long)
        weight = torch.zeros(1, 4, 4)
        weight[0, 0, 0] = 1.0
        loss = weighted_pixel_ce_loss(logits, target, weight)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, :, 0, 0].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[0, :, 3, 3].abs().sum().item(), 0.0)


class NesyscribStepTests(unittest.TestCase):
    def test_step_runs_and_gradients_reach_only_the_student(self):
        torch.manual_seed(0)
        model = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        model_ema = UNet2D(in_chns=1, class_num=4, feature_chns=(4, 8, 16, 24, 32))
        model_ema.load_state_dict(model.state_dict())
        for parameter in model_ema.parameters():
            parameter.requires_grad_(False)

        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 4, (2, 32, 32)),
        }
        batch["label"][:, :4, :4] = IGNORE_INDEX  # leave most pixels unlabeled

        class Args:
            noise_std = 0.1
            repair_sigma = 3.0

        loss_scrib, loss_pseudo, diagnostics = nesyscrib_step(
            model, model_ema, batch, torch.device("cpu"), IGNORE_INDEX, Args()
        )
        total = loss_scrib + loss_pseudo
        total.backward()

        self.assertTrue(torch.isfinite(loss_scrib))
        self.assertTrue(torch.isfinite(loss_pseudo))
        self.assertIn("mean_R", diagnostics)
        self.assertGreaterEqual(diagnostics["mean_R"], 0.0)
        self.assertLessEqual(diagnostics["mean_R"], 1.0)
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model_ema.parameters()))


if __name__ == "__main__":
    unittest.main()
