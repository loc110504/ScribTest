"""Regression tests for ``select_eval_state_dict`` in test_pce_2d.py/
test_pce_3d.py -- the checkpoint-shape resolver behind ``--eval_target``.

Both evaluators define an identical copy of this function (matching the
repo's existing pattern of small duplicated helpers between the 2D/3D
evaluator pair, e.g. ``save_prediction``), so both are exercised here with
the same table of checkpoint shapes.
"""

import os
import sys
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from test_pce_2d import select_eval_state_dict as select_eval_state_dict_2d  # noqa: E402
from test_pce_3d import select_eval_state_dict as select_eval_state_dict_3d  # noqa: E402


class SelectEvalStateDictTests(unittest.TestCase):
    IMPLEMENTATIONS = (select_eval_state_dict_2d, select_eval_state_dict_3d)

    def test_voxtrust3d_shape_student_default(self):
        # model_state_dict = teacher, student_state_dict = student.
        checkpoint = {"model_state_dict": "teacher_weights", "student_state_dict": "student_weights"}
        for select in self.IMPLEMENTATIONS:
            self.assertEqual(select(checkpoint, "student"), "student_weights")

    def test_voxtrust3d_shape_teacher_opt_in(self):
        checkpoint = {"model_state_dict": "teacher_weights", "student_state_dict": "student_weights"}
        for select in self.IMPLEMENTATIONS:
            self.assertEqual(select(checkpoint, "teacher"), "teacher_weights")

    def test_effdnet_shape_student_default(self):
        # model_state_dict = student (already), ema_state_dict = teacher.
        checkpoint = {"model_state_dict": "student_weights", "ema_state_dict": "teacher_weights"}
        for select in self.IMPLEMENTATIONS:
            self.assertEqual(select(checkpoint, "student"), "student_weights")

    def test_effdnet_shape_teacher_opt_in(self):
        checkpoint = {"model_state_dict": "student_weights", "ema_state_dict": "teacher_weights"}
        for select in self.IMPLEMENTATIONS:
            self.assertEqual(select(checkpoint, "teacher"), "teacher_weights")

    def test_no_teacher_shape_ignores_eval_target(self):
        # pCE/CycleMix/SDT-Net/ModelMix: only one model, no teacher at all.
        checkpoint = {"model_state_dict": "only_weights"}
        for select in self.IMPLEMENTATIONS:
            self.assertEqual(select(checkpoint, "student"), "only_weights")

    def test_no_teacher_shape_rejects_explicit_teacher_request(self):
        checkpoint = {"model_state_dict": "only_weights"}
        for select in self.IMPLEMENTATIONS:
            with self.assertRaisesRegex(ValueError, "no EMA teacher"):
                select(checkpoint, "teacher")


if __name__ == "__main__":
    unittest.main()
