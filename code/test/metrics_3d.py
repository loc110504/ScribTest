"""Shared per-class Dice/HD95/ASSD computation for the ScribbleBench evaluators.

Both ``test_pce_3d.py`` and ``test_dmsps_3d.py`` score predictions the same
way; this module holds that one shared implementation instead of duplicating
it in each script.

Edge-case policy (a class can be entirely absent from ``prediction``,
``target``, or both, since ScribbleBench organs/structures do not appear in
every case):

- Both empty: the class does not apply to this case. Dice/HD95/ASSD are all
  ``NaN`` and the case is excluded from every mean for that class.
- Exactly one empty (a complete miss or a false positive): Dice is a
  well-defined ``0.0`` and stays in the Dice mean. HD95/ASSD have no
  well-defined value (there is no opposing surface to measure a distance to)
  and cannot be computed, but silently excluding them from the mean would
  hide the failure. They are therefore reported as ``NaN`` *and* counted
  separately under ``per_class_missed_cases`` in the aggregate summary, so a
  reader can see how often this happened instead of an artificially rosy
  HD95/ASSD average.
"""

import math
import warnings

import numpy as np

# MedPy 0.4.0's binary metrics call the long-removed ``numpy.bool`` alias
# (removed in NumPy >= 1.24, which this repo pins via numpy==1.26.4).
# Restoring the alias is the standard compatibility shim; it does not change
# any computed value (verified against known Dice/HD95/ASSD results). Even
# probing for the attribute triggers NumPy's own deprecation warning, hence
# the explicit filter around it.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "bool"):
        np.bool = bool  # noqa: A001 - required MedPy 0.4.0 / NumPy 1.26 compatibility shim

from medpy import metric  # noqa: E402


def per_class_metrics(prediction, target, class_id, spacing):
    """Dice/HD95/ASSD for one foreground class of one case.

    Args:
        prediction: ``[D, H, W]`` integer array of predicted class ids.
        target: ``[D, H, W]`` integer array of ground-truth class ids.
        class_id: the foreground class to score.
        spacing: physical voxel spacing ``(D, H, W)``, e.g. from the NIfTI
            header, used so HD95/ASSD are reported in millimeters rather
            than voxel counts.

    Returns:
        ``{"dice": float, "hd95": float, "assd": float, "status": str}``,
        ``status`` in ``{"ok", "one_empty", "both_empty"}`` (see module
        docstring for the NaN/exclusion policy).
    """
    pred_mask = prediction == class_id
    truth_mask = target == class_id
    pred_empty = not pred_mask.any()
    truth_empty = not truth_mask.any()

    if pred_empty and truth_empty:
        return {"dice": math.nan, "hd95": math.nan, "assd": math.nan, "status": "both_empty"}
    if pred_empty or truth_empty:
        return {"dice": 0.0, "hd95": math.nan, "assd": math.nan, "status": "one_empty"}

    dice = float(2 * np.count_nonzero(pred_mask & truth_mask) / (pred_mask.sum() + truth_mask.sum()))
    hd95 = float(metric.binary.hd95(pred_mask, truth_mask, voxelspacing=spacing))
    assd = float(metric.binary.assd(pred_mask, truth_mask, voxelspacing=spacing))
    return {"dice": dice, "hd95": hd95, "assd": assd, "status": "ok"}


def finite_mean(values):
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


def summarize_case(case_name, prediction, target, num_classes, spacing):
    """Per-case Dice/HD95/ASSD over every foreground class."""
    per_class = {
        str(class_id): per_class_metrics(prediction, target, class_id, spacing)
        for class_id in range(1, num_classes)
    }
    return {
        "case": case_name,
        "mean_foreground_dice": finite_mean(entry["dice"] for entry in per_class.values()),
        "mean_foreground_hd95": finite_mean(entry["hd95"] for entry in per_class.values()),
        "mean_foreground_assd": finite_mean(entry["assd"] for entry in per_class.values()),
        "per_class_dice": {key: entry["dice"] for key, entry in per_class.items()},
        "per_class_hd95": {key: entry["hd95"] for key, entry in per_class.items()},
        "per_class_assd": {key: entry["assd"] for key, entry in per_class.items()},
        "per_class_status": {key: entry["status"] for key, entry in per_class.items()},
    }


def aggregate_summary(cases, num_classes):
    """Combine per-case ``summarize_case`` results into the evaluator summary."""
    class_ids = [str(class_id) for class_id in range(1, num_classes)]
    per_class_missed_cases = {
        class_id: sum(1 for case in cases if case["per_class_status"][class_id] == "one_empty")
        for class_id in class_ids
    }
    per_class_evaluated_cases = {
        class_id: sum(1 for case in cases if case["per_class_status"][class_id] == "ok")
        for class_id in class_ids
    }
    return {
        "scribblebench_mean_dice": finite_mean(case["mean_foreground_dice"] for case in cases),
        "scribblebench_mean_hd95": finite_mean(case["mean_foreground_hd95"] for case in cases),
        "scribblebench_mean_assd": finite_mean(case["mean_foreground_assd"] for case in cases),
        "per_class_dice": {
            class_id: finite_mean(case["per_class_dice"][class_id] for case in cases)
            for class_id in class_ids
        },
        "per_class_hd95": {
            class_id: finite_mean(case["per_class_hd95"][class_id] for case in cases)
            for class_id in class_ids
        },
        "per_class_assd": {
            class_id: finite_mean(case["per_class_assd"][class_id] for case in cases)
            for class_id in class_ids
        },
        "per_class_missed_cases": per_class_missed_cases,
        "per_class_evaluated_cases": per_class_evaluated_cases,
        "num_cases": len(cases),
    }
