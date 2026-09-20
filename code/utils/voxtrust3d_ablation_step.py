"""Shared training-step function for Table 2's three "w/o ..." design-choice
ablations (``paper_icassp2027/main.tex``, "Ablating Trust Calibration"):
each swaps exactly one internal mechanism of the full VoxTrust-3D/DCC
method (Wilson lower bound, abstention, or the reliability signal), keeping
everything else -- held-out calibration, distance conditioning -- unchanged.

This module is new, standalone code. It does **not** modify
``train/train_voxtrust3d_2d.py`` or ``train/train_voxtrust3d_3d.py``, whose
own ``voxtrust_step`` continues to reproduce only the base method and the
four existing Table 2 ladder arms (``full``/``class_only``/
``global_confidence``/``all_pseudo_labels``), unchanged. Each new
``train_voxtrust3d_2d_ablation_*.py`` script imports :func:`voxtrust_step_knockout`
from here instead.

:func:`voxtrust_step_knockout` is a faithful copy of ``voxtrust_step``'s
``"full"``-ablation branch (``distance_conditioning=True``), verified
line-by-line against that function, with exactly one swap applied at the
one relevant call:

- ``calibration_estimator="raw"`` (default ``"wilson"``): Eq. 15's Wilson
  lower confidence bound is replaced by the raw sample ratio ``k/n`` in
  ``RollingCalibrationBuffer.fit_thresholds``.
- ``abstain_policy="extrapolate"`` (default ``"abstain"``): an abstaining
  cell borrows the nearest non-abstaining bin's threshold
  (``utils.voxtrust3d.extrapolate_thresholds``) and the ``d_max`` "rejected
  rather than extrapolated" cutoff is dropped, both inside
  ``build_pseudo_targets``.
- ``reliability_signal="top1_confidence"`` (default ``"margin_agreement"``):
  ``R_i`` (Eq. 5-7) is replaced by the teacher's plain top-1 softmax
  probability everywhere downstream (``utils.voxtrust3d.select_reliability``).

Trust-Advantage EMA is intentionally not wired in here: it is this
project's own extension to the base method (see
``utils/voxtrust3d.py``'s module docstring), not one of the three
mechanisms Table 2 ablates, so every ablation script built on this module
trains with a fixed-rate EMA teacher update only.
"""

import numpy as np
import torch
import torch.nn.functional as F

from train.common_3d import partial_cross_entropy
from utils.voxtrust3d import (
    batch_transfer_distance,
    build_pseudo_targets,
    extrapolate_thresholds,
    masked_soft_ce_loss,
    reliability_score,
    select_reliability,
)


def voxtrust_step_knockout(
    model,
    model_ema,
    batch,
    device,
    ignore_index,
    case_trees,
    edges_t,
    d_max_t,
    calibrator,
    grid,
    args,
    calibration_active,
    augment_fn,
    calibration_estimator="wilson",
    abstain_policy="abstain",
    reliability_signal="margin_agreement",
):
    """One training iteration; always the full method's distance-conditioned
    branch (no ``all_pseudo_labels``/``global_confidence``/``class_only``
    arms, no Trust-Advantage EMA -- see module docstring), with exactly one
    of ``calibration_estimator``/``abstain_policy``/``reliability_signal``
    swapped away from its default (proposed) value.
    """
    weak_batch = batch["image"].to(device, non_blocking=True)
    sup_label = batch["sup_label"].to(device, non_blocking=True).long()
    cal_label = batch["cal_label"].to(device, non_blocking=True).long()
    cal_block_np = batch["cal_block"].numpy()
    coord_np = batch["coord"].numpy()
    spacing_np = batch["spacing"].numpy()
    cases = batch["case"]

    student_batch = augment_fn(weak_batch, args) if args.use_strong_aug else weak_batch

    with torch.no_grad():
        teacher_logits = model_ema(weak_batch)
        teacher_prob = F.softmax(teacher_logits, dim=1)

    student_logits = model(student_batch)
    student_prob = F.softmax(student_logits, dim=1)

    # Eq. 4: partial CE over Omega_sup only.
    loss_scrib, sup_voxels = partial_cross_entropy(student_logits, sup_label, ignore_index)

    loss_pl = student_logits.new_tensor(0.0)
    diagnostics = {"sup_voxels": sup_voxels.item(), "ema_alpha": args.ema_decay}

    if not calibration_active:
        return loss_scrib, loss_pl, diagnostics

    rel = reliability_score(student_prob, teacher_prob)
    teacher_pred = rel["teacher_pred"]
    reliability = select_reliability(rel, reliability_signal)
    omega_u = (sup_label == ignore_index) & (cal_label == ignore_index) & (batch["coord"][:, 0].to(device) >= 0)

    teacher_pred_np = teacher_pred.detach().cpu().numpy()
    cal_valid = (cal_label != ignore_index).detach().cpu().numpy()
    omega_u_np = omega_u.detach().cpu().numpy()
    candidate_np = cal_valid | omega_u_np

    case_trees_batch = [case_trees[case] for case in cases]
    distance_np = batch_transfer_distance(teacher_pred_np, coord_np, candidate_np, case_trees_batch, spacing_np)
    distance = torch.from_numpy(distance_np).to(device)

    # ---- Algorithm 1, lines 8-10: calibration update on Omega_cal ----
    true_label_np = cal_label.detach().cpu().numpy()
    score_np = reliability.detach().cpu().numpy()
    d_max_np = d_max_t.detach().cpu().numpy()
    edges_np = edges_t.detach().cpu().numpy()

    if cal_valid.any():
        class_ids = teacher_pred_np[cal_valid]
        correct = (teacher_pred_np[cal_valid] == true_label_np[cal_valid]).astype(np.float64)
        reliabilities = score_np[cal_valid]
        block_ids = cal_block_np[cal_valid]
        record_distance = distance_np[cal_valid]

        bin_ids = np.full(len(class_ids), -1, dtype=np.int64)
        dmax_for_class = d_max_np[class_ids]
        use_distance = np.isfinite(record_distance) & np.isfinite(dmax_for_class)
        within_support = use_distance & (record_distance <= dmax_for_class)
        if edges_np.shape[1] > 0:
            local_edges = edges_np[class_ids]
            stratum = (record_distance[:, None] >= local_edges).sum(axis=1)
        else:
            stratum = np.zeros(len(class_ids), dtype=np.int64)
        bin_ids[within_support] = stratum[within_support]

        calibrator.update(
            class_ids=class_ids,
            bin_ids=bin_ids,
            block_ids=block_ids,
            reliabilities=reliabilities,
            corrects=correct,
        )

    thresholds_np, class_only_np = calibrator.fit_thresholds(
        grid, args.calibration_min_samples, args.target_precision, args.wilson_delta,
        estimator=calibration_estimator,
    )
    if abstain_policy == "extrapolate":
        thresholds_np = extrapolate_thresholds(thresholds_np)
    thresholds_t = torch.from_numpy(thresholds_np).float().to(device)
    class_only_t = torch.from_numpy(class_only_np).float().to(device)

    # ---- Algorithm 1, lines 11-12: pseudo-label on Omega_u ----
    pseudo = build_pseudo_targets(
        teacher_prob=teacher_prob,
        omega_u=omega_u,
        distance=distance,
        teacher_pred=teacher_pred,
        reliability=reliability,
        stratum_edges=edges_t,
        thresholds_table=thresholds_t,
        class_only_thresholds=class_only_t,
        d_max=d_max_t,
        distance_conditioning=True,
        abstain_policy=abstain_policy,
    )
    loss_pl = masked_soft_ce_loss(student_logits, pseudo["target"], pseudo["mask"])

    diagnostics.update(
        {
            "reliability_mean": reliability.mean().item(),
            "margin_mean": rel["margin"].mean().item(),
            "stability_mean": rel["stability"].mean().item(),
            "accepted_ratio": pseudo["accepted_ratio"].item(),
            "distance_branch_ratio": pseudo["distance_branch_ratio"].item(),
            "fallback_branch_ratio": pseudo["fallback_branch_ratio"].item(),
            "finite_thresholds": int(np.isfinite(thresholds_np).sum()),
            "finite_class_only_thresholds": int(np.isfinite(class_only_np).sum()),
        }
    )
    return loss_scrib, loss_pl, diagnostics


__all__ = ["voxtrust_step_knockout"]
