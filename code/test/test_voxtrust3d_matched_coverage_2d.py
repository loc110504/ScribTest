"""Matched-coverage selector comparison for ``paper_icassp2027/main.tex``'s
"Matched-coverage analysis" paragraph (Sec. 3.3): at the SAME accepted
pseudo-label budget, does EPS's calibrated accept-rule pick a more accurate
subset of pixels than the raw signals it is built from (teacher top-1
confidence, or the uncalibrated reliability score R_i)?

**Why one frozen teacher, not three trained models.** Training a separate
"Confidence-selected" model and an "R_i-selected" model would confound the
comparison with two different learned teachers -- a worse number could then
mean either "this selector is worse" or "this run's teacher happened to be
worse", and the two are not separable after the fact. This script instead
takes the ONE already-trained ``ScribCal (full)`` checkpoint
(``train_voxtrust3d_2d.py --ablation full``, the same checkpoint
``test_voxtrust3d_ablation_2d.py`` scores as the "ScribCal (full)" row of
Table 2) and replays its frozen teacher/student pair over the held-out
validation split exactly once. Confidence, R_i, and EPS's accept/reject
decision are computed from that SAME forward pass -- only the selection
rule applied to those three signals differs.

**Matching coverage.** EPS's accept-rule (Eq. 16, ``build_pseudo_targets``)
is replayed unmodified, giving a target pixel budget ``target_n`` (the total
count of candidates EPS accepts across the whole validation split -- a
global budget, not a per-image quota, to match "annotation/pseudo-label
budget" rather than force every slice individually to the same coverage).
Confidence and R_i are then each restricted to their own top-``target_n``
scoring candidates (ranked over the SAME global pool, no ground truth
involved in ranking) so all three selectors accept exactly the same number
of pixels. Dense labels are used only afterward, to score PL-Acc -- never to
choose a threshold, matching every other diagnostic in this evaluator suite.

**PL-Acc and FG PL-Acc.** Overall PL-Acc is dominated by the easy background
class in cardiac MRI, which compresses all three selectors into a narrow
band near 99.5-99.8% (see Table 2) and can mask a real difference in
selector quality. FG PL-Acc restricts the SAME accuracy computation to
pixels the teacher predicted as a foreground class (RV/Myo/LV, i.e.
``teacher_pred != 0``) -- a harder, class-imbalance-free subset where a
selector's actual quality is expected to show up more clearly.

This script does not train anything and does not modify
``train_voxtrust3d_2d.py`` or ``test_voxtrust3d_ablation_2d.py``; it imports
their already-verified helpers (row-kind resolution, the native-resolution
resize, and the model builder) rather than duplicating them.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import ScribbleBench3DDataset  # noqa: E402
from train.train_pce_2d import resolve_case_split  # noqa: E402
from train.train_voxtrust3d_2d import VoxTrustSlice2DDataset  # noqa: E402
from utils.voxtrust3d import (  # noqa: E402
    RollingCalibrationBuffer,
    batch_transfer_distance,
    build_class_trees,
    build_pseudo_targets,
    fit_distance_bins,
    reliability_score,
    select_reliability,
    spatially_blocked_partition,
)

# This script's own directory (code/test/) is auto-added to sys.path, so
# this sibling module resolves as a bare import (see
# test_voxtrust3d_ablation_2d.py's own module docstring for why).
from test_voxtrust3d_ablation_2d import _build_unet2d, _resize_prob_to_native, resolve_row_config  # noqa: E402

SELECTORS = ("confidence", "margin_agreement_r_i", "eps")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Matched-coverage comparison of Confidence / R_i / EPS pseudo-label selectors on ONE "
            "frozen ScribCal (full) teacher, replayed over the held-out validation split"
        )
    )
    parser.add_argument("--checkpoint", required=True, help="ScribCal (full) checkpoint (--ablation full)")
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    return parser.parse_args()


def topk_mask(scores, k):
    """Boolean mask selecting the top-``k`` entries of ``scores`` (descending,
    ties broken by original order for determinism). No label information is
    used -- this only ranks a signal, exactly like a real deployment-time
    threshold would."""
    if k > scores.shape[0]:
        raise ValueError("k ({}) exceeds pool size ({})".format(k, scores.shape[0]))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(scores.shape[0], dtype=bool)
    mask[order[:k]] = True
    return mask


def selector_metrics(mask, correct_all, fg_all, candidate_total):
    accepted_total = int(mask.sum())
    accepted_correct = int((mask & correct_all).sum())
    fg_accepted_total = int((mask & fg_all).sum())
    fg_accepted_correct = int((mask & fg_all & correct_all).sum())
    return {
        "coverage": 100.0 * accepted_total / candidate_total,
        "pl_acc": (100.0 * accepted_correct / accepted_total) if accepted_total > 0 else None,
        "fg_pl_acc": (100.0 * fg_accepted_correct / fg_accepted_total) if fg_accepted_total > 0 else None,
        "accepted_total": accepted_total,
        "fg_accepted_total": fg_accepted_total,
    }


def evaluate(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint is not a schema-v1 checkpoint")
    if checkpoint.get("model_name") != "unet_2d":
        raise ValueError("test_voxtrust3d_matched_coverage_2d.py only evaluates plain-UNet2D-backbone checkpoints")

    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]
    dataset_name = data_config["dataset"]
    if dataset_name not in ("ACDC", "MSCMR"):
        raise ValueError("test_voxtrust3d_matched_coverage_2d.py only evaluates ACDC/MSCMR checkpoints")
    patch_size = tuple(int(v) for v in data_config["patch_size_hw"])
    ignore_index = data_config["ignore_index"]
    num_classes = model_config["class_num"]
    ckpt_args = checkpoint.get("args", {})

    row_kind, cfg = resolve_row_config(checkpoint)
    if row_kind != "full":
        raise ValueError(
            "test_voxtrust3d_matched_coverage_2d.py compares selectors on ONE frozen ScribCal (full) "
            "teacher; got row_kind='{}' (checkpoint training_method={}). Pass the 'ScribCal (full)' "
            "checkpoint (train_voxtrust3d_2d.py --ablation full).".format(row_kind, checkpoint.get("training_method"))
        )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    teacher = _build_unet2d(model_config, device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()
    student = _build_unet2d(model_config, device)
    student.load_state_dict(checkpoint["student_state_dict"], strict=True)
    student.eval()

    raw_train_dataset = ScribbleBench2DDataset(
        dataset_name, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    train_case_indices, val_case_indices, _, _, _ = resolve_case_split(raw_train_dataset.cases, dataset_name)

    num_strata = int(ckpt_args.get("distance_strata", 3))
    calibrator = RollingCalibrationBuffer(
        num_classes=num_classes,
        num_strata=num_strata,
        buffer_size=int(ckpt_args.get("calibration_buffer_size", 4096)),
        block_cap=int(ckpt_args.get("calibration_block_cap", 64)),
    )
    calibrator.load_state_dict(checkpoint["calibrator_state"])
    grid = np.linspace(0.0, 1.0, int(ckpt_args.get("score_grid_points", 101)))
    thresholds_np, class_only_np = calibrator.fit_thresholds(
        grid,
        int(ckpt_args.get("calibration_min_samples", 32)),
        float(ckpt_args.get("target_precision", 0.95)),
        float(ckpt_args.get("wilson_delta", 0.05)),
        estimator=cfg["estimator"],
    )
    thresholds_t = torch.from_numpy(thresholds_np).float().to(device)
    class_only_t = torch.from_numpy(class_only_np).float().to(device)

    logging.info("Reconstructing distance-stratum bin edges from the training split (deterministic replay)...")
    train_slice_positions = raw_train_dataset.slice_positions_for_volumes(train_case_indices)
    train_partition_dataset = VoxTrustSlice2DDataset(
        raw_train_dataset,
        train_slice_positions,
        holdout_fraction=float(ckpt_args.get("holdout_fraction", 0.15)),
        seed=int(ckpt_args.get("seed", 2026)),
        patch_size=patch_size,
    )
    edges_np, d_max_np = fit_distance_bins(train_partition_dataset.per_case_partitions(), num_classes, num_strata)
    edges_t = torch.from_numpy(edges_np).float().to(device)
    d_max_t = torch.from_numpy(d_max_np).float().to(device)

    val_dataset = ScribbleBench3DDataset(
        dataset_name, base_dir=args.root_path, split="train", sup_type="scribble", return_full_label=True
    )

    partition_rng = np.random.default_rng(0)  # holdout_fraction=0.0 below never actually draws from this

    conf_chunks, rel_chunks, eps_chunks, correct_chunks, fg_chunks = [], [], [], [], []

    count = len(val_case_indices) if args.case_limit is None else min(args.case_limit, len(val_case_indices))
    for position in tqdm(range(count), desc="{} matched-coverage replay".format(dataset_name)):
        case_index = val_case_indices[position]
        sample = val_dataset[case_index]
        image = sample["image"]
        scribble = sample["label"]
        gt = sample["gt_label"]
        spacing = np.asarray(sample["spacing"], dtype=np.float64)
        in_plane_spacing = spacing[1:]

        depth, height, width = image.shape

        for d in range(depth):
            scribble_slice = scribble[d]
            omega_u_native = scribble_slice == ignore_index
            if not omega_u_native.any():
                continue

            resized = zoom(image[d], (patch_size[0] / height, patch_size[1] / width), order=0)
            tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                teacher_prob_patch = F.softmax(teacher(tensor), dim=1)[0].cpu().numpy()
                student_prob_patch = F.softmax(student(tensor), dim=1)[0].cpu().numpy()

            teacher_prob_native = _resize_prob_to_native(teacher_prob_patch, (height, width))
            student_prob_native = _resize_prob_to_native(student_prob_patch, (height, width))

            omega_u_t = torch.from_numpy(omega_u_native).unsqueeze(0).to(device)
            teacher_prob_t = torch.from_numpy(teacher_prob_native).unsqueeze(0).to(device)
            student_prob_t = torch.from_numpy(student_prob_native).unsqueeze(0).to(device)

            rel = reliability_score(student_prob_t, teacher_prob_t)
            reliability_t = select_reliability(rel, "margin_agreement")
            confidence_t = select_reliability(rel, "top1_confidence")
            teacher_pred_t = rel["teacher_pred"]

            sup_coords, _, _ = spatially_blocked_partition(
                scribble_slice, ignore_index, num_classes, 0.0, partition_rng
            )
            tree = build_class_trees(sup_coords, in_plane_spacing)
            coord_h, coord_w = np.indices((height, width))
            coord_batch = np.stack([coord_h, coord_w], axis=0)[None].astype(np.int64)
            distance_np = batch_transfer_distance(
                teacher_pred_t.cpu().numpy(), coord_batch, omega_u_native[None], [tree], in_plane_spacing[None]
            )
            pseudo = build_pseudo_targets(
                teacher_prob=teacher_prob_t,
                omega_u=omega_u_t,
                distance=torch.from_numpy(distance_np).to(device),
                teacher_pred=teacher_pred_t,
                reliability=reliability_t,
                stratum_edges=edges_t,
                thresholds_table=thresholds_t,
                class_only_thresholds=class_only_t,
                d_max=d_max_t,
                distance_conditioning=True,
                abstain_policy="abstain",
            )
            eps_accept_native = pseudo["mask"].squeeze(0).squeeze(0).bool().cpu().numpy()

            teacher_pred_native = teacher_pred_t.squeeze(0).cpu().numpy()
            gt_slice = gt[d]
            correct_native = teacher_pred_native == gt_slice
            fg_native = teacher_pred_native != 0
            confidence_native = confidence_t.squeeze(0).cpu().numpy()
            reliability_native = reliability_t.squeeze(0).cpu().numpy()

            conf_chunks.append(confidence_native[omega_u_native].astype(np.float32))
            rel_chunks.append(reliability_native[omega_u_native].astype(np.float32))
            eps_chunks.append(eps_accept_native[omega_u_native])
            correct_chunks.append(correct_native[omega_u_native])
            fg_chunks.append(fg_native[omega_u_native])

    if not conf_chunks:
        raise RuntimeError("No Omega_u candidates found in the validation split -- nothing to compare")

    conf_all = np.concatenate(conf_chunks)
    rel_all = np.concatenate(rel_chunks)
    eps_all = np.concatenate(eps_chunks)
    correct_all = np.concatenate(correct_chunks)
    fg_all = np.concatenate(fg_chunks)

    candidate_total = conf_all.shape[0]
    target_n = int(eps_all.sum())
    if target_n == 0 or target_n >= candidate_total:
        raise RuntimeError(
            "EPS accepted {} / {} candidates -- matched-coverage comparison needs EPS coverage strictly "
            "between 0% and 100% of the candidate pool".format(target_n, candidate_total)
        )

    conf_mask = topk_mask(conf_all, target_n)
    rel_mask = topk_mask(rel_all, target_n)

    results = {
        "confidence": selector_metrics(conf_mask, correct_all, fg_all, candidate_total),
        "margin_agreement_r_i": selector_metrics(rel_mask, correct_all, fg_all, candidate_total),
        "eps": selector_metrics(eps_all, correct_all, fg_all, candidate_total),
    }

    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "eval_split": "held_out_validation",
        "num_cases": count,
        "candidate_total": candidate_total,
        "target_n": target_n,
        "selectors": results,
    }

    output_dir = Path(
        args.output_dir or REPO_ROOT / "results" / "ScribbleBench_VoxTrust3D_matched_coverage" / dataset_name
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def json_safe(value):
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, allow_nan=False)

    def fmt(value):
        return "n/a" if value is None else "{:.2f}".format(value)

    print("{} matched-coverage selector comparison ({} validation cases, {} candidates):".format(
        dataset_name, count, candidate_total
    ))
    print("{:<24}{:>10}{:>12}{:>14}".format("Selector", "Cov. (%)", "PL-Acc", "FG PL-Acc"))
    labels = {"confidence": "Confidence", "margin_agreement_r_i": "R_i (margin_agreement)", "eps": "EPS (ScribCal)"}
    for key in SELECTORS:
        entry = results[key]
        print("{:<24}{:>10.2f}{:>12}{:>14}".format(
            labels[key], entry["coverage"], fmt(entry["pl_acc"]), fmt(entry["fg_pl_acc"])
        ))
    print("Metrics: {}".format(output_dir / "metrics.json"))
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    evaluate(parse_args())
