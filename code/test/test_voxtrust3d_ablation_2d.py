"""Dice / accepted pseudo-label accuracy (PL-Acc) / pseudo-label coverage
(PL-Cov) evaluator for Table 2 ("Ablating Trust Calibration",
``paper_icassp2027/main.tex``) on ACDC/MSCMR, covering every row: the
pCE-only baseline (``train_pce_2d.py``), the original method's four ladder
arms (``full``/``class_only``/``global_confidence``/``all_pseudo_labels``,
from the unmodified ``train_voxtrust3d_2d.py``), and the three new
design-choice knockouts
(``train_voxtrust3d_2d_ablation_{raw_ratio,extrapolate,top1conf}.py``).

New, standalone evaluator -- does not modify ``test_pce_2d.py`` or any
training script. It fills the gap ``run_voxtrust3d_dcc_ablation.sh``'s own
header comment names explicitly: that script only ever produced the Dice
column via ``test_pce_2d.py``; this script produces Dice, PL-Acc, and
PL-Cov together, with the *same* accept-rule each checkpoint was trained
with re-applied at evaluation time. PL-Acc and PL-Cov are the accuracy and
coverage halves of the accept-rule's own accuracy-coverage (risk-coverage)
trade-off for selective prediction: PL-Acc alone can look arbitrarily good
by accepting almost nothing, so the two are always reported together (an
earlier revision of this evaluator additionally reported pixel-wise ECE;
that diagnostic answers a different question -- is the raw predictor's own
softmax confidence calibrated -- than "is DCC's *accept-rule* trustworthy
and how much does it cover", which is what Table 2 ablates, so it was
dropped from this table).

**Evaluation set: the held-out validation split, not the official test
split.** ``imagesTs``/``labelsTs`` never carry scribble annotations in this
benchmark (only ``imagesTr``/``labelsTr`` do), but PL-Acc/PL-Cov and the
``full``/``class_only`` rows' accept-rule all need a real scribble geometry
to measure "distance to the nearest supervised pixel of the predicted
class" and "was the accepted prediction actually correct" against. The
validation split (drawn from ``imagesTr``, held out from gradient training,
exactly the group ``build_val_dataset``/``resolve_case_split`` already use
for checkpoint selection) is the only case pool that has both real scribble
annotations *and* dense reference labels, so it is what this diagnostic
runs against for every row, keeping Dice/PL-Acc/PL-Cov mutually comparable
within one table. This mirrors the paper's own note that these diagnostics
never affect training or checkpoint selection -- they are read only after
the checkpoint is already picked.

**Reconstructing the accept-rule without touching any checkpoint schema.**
No training script here was modified to persist extra fields (per this
project's convention of keeping ablations in new files). Instead, whatever
this evaluator needs but a checkpoint does not carry is rebuilt
deterministically from ``checkpoint["args"]`` alone:

- The distance-stratum bin edges and per-class ``d_max`` (``full`` row
  only) are refit from scratch by rebuilding the *training* split's
  ``VoxTrustSlice2DDataset`` with the checkpoint's own recorded
  ``seed``/``holdout_fraction``/``distance_strata`` -- bit-identical to
  what the original training run computed at startup, since that
  partition is a pure function of those three values.
- The calibrated threshold table is refit from the checkpoint's saved
  ``calibrator_state`` (the raw rolling records) via
  ``RollingCalibrationBuffer.fit_thresholds``, using whichever
  ``calibration_estimator`` this row's ablation specifies.
- Each validation image's own full scribble (no further held-out split
  needed at evaluation time) stands in as its "Omega_sup" for the
  per-pixel transfer-distance query.

See ``ABLATION_ROW_CONFIG`` below for how a checkpoint's ``training_method``
(and, for the original method, its saved ``--ablation``) maps to the
calibration estimator / abstention policy / reliability signal replayed.
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
from networks.unet_2d import UNet2D  # noqa: E402
from train.train_pce_2d import resolve_case_split  # noqa: E402
from train.train_voxtrust3d_2d import VoxTrustSlice2DDataset  # noqa: E402
from utils.voxtrust3d import (  # noqa: E402
    RollingCalibrationBuffer,
    batch_transfer_distance,
    build_class_trees,
    build_pseudo_targets,
    extrapolate_thresholds,
    fit_distance_bins,
    global_threshold_pseudo_targets,
    reliability_score,
    select_reliability,
    spatially_blocked_partition,
    unconditional_pseudo_targets,
)

# Python auto-adds this script's own directory (code/test/) to sys.path, so
# the sibling module resolves as a bare import (avoids clashing with the
# stdlib `test` package that `from test.metrics_3d import ...` would hit).
from metrics_3d import aggregate_summary, summarize_case  # noqa: E402

# training_method -> (ablation ladder arm, calibration estimator, abstain
# policy, reliability signal). "voxtrust3d" (the unmodified
# train_voxtrust3d_2d.py) reads its ladder arm from the checkpoint's own
# saved --ablation instead of a fixed value here.
ABLATION_ROW_CONFIG = {
    "voxtrust3d_ablation_raw_ratio": {
        "ablation": "full", "estimator": "raw", "abstain_policy": "abstain", "signal": "margin_agreement",
    },
    "voxtrust3d_ablation_extrapolate": {
        "ablation": "full", "estimator": "wilson", "abstain_policy": "extrapolate", "signal": "margin_agreement",
    },
    "voxtrust3d_ablation_top1conf": {
        "ablation": "full", "estimator": "wilson", "abstain_policy": "abstain", "signal": "top1_confidence",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Dice/PL-Acc/PL-Cov for one Table 2 row (pCE-only, a VoxTrust-3D/DCC ladder arm, "
            "or a design-choice knockout)"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--case_limit", type=int, default=None)
    return parser.parse_args()


def pseudo_label_accuracy_and_coverage(accepted_correct, accepted_total, candidate_total):
    """PL-Acc (Eq. ``eq:placc``) and PL-Cov (Eq. ``eq:plcov``), the accuracy
    and coverage of one accept-rule's selected pseudo-labels.

    Args:
        accepted_correct: number of accepted pixels whose prediction matched
            the dense reference label.
        accepted_total: ``|A_eval|``, number of accepted pixels.
        candidate_total: ``|Omega_u,eval|``, number of unlabeled candidate
            pixels the accept-rule could have accepted.
    Returns:
        ``(pl_acc, pl_cov)``, each a percentage in ``[0, 100]`` or ``None``
        when undefined (``pl_acc`` needs at least one accepted pixel;
        ``pl_cov`` needs at least one candidate pixel).
    """
    if accepted_correct > accepted_total:
        raise ValueError("accepted_correct cannot exceed accepted_total")
    if accepted_total > candidate_total:
        raise ValueError("accepted_total cannot exceed candidate_total")
    pl_acc = (100.0 * accepted_correct / accepted_total) if accepted_total > 0 else None
    pl_cov = (100.0 * accepted_total / candidate_total) if candidate_total > 0 else None
    return pl_acc, pl_cov


def _build_unet2d(model_config, device):
    return UNet2D(
        in_chns=model_config["in_chns"],
        class_num=model_config["class_num"],
        feature_chns=tuple(model_config["feature_chns"]),
    ).to(device)


def _resize_prob_to_native(prob_patch, native_shape):
    """Per-channel bilinear resize back to native resolution, renormalized
    to sum to 1 per pixel (bilinear interpolation of separate class channels
    does not preserve that on its own)."""
    patch_h, patch_w = prob_patch.shape[1:]
    native_h, native_w = native_shape
    resized = zoom(prob_patch, (1.0, native_h / patch_h, native_w / patch_w), order=1)
    resized = np.clip(resized, 0.0, None)
    resized = resized / resized.sum(axis=0, keepdims=True).clip(min=1e-8)
    return resized.astype(np.float32)


def resolve_row_config(checkpoint):
    training_method = checkpoint.get("training_method", "pce")
    has_calibrator = checkpoint.get("calibrator_state") is not None
    if not has_calibrator:
        return "plain", None
    if training_method in ABLATION_ROW_CONFIG:
        cfg = ABLATION_ROW_CONFIG[training_method]
    elif training_method == "voxtrust3d":
        ckpt_args = checkpoint.get("args", {})
        cfg = {
            "ablation": ckpt_args.get("ablation", "full"),
            "estimator": "wilson",
            "abstain_policy": "abstain",
            "signal": "margin_agreement",
        }
    else:
        raise ValueError(
            "Checkpoint has a calibrator_state but an unrecognized training_method: {}".format(training_method)
        )
    return cfg["ablation"], cfg


def evaluate(args):
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("schema_version") != 1:
        raise ValueError("checkpoint is not a schema-v1 checkpoint")
    if checkpoint.get("model_name") != "unet_2d":
        raise ValueError("test_voxtrust3d_ablation_2d.py only evaluates plain-UNet2D-backbone checkpoints")

    model_config = checkpoint["model_config"]
    data_config = checkpoint["data_config"]
    dataset_name = data_config["dataset"]
    if dataset_name not in ("ACDC", "MSCMR"):
        raise ValueError("test_voxtrust3d_ablation_2d.py only evaluates ACDC/MSCMR checkpoints")
    patch_size = tuple(int(v) for v in data_config["patch_size_hw"])
    ignore_index = data_config["ignore_index"]
    num_classes = model_config["class_num"]
    ckpt_args = checkpoint.get("args", {})

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    teacher = _build_unet2d(model_config, device)
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    teacher.eval()

    row_kind, cfg = resolve_row_config(checkpoint)
    logging.info("Row kind: %s (checkpoint training_method=%s)", row_kind, checkpoint.get("training_method"))

    student = None
    if row_kind != "plain":
        student = _build_unet2d(model_config, device)
        student.load_state_dict(checkpoint["student_state_dict"], strict=True)
        student.eval()

    raw_train_dataset = ScribbleBench2DDataset(
        dataset_name, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    train_case_indices, val_case_indices, _, _, _ = resolve_case_split(raw_train_dataset.cases, dataset_name)

    calibrator = None
    thresholds_t = class_only_t = edges_t = d_max_t = None
    global_confidence_threshold = float(ckpt_args.get("global_confidence_threshold", 0.75))

    if row_kind in ("full", "class_only"):
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
        if cfg["abstain_policy"] == "extrapolate":
            thresholds_np = extrapolate_thresholds(thresholds_np)
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

    case_records = []
    accepted_correct = 0
    accepted_total = 0
    candidate_total = 0
    partition_rng = np.random.default_rng(0)  # holdout_fraction=0.0 below never actually draws from this

    count = len(val_case_indices) if args.case_limit is None else min(args.case_limit, len(val_case_indices))
    for position in tqdm(range(count), desc="{} validation ({})".format(dataset_name, row_kind)):
        case_index = val_case_indices[position]
        sample = val_dataset[case_index]
        image = sample["image"]
        scribble = sample["label"]
        gt = sample["gt_label"]
        spacing = np.asarray(sample["spacing"], dtype=np.float64)
        in_plane_spacing = spacing[1:]

        depth, height, width = image.shape
        pred_volume = np.zeros((depth, height, width), dtype=np.int64)

        for d in range(depth):
            resized = zoom(image[d], (patch_size[0] / height, patch_size[1] / width), order=0)
            tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                teacher_prob_patch = F.softmax(teacher(tensor), dim=1)[0].cpu().numpy()
                if student is not None:
                    student_prob_patch = F.softmax(student(tensor), dim=1)[0].cpu().numpy()
                else:
                    student_prob_patch = teacher_prob_patch

            teacher_prob_native = _resize_prob_to_native(teacher_prob_patch, (height, width))
            student_prob_native = _resize_prob_to_native(student_prob_patch, (height, width))

            teacher_pred_native = np.argmax(teacher_prob_native, axis=0)
            pred_volume[d] = teacher_pred_native

            gt_slice = gt[d]
            correct_native = teacher_pred_native == gt_slice

            if row_kind == "plain":
                continue

            scribble_slice = scribble[d]
            omega_u_native = scribble_slice == ignore_index
            candidate_total += int(omega_u_native.sum())
            omega_u_t = torch.from_numpy(omega_u_native).unsqueeze(0)
            teacher_prob_t = torch.from_numpy(teacher_prob_native).unsqueeze(0)
            student_prob_t = torch.from_numpy(student_prob_native).unsqueeze(0)

            if row_kind == "all_pseudo_labels":
                pseudo = unconditional_pseudo_targets(teacher_prob_t, omega_u_t)
            elif row_kind == "global_confidence":
                rel = reliability_score(student_prob_t, teacher_prob_t)
                reliability_t = select_reliability(rel, "margin_agreement")
                pseudo = global_threshold_pseudo_targets(teacher_prob_t, omega_u_t, reliability_t, global_confidence_threshold)
            else:  # "full" or "class_only"
                rel = reliability_score(student_prob_t, teacher_prob_t)
                reliability_t = select_reliability(rel, cfg["signal"])
                teacher_pred_t = rel["teacher_pred"]

                sup_coords, _, _ = spatially_blocked_partition(
                    scribble_slice, ignore_index, num_classes, 0.0, partition_rng
                )
                tree = build_class_trees(sup_coords, in_plane_spacing)
                coord_h, coord_w = np.indices((height, width))
                coord_batch = np.stack([coord_h, coord_w], axis=0)[None].astype(np.int64)
                distance_np = batch_transfer_distance(
                    teacher_pred_t.numpy(), coord_batch, omega_u_native[None], [tree], in_plane_spacing[None]
                )
                pseudo = build_pseudo_targets(
                    teacher_prob=teacher_prob_t,
                    omega_u=omega_u_t,
                    distance=torch.from_numpy(distance_np),
                    teacher_pred=teacher_pred_t,
                    reliability=reliability_t,
                    stratum_edges=edges_t,
                    thresholds_table=thresholds_t,
                    class_only_thresholds=class_only_t,
                    d_max=d_max_t,
                    distance_conditioning=(row_kind == "full"),
                    abstain_policy=cfg["abstain_policy"],
                )

            accept_native = pseudo["mask"].squeeze(0).squeeze(0).bool().numpy()
            accepted_total += int(accept_native.sum())
            accepted_correct += int((accept_native & correct_native).sum())

        case_records.append(summarize_case(sample["case"], pred_volume, gt, num_classes, spacing))

    summary = aggregate_summary(case_records, num_classes)
    pl_acc, pl_cov = pseudo_label_accuracy_and_coverage(accepted_correct, accepted_total, candidate_total)

    output_dir = Path(
        args.output_dir or REPO_ROOT / "results" / "ScribbleBench_VoxTrust3D_dcc_ablation" / dataset_name / row_kind
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "dataset": dataset_name,
        "row_kind": row_kind,
        "training_method": checkpoint.get("training_method"),
        "eval_split": "held_out_validation",
        "mean_dice": summary["scribblebench_mean_dice"],
        "pl_acc": pl_acc,
        "pl_cov": pl_cov,
        "num_cases": summary["num_cases"],
        "accepted_pixels": accepted_total,
        "candidate_pixels": candidate_total,
    }

    def json_safe(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({key: json_safe(value) for key, value in payload.items()}, handle, indent=2, allow_nan=False)

    def fmt(value):
        return "n/a" if value is None else "{:.6f}".format(value)

    print(
        "{} [{}] Dice: {:.6f} | PL-Acc: {} | PL-Cov: {}".format(
            dataset_name, row_kind, payload["mean_dice"], fmt(pl_acc), fmt(pl_cov),
        )
    )
    print("Metrics: {}".format(output_dir / "metrics.json"))
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    evaluate(parse_args())
