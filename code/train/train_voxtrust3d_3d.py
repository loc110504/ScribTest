"""VNet3D + VoxTrust-3D (VoxTrust3D_CVPR2026_Proposed_Method.pdf) on WORD's
full-3D protocol. ACDC/MSCMR train as independent 2D slices instead; see
``train_voxtrust3d_2d.py``.

A single 3D U-Net student is paired with an EMA teacher (Mean Teacher). The
method's contribution is not the network but *which* teacher pseudo-labels
the student is allowed to learn from: the scribbles already present in each
volume are split once, at the start of the run, into a directly-supervised
part (``Omega_sup``) and a held-out calibration part (``Omega_cal``), at
spatial block granularity (Sec. 4.1). A bounded reliability score (teacher
margin x weak-to-strong stability, Sec. 4.2) is calibrated online against
``Omega_cal``, conditioned on the teacher-predicted class and the physical 3D
distance to the nearest ``Omega_sup`` voxel of that class (Sec. 4.3-4.4), via
a Wilson lower-confidence-bound acceptance rule. Only unlabeled voxels that
clear the calibrated threshold receive a soft teacher target (Sec. 4.5). See
``code/utils/voxtrust3d.py`` for the algorithm itself and the engineering
choices made where the paper leaves an implementation detail open (block
definition, KD-tree-based transfer distance, coordinate tracking through
augmentation).

Trust-Advantage EMA (TA-EMA, ``--ta_ema``, this project's own extension, not
part of the base paper): the same ``Omega_cal`` evidence additionally gates
the EMA teacher update itself, throttling how much of each student iterate
the teacher absorbs whenever the student's calibrated accuracy on held-out
scribble voxels does not exceed the teacher's own. See
``utils.voxtrust3d.trust_advantage_alpha``/``RollingAccuracyBuffer`` for the
mechanism and ``voxtrust_step`` below for how it composes with the
pseudo-label calibration above.

``--ablation`` selects among the four arms of Table 2 in
``paper_icassp2027/main.tex`` ("Ablating Trust Calibration"): ``full``
(default, the method described above), ``class_only`` (held-out calibration
without distance conditioning), ``global_confidence`` (one fixed,
uncalibrated reliability cutoff, ``--global_confidence_threshold``), and
``all_pseudo_labels`` (unfiltered Mean Teacher transfer). The latter two
never consult ``Omega_cal``, so pass ``--holdout_fraction 0`` with them --
see ``utils/voxtrust3d.py``'s module docstring and
``code/train/run_voxtrust3d_dcc_ablation.sh`` for the sweep that reproduces
the whole table.

Only sparse labels in ``labelsTr`` (split further into Omega_sup/Omega_cal)
contribute to optimization. Dense training labels are accessed exclusively
for model selection on a patient-level holdout, exactly as in
``train_pce_3d.py``.

Deployed model: the checkpoint saves BOTH nets (``model_state_dict`` = EMA
teacher, ``student_state_dict`` = student), but model selection (which
iteration becomes ``best.pth``) and ``code/test/test_pce_3d.py``'s default
evaluation target are both the **student**, not the teacher -- a deliberate
departure from Sec. 5's own protocol ("only one EMA network is required at
inference; the calibrator is removed"), so that every baseline in this repo
is finally judged by the network that gradient descent actually trained
(EFFDNet/SDT-Net already deployed their student; VoxTrust-3D previously did
not). Pass ``test_pce_3d.py --eval_target teacher`` to recover the old
teacher-based number for comparison.
"""

import argparse
import json
import logging
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_3d import DATASET_CONFIGS, ScribbleBench3DDataset  # noqa: E402
from networks.vnet_3d import VNet3D  # noqa: E402
from train.common_3d import (  # noqa: E402
    atomic_torch_save,
    checkpoint_due,
    guard_fresh_output_dir,
    make_published_split,
    partial_cross_entropy,
    seed_everything,
    seed_worker,
    validate,
)
from utils.ema_optim import WeightEMA  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402
from utils.voxtrust3d import (  # noqa: E402
    RollingAccuracyBuffer,
    RollingCalibrationBuffer,
    batch_transfer_distance,
    bin_distance_by_class,
    build_class_trees,
    build_patch_coordinates,
    build_pseudo_targets,
    choose_patch_origin,
    fit_distance_bins,
    gather_patch,
    global_threshold_pseudo_targets,
    masked_soft_ce_loss,
    random_flip_rotate,
    reliability_score,
    scatter_points_into_patch,
    spatially_blocked_partition,
    stable_seed,
    unconditional_pseudo_targets,
    strong_intensity_augment_3d,
    trust_advantage_alpha,
)

SUPPORTED_DATASETS = ("WORD",)
DEFAULTS = {
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D VNet from ScribbleBench scribbles with VoxTrust-3D"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=3, type=int, default=None, metavar=("D", "H", "W"))
    parser.add_argument("--n_filters", type=int, default=16, help="VNet base channel width")
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument(
        "--early_interval", type=int, default=5000,
        help="eval+checkpoint cadence for iterations <= --late_phase_start",
    )
    parser.add_argument(
        "--late_interval", type=int, default=1000,
        help="eval+checkpoint cadence for iterations > --late_phase_start",
    )
    parser.add_argument(
        "--late_phase_start", type=int, default=20000,
        help="iteration at which the finer --late_interval cadence begins",
    )
    parser.add_argument("--val_overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--max_accumulator_mb", type=int, default=1024)
    parser.add_argument("--temp_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--foreground_crop_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)

    # EMA teacher (Eq. 2).
    parser.add_argument("--ema_decay", type=float, default=0.99)

    # Trust-Advantage EMA (this project's own extension; see module docstring).
    parser.add_argument(
        "--ta_ema", type=int, default=1, choices=[0, 1],
        help="gate the EMA teacher update by calibrated student-vs-teacher trust advantage (else fixed ema_decay)",
    )
    parser.add_argument(
        "--ta_ema_min_scale", type=float, default=0.1,
        help="floor scale applied to the EMA update rate when the student lacks a calibrated trust advantage",
    )

    # Spatially blocked scribble calibration (Sec. 4.1, Eq. 3).
    parser.add_argument(
        "--holdout_fraction", type=float, default=0.15, help="eta; 0 disables the Omega_cal split entirely"
    )

    # Class- and distance-conditioned risk calibration (Sec. 4.3-4.4, Eq. 8-15).
    parser.add_argument("--distance_strata", type=int, default=3, help="B")
    parser.add_argument("--target_precision", type=float, default=0.95, help="rho")
    parser.add_argument("--wilson_delta", type=float, default=0.05, help="delta")
    parser.add_argument("--calibration_min_samples", type=int, default=32, help="n_min")
    parser.add_argument("--calibration_buffer_size", type=int, default=4096, help="N_max")
    parser.add_argument("--calibration_block_cap", type=int, default=64, help="m_max")
    parser.add_argument("--score_grid_points", type=int, default=101, help="|T|, grid over [0, 1]")

    # Table 2 component ablation ("Ablating Trust Calibration",
    # paper_icassp2027/main.tex). See utils/voxtrust3d.py's module docstring.
    parser.add_argument(
        "--ablation",
        default="full",
        choices=["full", "class_only", "global_confidence", "all_pseudo_labels"],
        help=(
            "full: proposed method (held-out calibration + distance conditioning). "
            "class_only: held-out calibration pooled across distance strata. "
            "global_confidence: one fixed, uncalibrated reliability cutoff "
            "(--global_confidence_threshold), no Omega_cal. all_pseudo_labels: "
            "unfiltered Mean Teacher transfer, no Omega_cal. Pass --holdout_fraction 0 "
            "with the latter two, since they never consult Omega_cal."
        ),
    )
    parser.add_argument(
        "--global_confidence_threshold", type=float, default=0.75,
        help="fixed reliability cutoff used only by --ablation global_confidence",
    )

    # Warm-up / pseudo-label ramp-up (Eq. 18; Algorithm 1 lines 7-13).
    parser.add_argument(
        "--warmup_frac", type=float, default=0.1,
        help="fraction of max_iterations trained on L_scrib only, before calibration starts",
    )
    parser.add_argument(
        "--rampup_frac", type=float, default=0.2,
        help="fraction of max_iterations over which lambda(t) sigmoid-ramps after warm-up",
    )
    parser.add_argument("--pseudo_loss_weight", type=float, default=8.0, help="lambda_max")

    # Weak(teacher)/strong(student) intensity augmentation (Sec. 5, "Augmentation").
    parser.add_argument("--use_strong_aug", type=int, default=1, choices=[0, 1])
    parser.add_argument("--strong_brightness", type=float, default=0.2)
    parser.add_argument("--strong_brightness_prob", type=float, default=0.5)
    parser.add_argument("--strong_contrast", type=float, default=0.2)
    parser.add_argument("--strong_contrast_prob", type=float, default=0.5)
    parser.add_argument("--strong_gamma", type=float, default=0.3)
    parser.add_argument("--strong_gamma_prob", type=float, default=0.3)
    parser.add_argument("--strong_noise_std", type=float, default=0.05)
    parser.add_argument("--strong_noise_prob", type=float, default=0.5)
    parser.add_argument("--strong_blur_prob", type=float, default=0.3)
    parser.add_argument("--strong_blur_sigma_min", type=float, default=0.2)
    parser.add_argument("--strong_blur_sigma_max", type=float, default=1.0)
    return parser.parse_args()


def validate_args(args):
    defaults = DEFAULTS[args.dataset]
    args.patch_size = tuple(args.patch_size or defaults["patch_size"])
    args.batch_size = args.batch_size or defaults["batch_size"]
    if args.max_iterations < 1 or args.batch_size < 1:
        raise ValueError("max_iterations and batch_size must be positive")
    if args.n_filters < 1:
        raise ValueError("n_filters must be positive")
    if any(value < 1 for value in args.patch_size):
        raise ValueError("patch_size values must be positive")
    # VNet3D downsamples D/H/W uniformly by 16 (4 stride-2 stages).
    if any(size % 16 for size in args.patch_size):
        raise ValueError("patch_size must be divisible by 16 in DHW order")
    bottleneck_shape = [size // 16 for size in args.patch_size]
    if np.prod(bottleneck_shape) <= 1:
        raise ValueError("patch_size produces a one-voxel bottleneck, which BatchNorm3d cannot use")
    if args.early_interval < 1 or args.late_interval < 1 or args.num_workers < 0:
        raise ValueError("early_interval/late_interval must be positive and num_workers non-negative")
    if args.late_phase_start < 0:
        raise ValueError("late_phase_start must be non-negative")
    if not 0 <= args.val_overlap < 1:
        raise ValueError("val_overlap must satisfy 0 <= val_overlap < 1")
    if not 0 <= args.foreground_crop_prob <= 1:
        raise ValueError("foreground_crop_prob must satisfy 0 <= p <= 1")
    if args.sw_batch_size < 1 or args.max_accumulator_mb < 0:
        raise ValueError("invalid sliding-window settings")
    if args.temp_dir is not None and not Path(args.temp_dir).is_dir():
        raise ValueError("temp_dir does not exist: {}".format(args.temp_dir))
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema_decay must satisfy 0 < alpha < 1")
    if not 0.0 <= args.ta_ema_min_scale <= 1.0:
        raise ValueError("ta_ema_min_scale must be in [0, 1]")
    if not 0.0 <= args.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must satisfy 0 <= eta < 1")
    if args.distance_strata < 1:
        raise ValueError("distance_strata must be >= 1")
    if not 0.0 < args.target_precision <= 1.0:
        raise ValueError("target_precision must satisfy 0 < rho <= 1")
    if not 0.0 < args.wilson_delta < 1.0:
        raise ValueError("wilson_delta must satisfy 0 < delta < 1")
    if args.calibration_min_samples < 1:
        raise ValueError("calibration_min_samples must be positive")
    if args.calibration_buffer_size < 1 or args.calibration_block_cap < 1:
        raise ValueError("calibration_buffer_size/calibration_block_cap must be positive")
    if args.score_grid_points < 2:
        raise ValueError("score_grid_points must be >= 2")
    if not 0.0 <= args.global_confidence_threshold <= 1.0:
        raise ValueError("global_confidence_threshold must be in [0, 1]")
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("warmup_frac must satisfy 0 <= warmup_frac < 1")
    if args.rampup_frac <= 0.0:
        raise ValueError("rampup_frac must be positive")
    if args.pseudo_loss_weight < 0:
        raise ValueError("pseudo_loss_weight must be non-negative")
    return args


# ---------------------------------------------------------------------------
# Dataset wrapper: fixed Omega_sup/Omega_cal partition + coordinate-tracking
# patch sampler (see utils/voxtrust3d.py's module docstring for why this is
# not built on dataloader.scribblebench_3d's RandomCrop3D/RandomFlipRotate3D).
# Defined here, not in utils/, matching this repo's convention of keeping a
# method's own dataset variant next to its training script (e.g. DMSPS's
# ExpandedLabelDataset in train_dmsps_3d.py).
# ---------------------------------------------------------------------------


class VoxTrustPatch3DDataset(Dataset):
    """Serves training patches carrying everything VoxTrust-3D needs.

    Every case's Omega_sup/Omega_cal scribble partition (Eq. 3) is computed
    once, here, at construction time -- not re-randomized per iteration or
    per epoch, matching Sec. 4.1 ("we split the observed scribbles once at
    the beginning of a training run").
    """

    def __init__(self, base_dataset, indices, holdout_fraction, seed, patch_size, foreground_prob=0.0):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.num_classes = base_dataset.num_classes
        self.ignore_index = base_dataset.ignore_index
        self.patch_size = tuple(int(value) for value in patch_size)
        self.foreground_prob = float(foreground_prob)

        self.case_of_index = {}
        self.spacing = {}
        self.sup_coords = {}
        self.cal_coords = {}
        self.cal_block_id = {}
        for index in tqdm(self.indices, desc="VoxTrust-3D scribble partition", leave=False):
            sample = base_dataset[index]
            case = sample["case"]
            self.case_of_index[index] = case
            self.spacing[case] = np.asarray(sample["spacing"], dtype=np.float64)
            rng = np.random.default_rng(stable_seed(seed, case))
            sup_coords, cal_coords, cal_block_id = spatially_blocked_partition(
                sample["label"], self.ignore_index, self.num_classes, holdout_fraction, rng
            )
            self.sup_coords[case] = sup_coords
            self.cal_coords[case] = cal_coords
            self.cal_block_id[case] = cal_block_id

    def __len__(self):
        return len(self.indices)

    def case_trees(self, case):
        return build_class_trees(self.sup_coords[case], self.spacing[case])

    def per_case_partitions(self):
        return [
            {"sup_coords": self.sup_coords[case], "cal_coords": self.cal_coords[case], "spacing": self.spacing[case]}
            for case in self.sup_coords
        ]

    def __getitem__(self, i):
        index = self.indices[i]
        sample = self.base_dataset[index]
        case = sample["case"]
        image = sample["image"]
        raw_label = sample["label"]
        shape = raw_label.shape

        # Everything below gathers only a patch_size-sized window directly
        # out of the native-resolution image/label/calibration-point data
        # (fancy indexing / bounding-box point filtering) -- never a
        # full-volume-sized temporary. WORD volumes reach 512x512x241
        # voxels; padding/coordinate arrays at that size, once per
        # __getitem__ call and per DataLoader worker, is what caused an OOM
        # kill under --num_workers 4 before this was fixed.
        foreground_coords = None
        if self.foreground_prob > 0:
            foreground_coords = np.argwhere((raw_label > 0) & (raw_label < self.num_classes))
        origin = choose_patch_origin(shape, self.patch_size, foreground_coords, self.foreground_prob)

        coord_d, coord_h, coord_w, valid = build_patch_coordinates(shape, origin, self.patch_size)
        image_patch = gather_patch(image, origin, self.patch_size, valid, 0.0).astype(np.float32)
        raw_label_patch = gather_patch(raw_label, origin, self.patch_size, valid, self.ignore_index)

        cal_mask_patch = np.zeros(self.patch_size, dtype=bool)
        block_patch = np.full(self.patch_size, -1, dtype=np.int64)
        for class_id, coords in self.cal_coords[case].items():
            if len(coords) == 0:
                continue
            scatter_points_into_patch(coords, np.ones(len(coords), dtype=bool), origin, self.patch_size, cal_mask_patch)
            scatter_points_into_patch(
                coords, self.cal_block_id[case][class_id], origin, self.patch_size, block_patch
            )

        sup_label_patch = np.where(
            cal_mask_patch | (raw_label_patch == self.ignore_index), self.ignore_index, raw_label_patch
        )
        cal_label_patch = np.where(cal_mask_patch, raw_label_patch, self.ignore_index)

        arrays = {
            "image": image_patch,
            "sup_label": sup_label_patch.astype(np.int64),
            "cal_label": cal_label_patch.astype(np.int64),
            "cal_block": block_patch,
            "coord_d": coord_d,
            "coord_h": coord_h,
            "coord_w": coord_w,
        }
        augmented = random_flip_rotate(arrays)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(augmented["image"], dtype=np.float32)).unsqueeze(0),
            "sup_label": torch.from_numpy(np.ascontiguousarray(augmented["sup_label"], dtype=np.int64)),
            "cal_label": torch.from_numpy(np.ascontiguousarray(augmented["cal_label"], dtype=np.int64)),
            "cal_block": torch.from_numpy(np.ascontiguousarray(augmented["cal_block"], dtype=np.int64)),
            "coord": torch.from_numpy(
                np.ascontiguousarray(
                    np.stack([augmented["coord_d"], augmented["coord_h"], augmented["coord_w"]], axis=0),
                    dtype=np.int64,
                )
            ),
            "case": case,
            "spacing": np.asarray(self.spacing[case], dtype=np.float32),
        }


def create_model(num_classes, n_filters, device):
    return VNet3D(in_chns=1, class_num=num_classes, n_filters=n_filters).to(device)


def checkpoint_payload(
    model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score,
    student_quality=None, teacher_quality=None,
):
    return {
        "schema_version": 1,
        "model_name": "vnet_3d",
        "training_method": "voxtrust3d",
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[args.dataset]["class_names"]),
            "n_filters": args.n_filters,
        },
        "data_config": {
            "dataset": args.dataset,
            "root_path": str(args.root_path) if args.root_path else None,
            "patch_size_dhw": list(args.patch_size),
            "ignore_index": DATASET_CONFIGS[args.dataset]["ignore_index"],
        },
        "global_step": step,
        "best_val_mean_dice": best_score,
        # The deployed model is the EMA teacher (Sec. 5: "only one EMA
        # network is required at inference"), saved under the same key the
        # sibling baselines use so test_pce_3d.py can evaluate it directly.
        "model_state_dict": model_ema.state_dict(),
        "student_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "calibrator_state": calibrator.state_dict(),
        # Trust-Advantage EMA's rolling student/teacher accuracy buffers;
        # None when --ta_ema 0 (nothing to resume).
        "ta_ema_student_state": student_quality.state_dict() if student_quality is not None else None,
        "ta_ema_teacher_state": teacher_quality.state_dict() if teacher_quality is not None else None,
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(
    path, model, model_ema, optimizer, scaler, calibrator, args, split,
    student_quality=None, teacher_quality=None,
):
    checkpoint = torch.load(path, map_location="cpu")
    expected_model = checkpoint.get("model_config", {})
    expected_data = checkpoint.get("data_config", {})
    if expected_model.get("n_filters") != args.n_filters:
        raise ValueError("resume checkpoint n_filters does not match")
    if expected_data.get("dataset") != args.dataset:
        raise ValueError("resume checkpoint dataset does not match")
    if checkpoint.get("split") != split:
        raise ValueError("resume checkpoint train/val split does not match")
    model.load_state_dict(checkpoint["student_state_dict"], strict=True)
    model_ema.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
    if checkpoint.get("calibrator_state") is not None:
        calibrator.load_state_dict(checkpoint["calibrator_state"])
    # A checkpoint saved before TA-EMA existed, or one saved with --ta_ema 0,
    # has no state to resume here -- buffers just start fresh, matching how
    # a freshly warmed-up run would begin gating.
    if student_quality is not None and checkpoint.get("ta_ema_student_state") is not None:
        student_quality.load_state_dict(checkpoint["ta_ema_student_state"])
    if teacher_quality is not None and checkpoint.get("ta_ema_teacher_state") is not None:
        teacher_quality.load_state_dict(checkpoint["ta_ema_teacher_state"])
    return int(checkpoint["global_step"]), float(checkpoint["best_val_mean_dice"])


def configure_logging(output_dir):
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def voxtrust_step(
    model,
    model_ema,
    batch,
    device,
    ignore_index,
    num_classes,
    case_trees,
    edges_t,
    d_max_t,
    calibrator,
    grid,
    args,
    calibration_active,
    student_quality=None,
    teacher_quality=None,
    augment_fn=strong_intensity_augment_3d,
    ablation="full",
):
    """One VoxTrust-3D training iteration (Algorithm 1).

    Every calculation below (partial CE, reliability, transfer distance,
    calibration update, pseudo-target masking) is shape-agnostic over the
    spatial rank, so this same function drives both the 3D volume-patch
    pipeline (``augment_fn=strong_intensity_augment_3d``, the default) and
    the ACDC/MSCMR 2D slice pipeline (``train_voxtrust3d_2d.py`` passes
    ``strong_intensity_augment_2d``) -- only the student's strong-view
    augmentation differs by tensor rank.

    ``student_quality``/``teacher_quality`` (``RollingAccuracyBuffer``, or
    both ``None`` to disable) drive Trust-Advantage EMA: the caller passes
    ``None`` for both when ``args.ta_ema`` is off, in which case
    ``diagnostics["ema_alpha"]`` is always ``args.ema_decay`` and the
    training loop's fixed-rate EMA is unchanged. When enabled,
    ``diagnostics["ema_alpha"]`` is the caller's ``ema_optimizer.step(alpha=...)``
    rate for this iteration (see ``utils.voxtrust3d.trust_advantage_alpha``).

    ``ablation`` selects one of Table 2's four arms
    (``paper_icassp2027/main.tex``, "Ablating Trust Calibration"; see
    ``utils/voxtrust3d.py``'s module docstring for the full description):

    - ``"full"`` (default): the proposed method, unchanged from before this
      parameter existed.
    - ``"class_only"``: still spends ``Omega_cal`` on a Wilson-calibrated
      threshold, but pools it across distance strata
      (``build_pseudo_targets(..., distance_conditioning=False)``).
    - ``"global_confidence"``: a single fixed, uncalibrated reliability
      cutoff (``args.global_confidence_threshold``); ``Omega_cal`` is never
      touched (the caller should train with ``--holdout_fraction 0``, so
      ``cal_label`` is all ``ignore_index`` and this branch is reached
      trivially).
    - ``"all_pseudo_labels"``: unfiltered Mean Teacher pseudo-label transfer
      (every ``Omega_u`` candidate accepted); also never touches
      ``Omega_cal``.

    The last two arms skip the transfer-distance KD-tree query, the
    calibration-buffer update, threshold fitting, and Trust-Advantage EMA
    entirely (there is no ``Omega_cal`` evidence for any of them to use), so
    ``diagnostics["ema_alpha"]`` stays at the fixed ``args.ema_decay`` for
    those two arms even when ``args.ta_ema`` is on.
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
    ta_ema_enabled = student_quality is not None and teacher_quality is not None

    if calibration_active:
        rel = reliability_score(student_prob, teacher_prob)
        teacher_pred = rel["teacher_pred"]
        omega_u = (sup_label == ignore_index) & (cal_label == ignore_index) & (batch["coord"][:, 0].to(device) >= 0)

        if ablation == "all_pseudo_labels":
            # Table 2, "MT, all pseudo-labels": no Omega_cal evidence, no
            # reliability filtering -- skip the transfer-distance query and
            # calibration bookkeeping entirely, they would be unused.
            pseudo = unconditional_pseudo_targets(teacher_prob, omega_u)
            loss_pl = masked_soft_ce_loss(student_logits, pseudo["target"], pseudo["mask"])
            diagnostics.update(
                {
                    "reliability_mean": rel["score"].mean().item(),
                    "accepted_ratio": pseudo["accepted_ratio"].item(),
                }
            )
        elif ablation == "global_confidence":
            # Table 2, "MT, global confidence": one fixed, uncalibrated
            # cutoff -- likewise no Omega_cal evidence needed.
            pseudo = global_threshold_pseudo_targets(
                teacher_prob, omega_u, rel["score"], args.global_confidence_threshold
            )
            loss_pl = masked_soft_ce_loss(student_logits, pseudo["target"], pseudo["mask"])
            diagnostics.update(
                {
                    "reliability_mean": rel["score"].mean().item(),
                    "accepted_ratio": pseudo["accepted_ratio"].item(),
                }
            )
        else:
            # "full" (default) and "class_only": both spend Omega_cal on a
            # Wilson-calibrated threshold; only build_pseudo_targets's
            # distance_conditioning flag below differs between them.
            teacher_pred_np = teacher_pred.detach().cpu().numpy()

            cal_valid = (cal_label != ignore_index).detach().cpu().numpy()
            omega_u_np = omega_u.detach().cpu().numpy()
            candidate_np = cal_valid | omega_u_np

            case_trees_batch = [case_trees[case] for case in cases]
            distance_np = batch_transfer_distance(teacher_pred_np, coord_np, candidate_np, case_trees_batch, spacing_np)
            distance = torch.from_numpy(distance_np).to(device)

            # ---- Algorithm 1, lines 8-10: calibration update on Omega_cal ----
            true_label_np = cal_label.detach().cpu().numpy()
            score_np = rel["score"].detach().cpu().numpy()
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

                # ---- Trust-Advantage EMA: student-vs-teacher accuracy on the
                # same Omega_cal voxels, grouped by TRUE class/distance (see
                # utils.voxtrust3d's TA-EMA section docstring for why this
                # differs from calibrator.update's predicted-class grouping).
                if ta_ema_enabled:
                    true_classes = true_label_np[cal_valid]
                    true_class_distance_np = batch_transfer_distance(
                        true_label_np, coord_np, cal_valid, case_trees_batch, spacing_np
                    )
                    true_bin_ids = bin_distance_by_class(
                        true_class_distance_np[cal_valid], true_classes, edges_np
                    )
                    # Fair comparison requires the student on the same weak view
                    # the teacher sees, not the strong-augmented view it trains
                    # on; eval() avoids a second, redundant BatchNorm running-
                    # stat update from this diagnostic-only forward pass (the
                    # real training forward on student_batch already happened
                    # above and is unaffected by this mode toggle).
                    was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        student_pred_weak_np = model(weak_batch).argmax(dim=1).detach().cpu().numpy()
                    model.train(was_training)
                    student_correct = (student_pred_weak_np[cal_valid] == true_classes).astype(np.float64)

                    student_quality.update(true_classes, true_bin_ids, student_correct)
                    teacher_quality.update(true_classes, true_bin_ids, correct)

            if ta_ema_enabled:
                q_student = student_quality.quality(args.calibration_min_samples, args.wilson_delta)
                q_teacher = teacher_quality.quality(args.calibration_min_samples, args.wilson_delta)
                diagnostics["ema_alpha"] = trust_advantage_alpha(
                    args.ema_decay, q_student, q_teacher, args.ta_ema_min_scale
                )
                diagnostics["ta_ema_q_student"] = q_student if q_student is not None else float("nan")
                diagnostics["ta_ema_q_teacher"] = q_teacher if q_teacher is not None else float("nan")
                if q_student is not None and q_teacher is not None:
                    diagnostics["ta_ema_admitted"] = 1.0 if q_student > q_teacher else 0.0

            thresholds_np, class_only_np = calibrator.fit_thresholds(
                grid, args.calibration_min_samples, args.target_precision, args.wilson_delta
            )
            thresholds_t = torch.from_numpy(thresholds_np).float().to(device)
            class_only_t = torch.from_numpy(class_only_np).float().to(device)

            # ---- Algorithm 1, lines 11-12: pseudo-label on Omega_u ----
            pseudo = build_pseudo_targets(
                teacher_prob=teacher_prob,
                omega_u=omega_u,
                distance=distance,
                teacher_pred=teacher_pred,
                reliability=rel["score"],
                stratum_edges=edges_t,
                thresholds_table=thresholds_t,
                class_only_thresholds=class_only_t,
                d_max=d_max_t,
                distance_conditioning=(ablation != "class_only"),
            )
            loss_pl = masked_soft_ce_loss(student_logits, pseudo["target"], pseudo["mask"])

            diagnostics.update(
                {
                    "reliability_mean": rel["score"].mean().item(),
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


def train(args):
    args = validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_VoxTrust3D" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_fresh_output_dir(output_dir, args.resume)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    raw_train_dataset = ScribbleBench3DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    val_dataset = ScribbleBench3DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", return_full_label=True
    )
    train_indices, val_indices, train_groups, val_groups, protocol = make_published_split(
        raw_train_dataset.samples, args.dataset
    )
    split = {
        "protocol": protocol,
        "grouped_by_patient": True,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "train_cases": [raw_train_dataset.samples[index]["case"] for index in train_indices],
        "val_cases": [raw_train_dataset.samples[index]["case"] for index in val_indices],
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    num_classes = raw_train_dataset.num_classes
    ignore_index = raw_train_dataset.ignore_index

    logging.info("Building the fixed Omega_sup/Omega_cal scribble partition (eta=%.3f)...", args.holdout_fraction)
    train_dataset = VoxTrustPatch3DDataset(
        raw_train_dataset,
        train_indices,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        patch_size=args.patch_size,
        foreground_prob=args.foreground_crop_prob,
    )
    case_trees = {case: train_dataset.case_trees(case) for case in train_dataset.sup_coords}

    logging.info("Fitting distance-stratum bin edges and d_c^max (B=%d)...", args.distance_strata)
    edges_np, d_max_np = fit_distance_bins(
        train_dataset.per_case_partitions(), num_classes, args.distance_strata
    )
    edges_t = torch.from_numpy(edges_np).float().to(device)
    d_max_t = torch.from_numpy(d_max_np).float().to(device)
    logging.info("d_c^max per class: %s", np.round(d_max_np, 2).tolist())

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty")

    model = create_model(num_classes, args.n_filters, device)
    model_ema = create_model(num_classes, args.n_filters, device)
    # Teacher <- student before constructing WeightEMA: with equal initial
    # weights the copy direction of WeightEMA's constructor is irrelevant,
    # sidestepping its documented student<-teacher copy-direction quirk
    # (see utils/sdtnet.py's docstring); this exact pattern is already used
    # by this repo's 2D VoxTrust-style reference (train_sample2d.py).
    model_ema.load_state_dict(model.state_dict())
    for parameter in model_ema.parameters():
        parameter.requires_grad_(False)
    ema_optimizer = WeightEMA(model, model_ema, alpha=args.ema_decay)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    calibrator = RollingCalibrationBuffer(
        num_classes=num_classes,
        num_strata=args.distance_strata,
        buffer_size=args.calibration_buffer_size,
        block_cap=args.calibration_block_cap,
        rng=np.random.default_rng(stable_seed(args.seed, "calibrator")),
    )
    grid = np.linspace(0.0, 1.0, args.score_grid_points)
    student_quality = (
        RollingAccuracyBuffer(num_classes, args.distance_strata, args.calibration_buffer_size)
        if args.ta_ema else None
    )
    teacher_quality = (
        RollingAccuracyBuffer(num_classes, args.distance_strata, args.calibration_buffer_size)
        if args.ta_ema else None
    )

    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(
            args.resume, model, model_ema, optimizer, scaler, calibrator, args, split,
            student_quality=student_quality, teacher_quality=teacher_quality,
        )
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    warmup_iters = int(round(args.warmup_frac * args.max_iterations))
    rampup_iters = max(1, int(round(args.rampup_frac * args.max_iterations)))

    logging.info(
        "dataset=%s classes=%d ignore=%d train=%d val=%d patch=%s device=%s warmup_iters=%d rampup_iters=%d",
        args.dataset, num_classes, ignore_index, len(train_indices), len(val_indices), args.patch_size, device,
        warmup_iters, rampup_iters,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    metrics_path = output_dir / "validation.jsonl"
    last_eval_step = -1

    try:
        while step < args.max_iterations:
            model.train()
            model_ema.train()
            for batch in loader:
                calibration_active = step >= warmup_iters
                lr = args.learning_rate * (1.0 - step / args.max_iterations) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
                )
                with amp_context:
                    loss_scrib, loss_pl, diagnostics = voxtrust_step(
                        model, model_ema, batch, device, ignore_index, num_classes,
                        case_trees, edges_t, d_max_t, calibrator, grid, args, calibration_active,
                        student_quality=student_quality, teacher_quality=teacher_quality,
                        ablation=args.ablation,
                    )
                    pseudo_weight = (
                        args.pseudo_loss_weight * sigmoid_rampup(step - warmup_iters, rampup_iters)
                        if calibration_active
                        else 0.0
                    )
                    loss = loss_scrib + pseudo_weight * loss_pl
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema_optimizer.step(alpha=diagnostics["ema_alpha"])
                step += 1

                writer.add_scalar("train/total", loss.item(), step)
                writer.add_scalar("train/scrib", loss_scrib.item(), step)
                writer.add_scalar("train/pseudo", loss_pl.item(), step)
                writer.add_scalar("train/pseudo_weight", pseudo_weight, step)
                writer.add_scalar("train/learning_rate", lr, step)
                for name, value in diagnostics.items():
                    writer.add_scalar("train/{}".format(name), value, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d loss=%.6f scrib=%.6f pseudo=%.6f pseudo_w=%.4f lr=%.6g",
                        step, args.max_iterations, loss.item(), loss_scrib.item(), loss_pl.item(), pseudo_weight, lr,
                    )

                should_checkpoint = (
                    checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                    or step == args.max_iterations
                )
                if should_checkpoint:
                    # Checkpoint selection is by the STUDENT's validation Dice, since the
                    # student (checkpoint's student_state_dict) is what test_pce_3d.py
                    # evaluates by default; see checkpoint_payload() and this module's
                    # docstring. (This departs from Sec. 5's "only the EMA teacher is
                    # deployed at inference" -- see the docstring for why.)
                    result = validate(model, val_dataset, val_indices, args, device, num_classes)
                    last_eval_step = step
                    score = result["mean_dice"]
                    if not math.isfinite(score):
                        raise RuntimeError("validation mean Dice is not finite")
                    writer.add_scalar("val/mean_dice", score, step)
                    for class_id, class_score in result["per_class_dice"].items():
                        if math.isfinite(class_score):
                            writer.add_scalar("val/dice_class_{}".format(class_id), class_score, step)
                    record = {
                        "global_step": step,
                        "mean_dice": result["mean_dice"],
                        "num_cases": result["num_cases"],
                        "per_class_dice": {
                            key: value if math.isfinite(value) else None
                            for key, value in result["per_class_dice"].items()
                        },
                    }
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    if score > best_score:
                        best_score = score
                        payload = checkpoint_payload(
                            model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score,
                            student_quality=student_quality, teacher_quality=teacher_quality,
                        )
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()
                    model_ema.train()
                    atomic_torch_save(
                        checkpoint_payload(
                            model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score,
                            student_quality=student_quality, teacher_quality=teacher_quality,
                        ),
                        output_dir / "last.pth",
                    )
                if step >= args.max_iterations:
                    break
    finally:
        writer.close()

    if last_eval_step != step:
        raise RuntimeError("final iteration was not validated; checkpoint invariant broken")
    if not (output_dir / "best.pth").is_file() or not (output_dir / "last.pth").is_file():
        raise RuntimeError("expected best.pth and last.pth were not created")
    logging.info("Training complete. best_val_mean_dice=%.6f", best_score)
    return output_dir


if __name__ == "__main__":
    train(parse_args())
