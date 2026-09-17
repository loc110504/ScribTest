"""VNet3D + SC-MT (Scribble-Calibrated Mean Teacher) on WORD's full-3D protocol.

A single 3D U-Net student is paired with an EMA teacher (Mean Teacher). The
method's contribution is not the network but *how much* to trust the teacher
at every unlabeled voxel: every connected scribble stroke is assigned once,
at dataset-construction time, to one of ``num_folds`` rotating folds (see
``code/utils/scmt.py`` for the full algorithm). In training epoch ``t`` (one
full pass over the ``DataLoader``), the strokes in fold ``t mod num_folds``
are excluded from the partial-CE loss and instead used as a held-out probe:
since their true label is known, the teacher's prediction there is directly
checkable, and that outcome (correct/incorrect), conditioned on the
teacher's confidence, its predicted class and its physical transfer distance
to the nearest currently-supervised same-class voxel, populates an
EMA-calibrated reliability table. That table's *measured* reliability -- not
raw confidence, not a fixed threshold -- weights the Mean-Teacher consistency
loss on genuinely unlabeled voxels.

Only sparse labels in ``labelsTr`` (further split into this epoch's
supervised/held-out scribble) contribute to optimization. Dense training
labels are accessed exclusively for model selection on a patient-level
holdout, exactly as in ``train_pce_3d.py``. As in every other Mean-Teacher
method in this benchmark, the deployed/checkpointed model is the EMA
teacher, saved in the same schema as the sibling baselines, so
``code/test/test_pce_3d.py`` evaluates ``best.pth`` directly.

ACDC/MSCMR train as independent 2D slices instead; see ``train_scmt_2d.py``,
which imports ``scmt_step`` from here unchanged (every calculation it
performs is shape-agnostic over the spatial rank, see ``utils/scmt.py``).
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
    make_published_split,
    partial_cross_entropy,
    seed_everything,
    seed_worker,
    validate,
)
from utils.ema_optim import WeightEMA  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402
from utils.scmt import (  # noqa: E402
    ReliabilityCalibrationTable,
    assign_block_folds,
    assign_confidence_bin,
    assign_distance_bin,
    batch_transfer_distance_from_labels,
    fit_distance_bin_edges,
    random_crop_3d,
    random_flip_rotate_3d,
    reliability_weighted_consistency_loss,
    stable_seed,
    strong_intensity_augment,
    teacher_confidence,
)

SUPPORTED_DATASETS = ("WORD",)
DEFAULTS = {
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D VNet from ScribbleBench scribbles with SC-MT"
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
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers; note persistent_workers is always disabled for this script "
        "(see train()'s docstring) so this only controls per-epoch parallelism",
    )
    parser.add_argument("--foreground_crop_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)

    # EMA teacher (Mean Teacher).
    parser.add_argument("--ema_decay", type=float, default=0.99)

    # Rotating scribble-block hold-out (Sec. 1 of utils/scmt.py).
    parser.add_argument("--num_folds", type=int, default=5, help="K")

    # Reliability calibration (Sec. 2-3).
    parser.add_argument("--confidence_bins", type=int, default=5)
    parser.add_argument("--distance_bins", type=int, default=4, help="finite bins; a no-support bin is added on top")
    parser.add_argument("--calibration_momentum", type=float, default=0.9, help="EMA momentum for the calibration table")
    parser.add_argument("--calibration_min_samples", type=int, default=10, help="n_min before a table cell is trusted")

    # Warm-up / consistency-loss ramp-up.
    parser.add_argument(
        "--warmup_frac", type=float, default=0.1,
        help="fraction of max_iterations trained on partial-CE only before the consistency loss activates",
    )
    parser.add_argument(
        "--rampup_frac", type=float, default=0.2,
        help="fraction of max_iterations over which the consistency weight sigmoid-ramps after warm-up",
    )
    parser.add_argument("--consistency_weight", type=float, default=1.0, help="lambda_max")

    # Weak(teacher)/strong(student) intensity augmentation.
    parser.add_argument("--use_strong_aug", type=int, default=1, choices=[0, 1])
    parser.add_argument("--strong_brightness", type=float, default=0.2)
    parser.add_argument("--strong_brightness_prob", type=float, default=0.5)
    parser.add_argument("--strong_contrast", type=float, default=0.2)
    parser.add_argument("--strong_contrast_prob", type=float, default=0.5)
    parser.add_argument("--strong_noise_std", type=float, default=0.05)
    parser.add_argument("--strong_noise_prob", type=float, default=0.5)
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
    if args.num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if args.confidence_bins < 1 or args.distance_bins < 1:
        raise ValueError("confidence_bins/distance_bins must be positive")
    if not 0.0 <= args.calibration_momentum < 1.0:
        raise ValueError("calibration_momentum must satisfy 0 <= momentum < 1")
    if args.calibration_min_samples < 1:
        raise ValueError("calibration_min_samples must be positive")
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("warmup_frac must satisfy 0 <= warmup_frac < 1")
    if args.rampup_frac <= 0.0:
        raise ValueError("rampup_frac must be positive")
    if args.consistency_weight < 0:
        raise ValueError("consistency_weight must be non-negative")
    return args


# ---------------------------------------------------------------------------
# Dataset wrapper: per-case rotating scribble-block fold assignment (fixed at
# construction time) + a patch-cropping pipeline that carries the fold
# assignment through as an extra pixel-aligned channel. See
# utils/scmt.py's module docstring for why this needs no coordinate-tracking
# machinery (unlike VoxTrust-3D's VoxTrustPatch3DDataset): SC-MT's transfer
# distance is deliberately patch-local, so only the patch's own supervised
# voxels ever matter, not their position in the original volume.
# ---------------------------------------------------------------------------


class SCMTPatch3DDataset(Dataset):
    """Serves training patches carrying everything SC-MT needs.

    Every case's scribble-block-to-fold assignment (see
    ``utils.scmt.assign_block_folds``) is computed once, here, at
    construction time from sparse scribble coordinates (cheap: connected
    components only ever run over the handful of annotated slices, not the
    whole volume). ``held_out_fold`` is deliberately a *mutable* attribute,
    set by the training loop once per epoch (see ``train()``): which fold is
    held out changes every epoch, unlike VoxTrust-3D's fixed-for-the-run
    partition, so this dataset's ``__getitem__`` must reflect the *current*
    fold, not one computed once and frozen.
    """

    def __init__(self, base_dataset, indices, num_folds, seed, patch_size, foreground_prob=0.0):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.num_folds = num_folds
        self.patch_size = tuple(int(value) for value in patch_size)
        self.foreground_prob = float(foreground_prob)
        self.num_classes = base_dataset.num_classes
        self.ignore_index = base_dataset.ignore_index
        self.held_out_fold = 0

        self.coords_by_class = {}
        self.fold_by_class = {}
        self.spacing = {}
        for index in tqdm(self.indices, desc="SC-MT scribble-block fold assignment", leave=False):
            sample = base_dataset[index]
            case = sample["case"]
            self.spacing[case] = np.asarray(sample["spacing"], dtype=np.float64)
            rng = np.random.default_rng(stable_seed(seed, case))
            coords, folds = assign_block_folds(sample["label"], self.num_classes, num_folds, rng)
            self.coords_by_class[case] = coords
            self.fold_by_class[case] = folds

    def __len__(self):
        return len(self.indices)

    def per_case_partitions(self):
        return [
            {
                "coords_by_class": self.coords_by_class[case],
                "fold_by_class": self.fold_by_class[case],
                "spacing": self.spacing[case],
            }
            for case in self.coords_by_class
        ]

    def __getitem__(self, i):
        index = self.indices[i]
        sample = self.base_dataset[index]
        case = sample["case"]
        image = sample["image"]
        label = sample["label"]

        fold_map = np.full(label.shape, -1, dtype=np.int64)
        for class_id, coords in self.coords_by_class[case].items():
            if len(coords) == 0:
                continue
            fold_map[coords[:, 0], coords[:, 1], coords[:, 2]] = self.fold_by_class[case][class_id]

        arrays = {"image": image.astype(np.float32), "label": label, "fold": fold_map}
        cval = {"image": 0.0, "label": self.ignore_index, "fold": -1}
        cropped = random_crop_3d(
            arrays, cval, self.patch_size, self.foreground_prob, self.num_classes, foreground_key="label"
        )
        augmented = random_flip_rotate_3d(cropped)

        held_mask = augmented["fold"] == self.held_out_fold
        raw_label = augmented["label"]
        sup_label = np.where(held_mask | (raw_label == self.ignore_index), self.ignore_index, raw_label)
        cal_label = np.where(held_mask, raw_label, self.ignore_index)

        return {
            "image": torch.from_numpy(np.ascontiguousarray(augmented["image"], dtype=np.float32)).unsqueeze(0),
            "sup_label": torch.from_numpy(np.ascontiguousarray(sup_label, dtype=np.int64)),
            "cal_label": torch.from_numpy(np.ascontiguousarray(cal_label, dtype=np.int64)),
            "case": case,
            "spacing": np.asarray(self.spacing[case], dtype=np.float32),
        }


def create_model(num_classes, n_filters, device):
    return VNet3D(in_chns=1, class_num=num_classes, n_filters=n_filters).to(device)


def checkpoint_payload(model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "vnet_3d",
        "training_method": "scmt",
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
        # Mean Teacher: the deployed model is the EMA teacher, saved under
        # the same key the sibling baselines use so test_pce_3d.py can
        # evaluate it directly.
        "model_state_dict": model_ema.state_dict(),
        "student_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "calibrator_state": calibrator.state_dict(),
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, model_ema, optimizer, scaler, calibrator, args, split):
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
    return int(checkpoint["global_step"]), float(checkpoint["best_val_mean_dice"])


def configure_logging(output_dir):
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def scmt_step(
    model,
    model_ema,
    batch,
    device,
    ignore_index,
    num_classes,
    distance_edges,
    calibrator,
    args,
    calibration_active,
    augment_fn=strong_intensity_augment,
):
    """One SC-MT training iteration.

    Every calculation below (partial CE, teacher confidence, transfer
    distance, calibration bookkeeping, reliability-weighted consistency
    loss) is shape-agnostic over the spatial rank, so this same function
    drives both the 3D volume-patch pipeline here and the ACDC/MSCMR 2D
    slice pipeline (``train_scmt_2d.py`` imports it unchanged) -- only the
    student's strong-view augmentation ever needs a rank-specific callable,
    and ``strong_intensity_augment`` itself is already rank-agnostic (no
    spatial convolution to special-case, see ``utils/scmt.py``), so even
    ``augment_fn`` never actually changes between the two callers.

    ``calibrator.observe()`` always runs (cheap bookkeeping, giving the
    table a head start during warm-up); only the consistency loss's
    contribution to the total loss is gated by ``calibration_active``
    (``step >= warmup_iters``) by the caller via the returned ``loss_con``'s
    weight, matching this benchmark's other Mean-Teacher methods' warm-up
    convention.
    """
    weak = batch["image"].to(device, non_blocking=True)
    sup_label = batch["sup_label"].to(device, non_blocking=True).long()
    cal_label = batch["cal_label"].to(device, non_blocking=True).long()
    spacing_np = batch["spacing"].numpy()

    student_input = augment_fn(weak, args) if args.use_strong_aug else weak

    with torch.no_grad():
        teacher_logits = model_ema(weak)
        teacher_prob = F.softmax(teacher_logits, dim=1)
    student_logits = model(student_input)
    student_prob = F.softmax(student_logits, dim=1)

    loss_scrib, sup_voxels = partial_cross_entropy(student_logits, sup_label, ignore_index)
    diagnostics = {"sup_voxels": sup_voxels.item()}

    confidence, predicted_class = teacher_confidence(teacher_prob)
    confidence_bin = assign_confidence_bin(confidence, args.confidence_bins)

    omega_u = (sup_label == ignore_index) & (cal_label == ignore_index)
    cal_valid = cal_label != ignore_index
    candidate = omega_u | cal_valid

    sup_label_np = sup_label.detach().cpu().numpy()
    predicted_class_np = predicted_class.detach().cpu().numpy()
    candidate_np = candidate.detach().cpu().numpy()
    distance_np = batch_transfer_distance_from_labels(
        sup_label_np, predicted_class_np, candidate_np, spacing_np, num_classes
    )
    distance = torch.from_numpy(distance_np).to(device)
    distance_bin = assign_distance_bin(distance, predicted_class, distance_edges, args.distance_bins)

    loss_con = student_logits.new_tensor(0.0)
    omega_u_np = omega_u.detach().cpu().numpy()
    if calibration_active and omega_u_np.any():
        confidence_np = confidence.detach().cpu().numpy()
        confidence_bin_np = confidence_bin.detach().cpu().numpy()
        distance_bin_np = distance_bin.detach().cpu().numpy()

        reliability_np = np.zeros(predicted_class_np.shape, dtype=np.float64)
        reliability_np[omega_u_np] = calibrator.query(
            predicted_class_np[omega_u_np],
            confidence_bin_np[omega_u_np],
            distance_bin_np[omega_u_np],
            confidence_np[omega_u_np],
        )
        reliability = torch.from_numpy(reliability_np).float().to(device)
        loss_con = reliability_weighted_consistency_loss(student_prob, teacher_prob, reliability, omega_u)
        diagnostics["mean_reliability"] = float(reliability_np[omega_u_np].mean())
        diagnostics["mean_confidence"] = float(confidence_np[omega_u_np].mean())

    cal_valid_np = cal_valid.detach().cpu().numpy()
    if cal_valid_np.any():
        true_label_np = cal_label.detach().cpu().numpy()
        confidence_bin_np = confidence_bin.detach().cpu().numpy()
        distance_bin_np = distance_bin.detach().cpu().numpy()
        class_ids = predicted_class_np[cal_valid_np]
        corrects = (predicted_class_np[cal_valid_np] == true_label_np[cal_valid_np]).astype(np.float64)
        calibrator.observe(
            class_ids, confidence_bin_np[cal_valid_np], distance_bin_np[cal_valid_np], corrects
        )
        diagnostics["cal_voxels"] = int(cal_valid_np.sum())
        diagnostics["cal_accuracy"] = float(corrects.mean())
    else:
        diagnostics["cal_voxels"] = 0

    return loss_scrib, loss_con, diagnostics


def train(args):
    """Trains and validates SC-MT.

    Epoch bookkeeping: one "epoch" is one full pass over ``loader`` (this
    dataset's ``__len__`` many samples). Before each epoch, the training
    loop sets ``train_dataset.held_out_fold = epoch % args.num_folds`` and
    only then enters ``for batch in loader``, which -- because
    ``persistent_workers`` is deliberately left ``False`` below -- forks
    fresh worker processes off the *current* state of ``train_dataset``, so
    every worker sees this epoch's fold. (With ``persistent_workers=True``,
    already-forked workers would keep using whatever fold was set when they
    were first spawned, silently breaking the rotation; this is a real
    per-epoch worker-respawn cost, traded for correctness.) After the epoch,
    ``calibrator.commit_epoch()`` folds that epoch's held-out observations
    (accumulated per-batch by ``scmt_step``) into the persistent reliability
    table via EMA.
    """
    args = validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_SCMT" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
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

    logging.info("Assigning rotating scribble-block folds (K=%d)...", args.num_folds)
    train_dataset = SCMTPatch3DDataset(
        raw_train_dataset,
        train_indices,
        num_folds=args.num_folds,
        seed=args.seed,
        patch_size=args.patch_size,
        foreground_prob=args.foreground_crop_prob,
    )

    logging.info("Fitting distance-bin quantile edges (B=%d)...", args.distance_bins)
    edges_np = fit_distance_bin_edges(
        train_dataset.per_case_partitions(), num_classes, args.num_folds, args.distance_bins
    )
    edges_t = torch.from_numpy(edges_np).float().to(device)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        # Deliberately not persistent: held_out_fold is mutated once per
        # epoch on the main-process dataset object, and only a fresh fork
        # (which happens on every `iter(loader)` when workers are not
        # persistent) picks that up -- see train()'s docstring.
        persistent_workers=False,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty")

    model = create_model(num_classes, args.n_filters, device)
    model_ema = create_model(num_classes, args.n_filters, device)
    model_ema.load_state_dict(model.state_dict())
    for parameter in model_ema.parameters():
        parameter.requires_grad_(False)
    ema_optimizer = WeightEMA(model, model_ema, alpha=args.ema_decay)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    calibrator = ReliabilityCalibrationTable(
        num_classes=num_classes,
        num_confidence_bins=args.confidence_bins,
        num_distance_bins=args.distance_bins + 1,  # + the dedicated no-support bin
        momentum=args.calibration_momentum,
        min_samples=args.calibration_min_samples,
    )

    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(
            args.resume, model, model_ema, optimizer, scaler, calibrator, args, split
        )
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    warmup_iters = int(round(args.warmup_frac * args.max_iterations))
    rampup_iters = max(1, int(round(args.rampup_frac * args.max_iterations)))
    epoch = step // len(loader)

    logging.info(
        "dataset=%s classes=%d ignore=%d train=%d val=%d patch=%s device=%s warmup_iters=%d rampup_iters=%d "
        "num_folds=%d",
        args.dataset, num_classes, ignore_index, len(train_indices), len(val_indices), args.patch_size, device,
        warmup_iters, rampup_iters, args.num_folds,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    metrics_path = output_dir / "validation.jsonl"
    last_eval_step = -1

    try:
        while step < args.max_iterations:
            model.train()
            model_ema.train()
            train_dataset.held_out_fold = epoch % args.num_folds
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
                    loss_scrib, loss_con, diagnostics = scmt_step(
                        model, model_ema, batch, device, ignore_index, num_classes,
                        edges_t, calibrator, args, calibration_active,
                    )
                    consistency_weight = (
                        args.consistency_weight * sigmoid_rampup(step - warmup_iters, rampup_iters)
                        if calibration_active
                        else 0.0
                    )
                    loss = loss_scrib + consistency_weight * loss_con
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema_optimizer.step()
                step += 1

                writer.add_scalar("train/total", loss.item(), step)
                writer.add_scalar("train/scrib", loss_scrib.item(), step)
                writer.add_scalar("train/consistency", loss_con.item(), step)
                writer.add_scalar("train/consistency_weight", consistency_weight, step)
                writer.add_scalar("train/learning_rate", lr, step)
                writer.add_scalar("train/held_out_fold", train_dataset.held_out_fold, step)
                for name, value in diagnostics.items():
                    writer.add_scalar("train/{}".format(name), value, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d loss=%.6f scrib=%.6f consistency=%.6f weight=%.4f lr=%.6g fold=%d",
                        step, args.max_iterations, loss.item(), loss_scrib.item(), loss_con.item(),
                        consistency_weight, lr, train_dataset.held_out_fold,
                    )

                should_checkpoint = (
                    checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                    or step == args.max_iterations
                )
                if should_checkpoint:
                    result = validate(model_ema, val_dataset, val_indices, args, device, num_classes)
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
                            model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score
                        )
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()
                    model_ema.train()
                    atomic_torch_save(
                        checkpoint_payload(
                            model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score
                        ),
                        output_dir / "last.pth",
                    )
                if step >= args.max_iterations:
                    break
            calibrator.commit_epoch()
            epoch += 1
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
