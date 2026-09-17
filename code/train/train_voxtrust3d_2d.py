"""UNet2D + VoxTrust-3D (VoxTrust3D_CVPR2026_Proposed_Method.pdf) on ACDC/
MSCMR's 2D slice-supervised protocol.

Reuses ``voxtrust_step`` from ``train_voxtrust3d_3d.py`` unchanged (every
calculation it performs -- partial CE, reliability score, transfer-distance
lookup, calibration update, masked pseudo-target loss -- is shape-agnostic
over the spatial rank, see ``code/utils/voxtrust3d.py``); only the student's
strong-view augmentation differs, so this script passes
``augment_fn=strong_intensity_augment_2d``.

ACDC/MSCMR train as independent 2D slices, and Sec. 4.1's block-partition
unit ("all scribble voxels of one class on one annotated slice") is already
single-slice, so the 2D pipeline's Omega_sup/Omega_cal calibration partition
is computed per *slice* rather than per case, and the physical transfer
distance (Sec. 4.3) is an in-plane 2D distance (2-entry spacing) instead of a
3D one. Unlike the 3D pipeline, there is no sub-volume cropping: ACDC/MSCMR
slices are already small, so every training sample is the whole native slice
resized to ``patch_size`` (see ``utils.voxtrust3d.random_flip_rotate_resize_2d``,
the 2D counterpart of ``choose_patch_origin``/``build_patch_coordinates``/
``gather_patch``/``random_flip_rotate``, none of which are needed here).

Only sparse labels in ``labelsTr`` (split further into Omega_sup/Omega_cal)
contribute to optimization. Dense training labels are accessed exclusively
for model selection on a patient-level holdout, exactly as in
``train_pce_2d.py``. Per Sec. 5 ("only one EMA network is required at
inference; the calibrator is removed"), the deployed/checkpointed model is
the EMA teacher, saved in the same schema as the sibling 2D baselines -- so
``code/test/test_pce_2d.py`` evaluates ``best.pth`` directly. WORD stays a
full-3D VNet pipeline, see ``train_voxtrust3d_3d.py``.
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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import atomic_torch_save, checkpoint_due, seed_everything, seed_worker  # noqa: E402
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402
from train.train_voxtrust3d_3d import voxtrust_step  # noqa: E402
from utils.ema_optim import WeightEMA  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402
from utils.voxtrust3d import (  # noqa: E402
    RollingCalibrationBuffer,
    build_class_trees,
    fit_distance_bins,
    random_flip_rotate_resize_2d,
    spatially_blocked_partition,
    stable_seed,
    strong_intensity_augment_2d,
)

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net from ScribbleBench scribbles with VoxTrust-3D"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--feature_channels", nargs="+", type=int, default=(16, 32, 64, 128, 256))
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--holdout_fraction", type=float, default=0.15, help="eta")
    parser.add_argument("--distance_strata", type=int, default=3, help="B")
    parser.add_argument("--target_precision", type=float, default=0.95, help="rho")
    parser.add_argument("--wilson_delta", type=float, default=0.05, help="delta")
    parser.add_argument("--calibration_min_samples", type=int, default=32, help="n_min")
    parser.add_argument("--calibration_buffer_size", type=int, default=4096, help="N_max")
    parser.add_argument("--calibration_block_cap", type=int, default=64, help="m_max")
    parser.add_argument("--score_grid_points", type=int, default=101, help="|T|, grid over [0, 1]")
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--rampup_frac", type=float, default=0.2)
    parser.add_argument("--pseudo_loss_weight", type=float, default=8.0, help="lambda_max")

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
    args.feature_channels = tuple(args.feature_channels)
    if args.max_iterations < 1 or args.batch_size < 1:
        raise ValueError("max_iterations and batch_size must be positive")
    if len(args.feature_channels) != 5 or any(value < 1 for value in args.feature_channels):
        raise ValueError("feature_channels must contain five positive integers")
    if any(value < 1 for value in args.patch_size):
        raise ValueError("patch_size values must be positive")
    if any(size % 16 for size in args.patch_size):
        raise ValueError("patch_size must be divisible by 16")
    if args.early_interval < 1 or args.late_interval < 1 or args.num_workers < 0:
        raise ValueError("early_interval/late_interval must be positive and num_workers non-negative")
    if args.late_phase_start < 0:
        raise ValueError("late_phase_start must be non-negative")
    if not 0 < args.ema_decay < 1:
        raise ValueError("ema_decay must satisfy 0 < alpha < 1")
    if not 0.0 < args.holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must satisfy 0 < eta < 1")
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
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("warmup_frac must satisfy 0 <= warmup_frac < 1")
    if args.rampup_frac <= 0.0:
        raise ValueError("rampup_frac must be positive")
    if args.pseudo_loss_weight < 0:
        raise ValueError("pseudo_loss_weight must be non-negative")
    return args


class VoxTrustSlice2DDataset(Dataset):
    """Serves 2D training slices carrying everything VoxTrust-3D needs.

    Each training slice's Omega_sup/Omega_cal scribble partition (Eq. 3) is
    computed once, here, at construction time -- one partition per *slice*,
    matching Sec. 4.1 ("we split the observed scribbles once at the
    beginning of a training run") applied at the 2D pipeline's actual
    training-sample granularity (a slice, not a case).
    """

    def __init__(self, base_dataset, slice_positions, holdout_fraction, seed, patch_size):
        self.base_dataset = base_dataset
        self.slice_positions = list(slice_positions)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.num_classes = base_dataset.num_classes
        self.ignore_index = base_dataset.ignore_index

        self.slice_ids = []
        self.spacings = {}
        self.sup_coords = {}
        self.cal_coords = {}
        self.cal_block_id = {}
        for position in tqdm(self.slice_positions, desc="VoxTrust-3D (2D) scribble partition", leave=False):
            volume_index, slice_index = self.base_dataset.slice_index[position]
            case = self.base_dataset.cases[volume_index]
            slice_id = "{}_{}".format(case, slice_index)
            self.slice_ids.append(slice_id)
            spacing = np.asarray(self.base_dataset.spacings[volume_index], dtype=np.float64)
            self.spacings[slice_id] = spacing[1:]  # drop the through-plane axis; distance stays in-plane
            label = self.base_dataset.labels[volume_index][slice_index]
            rng = np.random.default_rng(stable_seed(seed, slice_id))
            sup_coords, cal_coords, cal_block_id = spatially_blocked_partition(
                label, self.ignore_index, self.num_classes, holdout_fraction, rng
            )
            self.sup_coords[slice_id] = sup_coords
            self.cal_coords[slice_id] = cal_coords
            self.cal_block_id[slice_id] = cal_block_id

    def __len__(self):
        return len(self.slice_positions)

    def case_trees(self, slice_id):
        return build_class_trees(self.sup_coords[slice_id], self.spacings[slice_id])

    def per_case_partitions(self):
        return [
            {"sup_coords": self.sup_coords[sid], "cal_coords": self.cal_coords[sid], "spacing": self.spacings[sid]}
            for sid in self.slice_ids
        ]

    def __getitem__(self, i):
        position = self.slice_positions[i]
        volume_index, slice_index = self.base_dataset.slice_index[position]
        slice_id = self.slice_ids[i]
        image = self.base_dataset.images[volume_index][slice_index]
        raw_label = self.base_dataset.labels[volume_index][slice_index]
        height, width = raw_label.shape

        coord_h, coord_w = (index.astype(np.int64) for index in np.indices((height, width)))

        cal_mask = np.zeros((height, width), dtype=bool)
        block = np.full((height, width), -1, dtype=np.int64)
        for class_id, coords in self.cal_coords[slice_id].items():
            if len(coords) == 0:
                continue
            cal_mask[coords[:, 0], coords[:, 1]] = True
            block[coords[:, 0], coords[:, 1]] = self.cal_block_id[slice_id][class_id]

        sup_label = np.where(cal_mask | (raw_label == self.ignore_index), self.ignore_index, raw_label)
        cal_label = np.where(cal_mask, raw_label, self.ignore_index)

        arrays = {
            "image": image.astype(np.float32),
            "sup_label": sup_label.astype(np.int64),
            "cal_label": cal_label.astype(np.int64),
            "cal_block": block,
            "coord_h": coord_h,
            "coord_w": coord_w,
        }
        cval = {
            "image": 0.0,
            "sup_label": self.ignore_index,
            "cal_label": self.ignore_index,
            "cal_block": -1,
            "coord_h": -1,
            "coord_w": -1,
        }
        augmented = random_flip_rotate_resize_2d(arrays, cval, self.patch_size)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(augmented["image"], dtype=np.float32)).unsqueeze(0),
            "sup_label": torch.from_numpy(np.ascontiguousarray(augmented["sup_label"], dtype=np.int64)),
            "cal_label": torch.from_numpy(np.ascontiguousarray(augmented["cal_label"], dtype=np.int64)),
            "cal_block": torch.from_numpy(np.ascontiguousarray(augmented["cal_block"], dtype=np.int64)),
            "coord": torch.from_numpy(
                np.ascontiguousarray(np.stack([augmented["coord_h"], augmented["coord_w"]], axis=0), dtype=np.int64)
            ),
            "case": slice_id,
            "spacing": np.asarray(self.spacings[slice_id], dtype=np.float32),
        }


def create_model(num_classes, feature_channels, device):
    return UNet2D(in_chns=1, class_num=num_classes, feature_chns=feature_channels).to(device)


def checkpoint_payload(model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "voxtrust3d",
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[args.dataset]["class_names"]),
            "feature_chns": list(args.feature_channels),
        },
        "data_config": {
            "dataset": args.dataset,
            "root_path": str(args.root_path) if args.root_path else None,
            "patch_size_hw": list(args.patch_size),
            "ignore_index": DATASET_CONFIGS[args.dataset]["ignore_index"],
        },
        "global_step": step,
        "best_val_mean_dice": best_score,
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
    if expected_model.get("feature_chns") != list(args.feature_channels):
        raise ValueError("resume checkpoint feature_channels do not match")
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


def train(args):
    args = validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_VoxTrust3D" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    raw_train_dataset = ScribbleBench2DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    val_dataset = build_val_dataset(args)

    train_case_indices, val_case_indices, train_groups, val_groups, protocol = resolve_case_split(
        raw_train_dataset.cases, args.dataset
    )
    split = {
        "protocol": protocol,
        "grouped_by_patient": True,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "train_cases": [raw_train_dataset.cases[index] for index in train_case_indices],
        "val_cases": [raw_train_dataset.cases[index] for index in val_case_indices],
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    num_classes = raw_train_dataset.num_classes
    ignore_index = raw_train_dataset.ignore_index

    logging.info("Building the fixed Omega_sup/Omega_cal scribble partition (eta=%.3f)...", args.holdout_fraction)
    train_slice_positions = raw_train_dataset.slice_positions_for_volumes(train_case_indices)
    train_dataset = VoxTrustSlice2DDataset(
        raw_train_dataset,
        train_slice_positions,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        patch_size=args.patch_size,
    )
    case_trees = {slice_id: train_dataset.case_trees(slice_id) for slice_id in train_dataset.slice_ids}

    logging.info("Fitting distance-stratum bin edges and d_c^max (B=%d)...", args.distance_strata)
    edges_np, d_max_np = fit_distance_bins(train_dataset.per_case_partitions(), num_classes, args.distance_strata)
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

    model = create_model(num_classes, args.feature_channels, device)
    model_ema = create_model(num_classes, args.feature_channels, device)
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

    logging.info(
        "dataset=%s classes=%d ignore=%d train_slices=%d val_cases=%d patch=%s device=%s warmup_iters=%d rampup_iters=%d",
        args.dataset, num_classes, ignore_index, len(train_slice_positions), len(val_case_indices), args.patch_size,
        device, warmup_iters, rampup_iters,
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
                        augment_fn=strong_intensity_augment_2d,
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
                ema_optimizer.step()
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
                    result = validate_2d(model_ema, val_dataset, val_case_indices, args, device, num_classes)
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
