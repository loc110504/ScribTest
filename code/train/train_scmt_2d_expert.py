"""UNet2D + SC-MT (Scribble-Calibrated Mean Teacher) on the expert-scribble
ACDC/MSCMR archive (see ``dataloader/expert_scribble_2d.py``), *in addition
to* (not instead of) ``train_scmt_2d.py``'s ScribbleBench pipeline.

Reuses ``scmt_step`` (from ``train_scmt_3d.py``) and ``SCMTSlice2DDataset``
(from ``train_scmt_2d.py``) unchanged: ``SCMTSlice2DDataset`` only ever reads
its ``base_dataset`` through the public attributes ``ExpertScribble2DDataset``
also implements, so it needs no changes at all to serve this dataset source
-- see ``train_pce_2d_expert.py``'s module docstring for the full rationale.

The deployed/checkpointed model is the EMA teacher, so
``code/test/test_pce_2d_expert.py`` evaluates ``best.pth`` directly.
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
from torch.utils.data import DataLoader

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.expert_scribble_2d import ExpertScribble2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import atomic_torch_save, checkpoint_due, guard_fresh_output_dir, seed_everything, seed_worker  # noqa: E402
from train.train_pce_2d_expert import build_val_dataset, resolve_case_split, resolve_val_indices  # noqa: E402
from train.train_scmt_2d import SCMTSlice2DDataset  # noqa: E402
from train.train_scmt_3d import scmt_step  # noqa: E402
from utils.ema_optim import WeightEMA  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402
from utils.scmt import ReliabilityCalibrationTable, fit_distance_bin_edges, stable_seed  # noqa: E402

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net from the expert-scribble ACDC/MSCMR archive with SC-MT"
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
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers; note persistent_workers is always disabled for this script "
        "(see train_scmt_3d.train()'s docstring) so this only controls per-epoch parallelism",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)

    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--num_folds", type=int, default=5, help="K")
    parser.add_argument("--confidence_bins", type=int, default=5)
    parser.add_argument("--distance_bins", type=int, default=4, help="finite bins; a no-support bin is added on top")
    parser.add_argument("--calibration_momentum", type=float, default=0.9, help="EMA momentum for the calibration table")
    parser.add_argument("--calibration_min_samples", type=int, default=10, help="n_min before a table cell is trusted")
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--rampup_frac", type=float, default=0.2)
    parser.add_argument("--consistency_weight", type=float, default=1.0, help="lambda_max")

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


def create_model(num_classes, feature_channels, device):
    return UNet2D(in_chns=1, class_num=num_classes, feature_chns=feature_channels).to(device)


def checkpoint_payload(model, model_ema, optimizer, scaler, calibrator, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "scmt",
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[args.dataset]["class_names"]),
            "feature_chns": list(args.feature_channels),
        },
        "data_config": {
            "dataset": args.dataset,
            "data_source": "expert_scribble",
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
        args.output_dir or REPO_ROOT / "checkpoints" / "ExpertScribble_SCMT" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_fresh_output_dir(output_dir, args.resume)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    raw_train_dataset = ExpertScribble2DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=None
    )
    val_dataset = build_val_dataset(args)

    train_case_indices, train_groups, val_groups, protocol = resolve_case_split(raw_train_dataset.cases, args.dataset)
    val_case_indices = resolve_val_indices(val_dataset.cases, args.dataset)
    split = {
        "protocol": protocol + " (expert-scribble archive)",
        "grouped_by_patient": True,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "train_cases": [raw_train_dataset.cases[index] for index in train_case_indices],
        "val_cases": [val_dataset.cases[index] for index in val_case_indices],
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    num_classes = raw_train_dataset.num_classes
    ignore_index = raw_train_dataset.ignore_index

    logging.info("Assigning rotating scribble-block folds (K=%d)...", args.num_folds)
    train_slice_positions = raw_train_dataset.slice_positions_for_volumes(train_case_indices)
    train_dataset = SCMTSlice2DDataset(
        raw_train_dataset,
        train_slice_positions,
        num_folds=args.num_folds,
        seed=args.seed,
        patch_size=args.patch_size,
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
        # Deliberately not persistent -- see train_scmt_3d.train()'s docstring.
        persistent_workers=False,
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
    calibrator = ReliabilityCalibrationTable(
        num_classes=num_classes,
        num_confidence_bins=args.confidence_bins,
        num_distance_bins=args.distance_bins + 1,
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
        "dataset=%s(expert) classes=%d ignore=%d train_slices=%d val_cases=%d patch=%s device=%s warmup_iters=%d "
        "rampup_iters=%d num_folds=%d",
        args.dataset, num_classes, ignore_index, len(train_slice_positions), len(val_case_indices), args.patch_size,
        device, warmup_iters, rampup_iters, args.num_folds,
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
