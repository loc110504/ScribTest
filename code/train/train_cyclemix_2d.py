"""UNet2D + CycleMix (Zhang & Zhuang, CVPR 2022) on ACDC/MSCMR's 2D
slice-supervised protocol.

Reuses ``cyclemix_step`` from ``train_cyclemix_3d.py`` unchanged: every loss
term it composes (``utils/cyclemix.py``'s mix/occlude/cuboid-mask/largest-
component helpers) is shape-agnostic over the spatial rank, so the exact same
four-loss framework -- unmix pCE, mix pCE, global mix-invariance consistency,
local connectivity consistency -- applies to 2D patches with no changes.

Only sparse labels in ``labelsTr`` contribute to optimization. Dense training
labels are accessed exclusively for model selection on a patient-level
holdout, exactly as in ``train_pce_2d.py``. WORD stays a full-3D VNet
pipeline, see ``train_cyclemix_3d.py``.
"""

import argparse
import json
import logging
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import RandomGenerator2D, ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import (  # noqa: E402
    atomic_torch_save,
    checkpoint_due,
    guard_fresh_output_dir,
    seed_everything,
    seed_worker,
)
from train.train_cyclemix_3d import cyclemix_step  # noqa: E402
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net from ScribbleBench scribbles with CycleMix"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--feature_channels", nargs="+", type=int, default=(16, 32, 64, 128, 256))
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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

    # CycleMix loss weights (paper Eq. 14; matches train_cyclemix_3d.py).
    parser.add_argument("--lambda_unmix", type=float, default=1.0)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--lambda_con_global", type=float, default=0.05)
    parser.add_argument("--lambda_con_local", type=float, default=1.0)
    parser.add_argument("--mix_frac_low", type=float, default=0.3)
    parser.add_argument("--mix_frac_high", type=float, default=0.7)
    parser.add_argument("--occlusion_frac_low", type=float, default=0.1)
    parser.add_argument("--occlusion_frac_high", type=float, default=0.3)
    return parser.parse_args()


def validate_args(args):
    defaults = DEFAULTS[args.dataset]
    args.patch_size = tuple(args.patch_size or defaults["patch_size"])
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.feature_channels = tuple(args.feature_channels)
    if args.batch_size < 2:
        raise ValueError("CycleMix pairs each sample with another one in the batch; batch_size must be >= 2")
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
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
    for name in ("mix_frac_low", "mix_frac_high", "occlusion_frac_low", "occlusion_frac_high"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError("{} must satisfy 0 < value <= 1".format(name))
    if args.mix_frac_low > args.mix_frac_high:
        raise ValueError("mix_frac_low must be <= mix_frac_high")
    if args.occlusion_frac_low > args.occlusion_frac_high:
        raise ValueError("occlusion_frac_low must be <= occlusion_frac_high")
    return args


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "cyclemix",
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
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, optimizer, scaler, args, split):
    checkpoint = torch.load(path, map_location="cpu")
    expected_model = checkpoint.get("model_config", {})
    expected_data = checkpoint.get("data_config", {})
    if expected_model.get("feature_chns") != list(args.feature_channels):
        raise ValueError("resume checkpoint feature_channels do not match")
    if expected_data.get("dataset") != args.dataset:
        raise ValueError("resume checkpoint dataset does not match")
    if checkpoint.get("split") != split:
        raise ValueError("resume checkpoint train/val split does not match")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
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
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_CycleMix" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_fresh_output_dir(output_dir, args.resume)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    train_transform = RandomGenerator2D(args.patch_size)
    train_dataset = ScribbleBench2DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=train_transform
    )
    val_dataset = build_val_dataset(args)

    train_case_indices, val_case_indices, train_groups, val_groups, protocol = resolve_case_split(
        train_dataset.cases, args.dataset
    )
    split = {
        "protocol": protocol,
        "grouped_by_patient": True,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "train_cases": [train_dataset.cases[index] for index in train_case_indices],
        "val_cases": [train_dataset.cases[index] for index in val_case_indices],
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    train_slice_positions = train_dataset.slice_positions_for_volumes(train_case_indices)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Subset(train_dataset, train_slice_positions),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,  # CycleMix pairs samples in-batch; a size-1 final batch cannot pair.
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty (need at least 2 * batch_size training slices)")

    num_classes = train_dataset.num_classes
    ignore_index = train_dataset.ignore_index
    model = UNet2D(in_chns=1, class_num=num_classes, feature_chns=args.feature_channels).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(args.resume, model, optimizer, scaler, args, split)
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    logging.info(
        "dataset=%s classes=%d ignore=%d train_slices=%d val_cases=%d patch=%s device=%s",
        args.dataset, num_classes, ignore_index, len(train_slice_positions), len(val_case_indices),
        args.patch_size, device,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    metrics_path = output_dir / "validation.jsonl"
    last_eval_step = -1

    try:
        while step < args.max_iterations:
            model.train()
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                target = batch["label"].to(device, non_blocking=True).long()
                lr = args.learning_rate * (1.0 - step / args.max_iterations) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
                )
                with amp_context:
                    loss, components = cyclemix_step(model, image, target, ignore_index, args, device)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1

                writer.add_scalar("train/total", loss.item(), step)
                for name, value in components.items():
                    writer.add_scalar("train/{}".format(name), value, step)
                writer.add_scalar("train/learning_rate", lr, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d loss=%.6f unmix=%.6f mix=%.6f con_g=%.6f con_l=%.6f lr=%.6g",
                        step, args.max_iterations, loss.item(), components["unmix"], components["mix"],
                        components["con_global"], components["con_local"], lr,
                    )

                should_checkpoint = (
                    checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                    or step == args.max_iterations
                )
                if should_checkpoint:
                    result = validate_2d(model, val_dataset, val_case_indices, args, device, num_classes)
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
                        payload = checkpoint_payload(model, optimizer, scaler, args, split, step, best_score)
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()
                    atomic_torch_save(
                        checkpoint_payload(model, optimizer, scaler, args, split, step, best_score),
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
