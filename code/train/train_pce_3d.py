"""UNet3D + partial cross-entropy baseline with published legacy splits.

Only sparse labels in ``labelsTr`` contribute to optimization. Dense training
labels are accessed exclusively for model selection on a patient-level holdout.
The official ``imagesTs``/``labelsTs`` split is never opened by this script.
"""

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_3d import (  # noqa: E402
    DATASET_CONFIGS,
    RandomGenerator3D,
    ScribbleBench3DDataset,
)
from networks.unet_3d import UNet3D  # noqa: E402
from utils.sliding_window_3d import sliding_window_predict  # noqa: E402
from train.legacy_splits import published_groups  # noqa: E402


DEFAULTS = {
    "ACDC": {"patch_size": (16, 128, 128), "batch_size": 2},
    "MSCMR": {"patch_size": (16, 128, 128), "batch_size": 2},
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D U-Net from ScribbleBench scribbles with pCE"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument(
        "--root_path",
        default=None,
        help="ScribbleBench root or one dataset directory (default: dataset/ScribbleBench)",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=3, type=int, default=None, metavar=("D", "H", "W"))
    parser.add_argument("--feature_channels", nargs="+", type=int, default=(16, 32, 64, 128, 256))
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--val_overlap", type=float, default=0.5)
    parser.add_argument("--sw_batch_size", type=int, default=1)
    parser.add_argument("--max_accumulator_mb", type=int, default=1024)
    parser.add_argument("--temp_dir", default=None, help="directory for validation memmaps")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--foreground_crop_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None, help="resume from a last.pth checkpoint")
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
    # UNet3D downsamples D by 8 and H/W by 16 with its default strides.
    divisors = (8, 16, 16)
    if any(size % divisor for size, divisor in zip(args.patch_size, divisors)):
        raise ValueError("patch_size must be divisible by (8, 16, 16) in DHW order")
    bottleneck_shape = [
        size // divisor for size, divisor in zip(args.patch_size, divisors)
    ]
    if np.prod(bottleneck_shape) <= 1:
        raise ValueError("patch_size produces a one-voxel bottleneck, which InstanceNorm cannot use")
    if args.eval_every < 1 or args.save_every < 1 or args.num_workers < 0:
        raise ValueError("eval_every/save_every must be positive and num_workers non-negative")
    if not 0 <= args.val_overlap < 1:
        raise ValueError("val_overlap must satisfy 0 <= val_overlap < 1")
    if not 0 <= args.foreground_crop_prob <= 1:
        raise ValueError("foreground_crop_prob must satisfy 0 <= p <= 1")
    if args.sw_batch_size < 1 or args.max_accumulator_mb < 0:
        raise ValueError("invalid sliding-window settings")
    if args.temp_dir is not None and not Path(args.temp_dir).is_dir():
        raise ValueError("temp_dir does not exist: {}".format(args.temp_dir))
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def patient_id(dataset_name, case_name):
    """Group ACDC phases and any modality suffixes from the same subject."""
    if dataset_name == "ACDC":
        match = re.match(r"^(patient\d+)", case_name)
        if not match:
            raise ValueError("Unexpected ACDC case name: {}".format(case_name))
        return match.group(1)
    if dataset_name == "MSCMR":
        return case_name.removesuffix("_DE")
    return case_name


def make_published_split(samples, dataset_name):
    """Map an exact published subject list to samples and reject mismatches."""
    groups = {}
    for index, sample in enumerate(samples):
        groups.setdefault(patient_id(dataset_name, sample["case"]), []).append(index)
    train_groups, val_groups, protocol = published_groups(dataset_name)
    expected_groups = set(train_groups) | set(val_groups)
    if len(expected_groups) != len(train_groups) + len(val_groups):
        raise RuntimeError("Published train/validation groups overlap for {}".format(dataset_name))
    observed_groups = set(groups)
    if observed_groups != expected_groups:
        missing = sorted(expected_groups - observed_groups)
        unexpected = sorted(observed_groups - expected_groups)
        raise RuntimeError(
            "Dataset {} does not match the published split; missing={}, unexpected={}".format(
                dataset_name, missing, unexpected
            )
        )
    train_indices = [index for group in train_groups for index in groups[group]]
    val_indices = [index for group in val_groups for index in groups[group]]
    return train_indices, val_indices, train_groups, val_groups, protocol


def partial_cross_entropy(logits, target, ignore_index):
    """Mean CE over annotated voxels only; unlabeled voxels have no gradient.

    With uniform random cropping (``foreground_crop_prob=0``, matching the
    official CycleMix/DMSPS training recipes), a patch can legitimately
    contain zero annotated voxels; it then contributes zero loss/gradient
    rather than aborting the run.
    """
    if logits.ndim != 5 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(
            "Expected logits [B,C,D,H,W] and target [B,D,H,W], got {} and {}".format(
                tuple(logits.shape), tuple(target.shape)
            )
        )
    valid = target != ignore_index
    invalid = valid & ((target < 0) | (target >= logits.shape[1]))
    if torch.any(invalid):
        raise ValueError("scribble contains a class outside [0, num_classes-1]")
    valid_count = valid.sum()
    if valid_count.item() == 0:
        return logits.sum() * 0.0, valid_count
    loss_sum = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction="sum")
    return loss_sum / valid_count, valid_count


def finite_mean(values):
    values = [value for value in values if math.isfinite(value)]
    return float(np.mean(values)) if values else math.nan


@torch.inference_mode()
def validate(model, dataset, indices, args, device, num_classes):
    """Select checkpoints using dense masks from the training holdout only."""
    model.eval()
    case_scores = []
    class_scores = {class_id: [] for class_id in range(1, num_classes)}
    for index in tqdm(indices, desc="validation", leave=False):
        sample = dataset[index]
        image = torch.from_numpy(sample["image"]).unsqueeze(0).unsqueeze(0).float()
        prediction = sliding_window_predict(
            model=model,
            image=image,
            num_classes=num_classes,
            patch_size=args.patch_size,
            device=device,
            overlap=args.val_overlap,
            sw_batch_size=args.sw_batch_size,
            use_amp=args.amp,
            max_accumulator_mb=args.max_accumulator_mb,
            temp_dir=args.temp_dir,
        )
        target = sample["gt_label"]
        foreground_scores = []
        for class_id in range(1, num_classes):
            pred_mask = prediction == class_id
            target_mask = target == class_id
            if not pred_mask.any() and not target_mask.any():
                score = math.nan
            elif not pred_mask.any() or not target_mask.any():
                score = 0.0
            else:
                intersection = np.count_nonzero(pred_mask & target_mask)
                score = 2.0 * intersection / (np.count_nonzero(pred_mask) + np.count_nonzero(target_mask))
            class_scores[class_id].append(score)
            if math.isfinite(score):
                foreground_scores.append(score)
        case_scores.append(finite_mean(foreground_scores))
    return {
        "mean_dice": finite_mean(case_scores),
        "per_class_dice": {
            str(class_id): finite_mean(scores) for class_id, scores in class_scores.items()
        },
        "num_cases": len(indices),
    }


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_3d",
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[args.dataset]["class_names"]),
            "feature_chns": list(args.feature_channels),
        },
        "data_config": {
            "dataset": args.dataset,
            "root_path": str(args.root_path) if args.root_path else None,
            "patch_size_dhw": list(args.patch_size),
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
        args.output_dir
        or REPO_ROOT / "checkpoints" / "ScribbleBench_pCE" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    train_transform = RandomGenerator3D(
        args.patch_size, foreground_prob=args.foreground_crop_prob
    )
    train_dataset = ScribbleBench3DDataset(
        args.dataset,
        base_dir=args.root_path,
        split="train",
        sup_type="scribble",
        transform=train_transform,
    )
    # No transform: validation runs on the complete normalized volume.
    val_dataset = ScribbleBench3DDataset(
        args.dataset,
        base_dir=args.root_path,
        split="train",
        sup_type="scribble",
        return_full_label=True,
    )
    train_indices, val_indices, train_groups, val_groups, protocol = make_published_split(
        train_dataset.samples, args.dataset
    )
    split = {
        "protocol": protocol,
        "grouped_by_patient": True,
        "train_groups": train_groups,
        "val_groups": val_groups,
        "train_cases": [train_dataset.samples[index]["case"] for index in train_indices],
        "val_cases": [train_dataset.samples[index]["case"] for index in val_indices],
    }
    with (output_dir / "split.json").open("w", encoding="utf-8") as handle:
        json.dump(split, handle, indent=2)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        Subset(train_dataset, train_indices),
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

    num_classes = train_dataset.num_classes
    ignore_index = train_dataset.ignore_index
    model = UNet3D(
        in_chns=1,
        class_num=num_classes,
        feature_chns=args.feature_channels,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(
            args.resume, model, optimizer, scaler, args, split
        )
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError(
                "resume checkpoint already reached max_iterations; increase --max_iterations"
            )

    logging.info(
        "dataset=%s classes=%d ignore=%d train=%d val=%d patch=%s device=%s",
        args.dataset,
        num_classes,
        ignore_index,
        len(train_indices),
        len(val_indices),
        args.patch_size,
        device,
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
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if args.amp
                    else nullcontext()
                )
                with amp_context:
                    logits = model(image)
                    loss, labeled_voxels = partial_cross_entropy(
                        logits, target, ignore_index
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                step += 1

                writer.add_scalar("train/pce", loss.item(), step)
                writer.add_scalar("train/labeled_voxels", labeled_voxels.item(), step)
                writer.add_scalar("train/learning_rate", lr, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d pCE=%.6f labeled_voxels=%d lr=%.6g",
                        step,
                        args.max_iterations,
                        loss.item(),
                        labeled_voxels.item(),
                        lr,
                    )

                should_evaluate = step % args.eval_every == 0 or step == args.max_iterations
                if should_evaluate:
                    result = validate(
                        model, val_dataset, val_indices, args, device, num_classes
                    )
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
                            model, optimizer, scaler, args, split, step, best_score
                        )
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()

                if step % args.save_every == 0 or step == args.max_iterations:
                    atomic_torch_save(
                        checkpoint_payload(
                            model, optimizer, scaler, args, split, step, best_score
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
