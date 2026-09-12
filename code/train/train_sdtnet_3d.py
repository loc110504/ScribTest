"""UNet3D + SDT-Net (Nguyen et al. 2026) on ScribbleBench.

Reimplements SDT-Net's dual-teacher/single-student framework -- Dynamic
Teacher Switching (DTS), Pick Reliable Pixels (PRP) pseudo-labeling and
Hierarchical Consistency (HiCo) feature alignment -- on top of this
repository's ``UNet3D`` and ``ScribbleBench3DDataset``. Ported from this
repository's 2D reference (``train_sdtnet_2d.py``), treated as the ground
truth for details the paper leaves ambiguous; see ``code/utils/sdtnet.py``
for the two spots where that reference script's *dependencies* (not the
method itself) turned out to have real bugs that are fixed here instead of
reproduced.

Only sparse labels in ``labelsTr`` contribute to optimization. Dense training
labels are accessed exclusively for model selection on a patient-level
holdout, exactly as in ``train_pce_3d.py``.
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
from torch.utils.data import DataLoader, Subset

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
from train.common_3d import (  # noqa: E402
    atomic_torch_save,
    make_published_split,
    partial_cross_entropy,
    seed_everything,
    seed_worker,
    validate,
)
from utils.sdtnet import TeacherEMA, feature_consistency_loss, pick_reliable_pixels, soft_dice_loss  # noqa: E402

DEFAULTS = {
    "ACDC": {"patch_size": (16, 128, 128), "batch_size": 2},
    "MSCMR": {"patch_size": (16, 128, 128), "batch_size": 2},
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D U-Net from ScribbleBench scribbles with SDT-Net"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--root_path", default=None)
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
    parser.add_argument("--temp_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--foreground_crop_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", default=None)

    # SDT-Net-specific hyperparameters (matching train_sdtnet_2d.py exactly).
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="PRP threshold tau")
    parser.add_argument("--ema_alpha", type=float, default=0.99, help="teacher EMA decay rate")
    parser.add_argument(
        "--ema_student_weight_decay", type=float, default=0.02 * 0.01,
        help="extra multiplicative decay applied to the student on every EMA step",
    )
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
    divisors = (8, 16, 16)
    if any(size % divisor for size, divisor in zip(args.patch_size, divisors)):
        raise ValueError("patch_size must be divisible by (8, 16, 16) in DHW order")
    bottleneck_shape = [size // divisor for size, divisor in zip(args.patch_size, divisors)]
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
    if not 0 < args.confidence_threshold < 1:
        raise ValueError("confidence_threshold must satisfy 0 < tau < 1")
    if not 0 < args.ema_alpha < 1:
        raise ValueError("ema_alpha must satisfy 0 < alpha < 1")
    return args


def sdtnet_step(student, teacher1, teacher2, image, target, ignore_index, num_classes, args):
    """One SDT-Net forward pass; returns the total loss, the selected
    teacher id (1 or 2, for the caller to EMA-update), and loss components.
    """
    with torch.no_grad():
        logits_t1, feats_t1 = teacher1(image, return_features=True)
        logits_t2, feats_t2 = teacher2(image, return_features=True)
        loss_t1, _ = partial_cross_entropy(logits_t1, target, ignore_index)
        loss_t2, _ = partial_cross_entropy(logits_t2, target, ignore_index)

        # Dynamic Teacher Switching (Eq. 3-4): pick the teacher with the
        # lower scribble pCE loss for this batch as the reliable teacher.
        if loss_t1.item() < loss_t2.item():
            selected = 1
            probs_teacher = F.softmax(logits_t1, dim=1)
            high_teacher, low_teacher = feats_t1["decoder"][0], feats_t1["decoder"][-1]
        else:
            selected = 2
            probs_teacher = F.softmax(logits_t2, dim=1)
            high_teacher, low_teacher = feats_t2["decoder"][0], feats_t2["decoder"][-1]
        pseudo_label = pick_reliable_pixels(probs_teacher, args.confidence_threshold, ignore_index)

    logits_s, feats_s = student(image, return_features=True)
    probs_s = F.softmax(logits_s, dim=1)
    high_student, low_student = feats_s["decoder"][0], feats_s["decoder"][-1]

    loss_scribble, labeled_voxels = partial_cross_entropy(logits_s, target, ignore_index)

    loss_pseudo_ce, pseudo_voxels = partial_cross_entropy(logits_s, pseudo_label, ignore_index)
    loss_pseudo_dice = soft_dice_loss(probs_s, pseudo_label, num_classes, ignore_index)
    loss_pseudo = loss_pseudo_ce + loss_pseudo_dice

    loss_high = feature_consistency_loss(high_student, high_teacher)
    loss_low = feature_consistency_loss(low_student, low_teacher)

    # Eq. 9: L_Total = L_Scribble + L_Pseudo + L_HiCo, where L_Pseudo already
    # carries Eq. 6's internal 0.5 and L_HiCo carries Eq. 8's internal 0.5
    # (both folded into the *0.5 factors here to match train_sdtnet_2d.py's
    # `loss_pseudo * 0.5` / `(loss_low + loss_high) * 0.5` exactly).
    total = loss_scribble + 0.5 * loss_pseudo + 0.5 * (loss_low + loss_high)
    components = {
        "scribble": loss_scribble.item(),
        "pseudo": loss_pseudo.item(),
        "hico_low": loss_low.item(),
        "hico_high": loss_high.item(),
        "labeled_voxels": labeled_voxels.item(),
        "pseudo_voxels": pseudo_voxels.item(),
        "selected_teacher": selected,
    }
    return total, selected, components


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_3d",
        "training_method": "sdtnet",
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
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_SDTNet" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    train_transform = RandomGenerator3D(args.patch_size, foreground_prob=args.foreground_crop_prob)
    train_dataset = ScribbleBench3DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", transform=train_transform
    )
    val_dataset = ScribbleBench3DDataset(
        args.dataset, base_dir=args.root_path, split="train", sup_type="scribble", return_full_label=True
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

    student = UNet3D(in_chns=1, class_num=num_classes, feature_chns=args.feature_channels).to(device)
    teacher1 = UNet3D(in_chns=1, class_num=num_classes, feature_chns=args.feature_channels).to(device)
    teacher2 = UNet3D(in_chns=1, class_num=num_classes, feature_chns=args.feature_channels).to(device)
    for teacher in (teacher1, teacher2):
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teacher.eval()
    ema1 = TeacherEMA(student, teacher1, alpha=args.ema_alpha, student_weight_decay=args.ema_student_weight_decay)
    ema2 = TeacherEMA(student, teacher2, alpha=args.ema_alpha, student_weight_decay=args.ema_student_weight_decay)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(args.resume, student, optimizer, scaler, args, split)
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    logging.info(
        "dataset=%s classes=%d ignore=%d train=%d val=%d patch=%s device=%s",
        args.dataset, num_classes, ignore_index, len(train_indices), len(val_indices), args.patch_size, device,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    metrics_path = output_dir / "validation.jsonl"
    last_eval_step = -1

    try:
        while step < args.max_iterations:
            student.train()
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
                    loss, selected, components = sdtnet_step(
                        student, teacher1, teacher2, image, target, ignore_index, num_classes, args
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                (ema1 if selected == 1 else ema2).step()
                step += 1

                writer.add_scalar("train/total", loss.item(), step)
                for name, value in components.items():
                    writer.add_scalar("train/{}".format(name), value, step)
                writer.add_scalar("train/learning_rate", lr, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d loss=%.6f scribble=%.6f pseudo=%.6f hico_low=%.6f hico_high=%.6f "
                        "teacher=%d lr=%.6g",
                        step, args.max_iterations, loss.item(), components["scribble"], components["pseudo"],
                        components["hico_low"], components["hico_high"], selected, lr,
                    )

                should_evaluate = step % args.eval_every == 0 or step == args.max_iterations
                if should_evaluate:
                    result = validate(student, val_dataset, val_indices, args, device, num_classes)
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
                        payload = checkpoint_payload(student, optimizer, scaler, args, split, step, best_score)
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    student.train()

                if step % args.save_every == 0 or step == args.max_iterations:
                    atomic_torch_save(
                        checkpoint_payload(student, optimizer, scaler, args, split, step, best_score),
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
