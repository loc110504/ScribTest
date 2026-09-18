"""VNet3D + EFFDNet (Liu et al., MICCAI 2025) on WORD's full-3D protocol.

Reimplements EFFDNet's Mean-Teacher framework -- partial cross-entropy on the
student, a dense cross-entropy against the EMA teacher's (noise-perturbed)
pseudo-label, a grid-based Foreground-Background Separation Loss (FBSL), and
a Foreground Augmentation with Diverse Context (FADC) copy-paste
augmentation -- on top of this repository's ``VNet3D`` and
``ScribbleBench3DDataset``; see ``code/utils/effdnet.py`` for the algorithm,
verified against the official ``Aurora-003-web/EFFDNet`` source. ACDC/MSCMR
train as independent 2D slices instead; see ``train_effdnet_2d.py``.

Only sparse labels in ``labelsTr`` contribute to optimization. Dense training
labels are accessed exclusively for model selection on a patient-level
holdout, exactly as in ``train_pce_3d.py``. The deployed/checkpointed model
is the *student* (matching the source, not an EMA teacher), so
``code/test/test_pce_3d.py`` evaluates ``best.pth`` directly.
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
from utils.effdnet import (  # noqa: E402
    foreground_augmentation_diverse_context,
    foreground_background_separation_loss,
    update_ema_variables,
)

SUPPORTED_DATASETS = ("WORD",)
DEFAULTS = {
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D VNet from ScribbleBench scribbles with EFFDNet"
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

    # EFFDNet-specific hyperparameters (paper defaults).
    parser.add_argument("--ema_alpha", type=float, default=0.99, help="teacher EMA decay rate")
    parser.add_argument("--lambda_value", type=float, default=0.6, help="pseudo-label loss weight (paper's lambda)")
    parser.add_argument("--delta", type=float, default=0.3, help="FBSL weight within the pseudo-label term")
    parser.add_argument("--num_regions", type=int, default=8, help="FBSL grid resolution per axis (paper's K)")
    parser.add_argument("--fbsl_temperature", type=float, default=0.07)
    parser.add_argument("--use_fbsl", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_fadc", type=int, default=1, choices=[0, 1])
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
    if not 0 < args.ema_alpha < 1:
        raise ValueError("ema_alpha must satisfy 0 < alpha < 1")
    if args.lambda_value < 0 or args.delta < 0:
        raise ValueError("lambda_value/delta must be non-negative")
    if args.num_regions < 1:
        raise ValueError("num_regions must be positive")
    return args


def effdnet_step(model, ema_model, image, target, ignore_index, args):
    """One EFFDNet training iteration; returns the total loss and its
    components. Shape-agnostic over the spatial rank (2D/3D), so this same
    function is reused verbatim by ``train_effdnet_2d.py``.
    """
    logits, features = model(image, return_features=True)
    feature = features["decoder"][-1]

    with torch.no_grad():
        noise = torch.clamp(torch.randn_like(image) * 0.1, -0.05, 0.05)
        ema_logits = ema_model(image + noise)
        pseudo_label = torch.argmax(torch.softmax(ema_logits, dim=1), dim=1)

    loss_scribble, labeled_voxels = partial_cross_entropy(logits, target, ignore_index)
    loss_pseudo = F.cross_entropy(logits, pseudo_label)
    components = {"scribble": loss_scribble.item(), "pseudo": loss_pseudo.item(), "labeled_voxels": labeled_voxels.item()}

    pseudo_term = loss_pseudo
    if args.use_fbsl:
        loss_fbsl = foreground_background_separation_loss(
            feature, target, ignore_index, num_regions=args.num_regions, temperature=args.fbsl_temperature
        )
        components["fbsl"] = loss_fbsl.item()
        pseudo_term = pseudo_term + args.delta * loss_fbsl

    total = loss_scribble + args.lambda_value * pseudo_term

    if args.use_fadc:
        aug_image, aug_target, aug_pseudo = foreground_augmentation_diverse_context(
            image, target, pseudo_label, ignore_index
        )
        aug_logits = model(aug_image)
        loss_scribble_aug, _ = partial_cross_entropy(aug_logits, aug_target, ignore_index)
        loss_pseudo_aug = F.cross_entropy(aug_logits, aug_pseudo)
        total = total + loss_scribble_aug + args.lambda_value * loss_pseudo_aug
        components["scribble_aug"] = loss_scribble_aug.item()
        components["pseudo_aug"] = loss_pseudo_aug.item()

    return total, components


def create_model(num_classes, n_filters, device):
    return VNet3D(in_chns=1, class_num=num_classes, n_filters=n_filters).to(device)


def checkpoint_payload(model, ema_model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "vnet_3d",
        "training_method": "effdnet",
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
        # Deployed model is the student, matching the source.
        "model_state_dict": model.state_dict(),
        "ema_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, ema_model, optimizer, scaler, args, split):
    checkpoint = torch.load(path, map_location="cpu")
    expected_model = checkpoint.get("model_config", {})
    expected_data = checkpoint.get("data_config", {})
    if expected_model.get("n_filters") != args.n_filters:
        raise ValueError("resume checkpoint n_filters does not match")
    if expected_data.get("dataset") != args.dataset:
        raise ValueError("resume checkpoint dataset does not match")
    if checkpoint.get("split") != split:
        raise ValueError("resume checkpoint train/val split does not match")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=True)
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
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_EFFDNet" / args.dataset
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_fresh_output_dir(output_dir, args.resume)
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
    # Student and teacher are independently initialized (verified against
    # source: two separate `create_model()` calls, no state-dict copy).
    model = create_model(num_classes, args.n_filters, device)
    ema_model = create_model(num_classes, args.n_filters, device)
    for parameter in ema_model.parameters():
        parameter.detach_()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(args.resume, model, ema_model, optimizer, scaler, args, split)
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
            model.train()
            ema_model.train()
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
                    loss, components = effdnet_step(model, ema_model, image, target, ignore_index, args)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                update_ema_variables(model, ema_model, args.ema_alpha, step)
                step += 1

                writer.add_scalar("train/total", loss.item(), step)
                for name, value in components.items():
                    writer.add_scalar("train/{}".format(name), value, step)
                writer.add_scalar("train/learning_rate", lr, step)
                if step % 20 == 0:
                    logging.info(
                        "iteration=%d/%d loss=%.6f scribble=%.6f pseudo=%.6f lr=%.6g",
                        step, args.max_iterations, loss.item(), components["scribble"], components["pseudo"], lr,
                    )

                should_checkpoint = (
                    checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                    or step == args.max_iterations
                )
                if should_checkpoint:
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
                        payload = checkpoint_payload(model, ema_model, optimizer, scaler, args, split, step, best_score)
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()
                    ema_model.train()
                    atomic_torch_save(
                        checkpoint_payload(model, ema_model, optimizer, scaler, args, split, step, best_score),
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
