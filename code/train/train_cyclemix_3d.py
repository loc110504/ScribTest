"""VNet3D + CycleMix (Zhang & Zhuang, CVPR 2022) on WORD's full-3D protocol.

Reimplements CycleMix's four-loss framework -- unmix pCE, mix pCE, global
mix-invariance consistency and local connectivity consistency -- on top of
this repository's ``VNet3D`` and ``ScribbleBench3DDataset``. See
``code/utils/cyclemix.py`` for the exact 3D adaptation notes (in particular,
the official Puzzle Mix solver is replaced with a 3D cuboid CutMix, which the
paper lists as an admissible mix operator). ACDC/MSCMR train as independent
2D slices instead; see ``train_cyclemix_2d.py``.

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
from utils.cyclemix import (  # noqa: E402
    largest_component_targets,
    mix_images,
    mix_labels,
    negative_cosine_similarity,
    occlude,
    sample_batch_cuboid_masks,
)

SUPPORTED_DATASETS = ("WORD",)
DEFAULTS = {
    # CycleMix mixes each sample with another one drawn from the same batch,
    # so batch_size must be >= 2; the pCE baseline's WORD default of 1 cannot
    # be reused here.
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 2},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 3D VNet from ScribbleBench scribbles with CycleMix"
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

    # CycleMix loss weights (paper Eq. 14; the official code hard-codes 0.1
    # for both consistency terms instead of the paper's reported 0.05/1.0 --
    # we default to the paper's reported values).
    parser.add_argument("--lambda_unmix", type=float, default=1.0)
    parser.add_argument("--lambda_mix", type=float, default=1.0)
    parser.add_argument("--lambda_con_global", type=float, default=0.05)
    parser.add_argument("--lambda_con_local", type=float, default=1.0)
    # Fraction of each patch axis covered by the mix box / occlusion box.
    parser.add_argument("--mix_frac_low", type=float, default=0.3)
    parser.add_argument("--mix_frac_high", type=float, default=0.7)
    parser.add_argument("--occlusion_frac_low", type=float, default=0.1)
    parser.add_argument("--occlusion_frac_high", type=float, default=0.3)
    return parser.parse_args()


def validate_args(args):
    defaults = DEFAULTS[args.dataset]
    args.patch_size = tuple(args.patch_size or defaults["patch_size"])
    args.batch_size = args.batch_size or defaults["batch_size"]
    if args.batch_size < 2:
        raise ValueError("CycleMix pairs each sample with another one in the batch; batch_size must be >= 2")
    if args.max_iterations < 1:
        raise ValueError("max_iterations must be positive")
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
    for name in ("mix_frac_low", "mix_frac_high", "occlusion_frac_low", "occlusion_frac_high"):
        value = getattr(args, name)
        if not 0 < value <= 1:
            raise ValueError("{} must satisfy 0 < value <= 1".format(name))
    if args.mix_frac_low > args.mix_frac_high:
        raise ValueError("mix_frac_low must be <= mix_frac_high")
    if args.occlusion_frac_low > args.occlusion_frac_high:
        raise ValueError("occlusion_frac_low must be <= occlusion_frac_high")
    return args


def cyclemix_step(model, image, target, ignore_index, args, device):
    """One CycleMix forward pass; returns the total loss and its components."""
    batch_size = image.shape[0]
    perm = (torch.arange(batch_size, device=device) + 1) % batch_size

    logits = model(image)
    loss_unmix, labeled_voxels = partial_cross_entropy(logits, target, ignore_index)
    probs = F.softmax(logits, dim=1)
    probs_partner = probs[perm]

    mix_frac = (args.mix_frac_low, args.mix_frac_high)
    occ_frac = (args.occlusion_frac_low, args.occlusion_frac_high)

    # Direction A: M(x, x[perm]) then occlude.
    mask_mix_a = sample_batch_cuboid_masks(batch_size, args.patch_size, mix_frac, device)
    mask_occ_a = sample_batch_cuboid_masks(batch_size, args.patch_size, occ_frac, device)
    image_mix_a = mix_images(image, image[perm], mask_mix_a)
    label_mix_a = mix_labels(target, target[perm], mask_mix_a)
    image_occ_a, label_occ_a = occlude(image_mix_a, label_mix_a, mask_occ_a, ignore_index)

    # Direction B: M(x[perm], x) with an independently sampled pair of boxes,
    # since the paper's mix operator is not symmetric.
    mask_mix_b = sample_batch_cuboid_masks(batch_size, args.patch_size, mix_frac, device)
    mask_occ_b = sample_batch_cuboid_masks(batch_size, args.patch_size, occ_frac, device)
    image_mix_b = mix_images(image[perm], image, mask_mix_b)
    label_mix_b = mix_labels(target[perm], target, mask_mix_b)
    image_occ_b, label_occ_b = occlude(image_mix_b, label_mix_b, mask_occ_b, ignore_index)

    logits_occ_a = model(image_occ_a)
    logits_occ_b = model(image_occ_b)
    loss_mix_a, _ = partial_cross_entropy(logits_occ_a, label_occ_a, ignore_index)
    loss_mix_b, _ = partial_cross_entropy(logits_occ_b, label_occ_b, ignore_index)
    loss_mix = 0.5 * (loss_mix_a + loss_mix_b)

    # Global consistency: mixing the two original predictions and zeroing the
    # occluded region should match segmenting the occluded-mixed image directly.
    keep_a = (~mask_occ_a).to(probs.dtype)
    keep_b = (~mask_occ_b).to(probs.dtype)
    target_probs_a = mix_images(probs, probs_partner, mask_mix_a) * keep_a
    target_probs_b = mix_images(probs_partner, probs, mask_mix_b) * keep_b
    pred_probs_a = F.softmax(logits_occ_a, dim=1) * keep_a
    pred_probs_b = F.softmax(logits_occ_b, dim=1) * keep_b
    loss_con_global = 0.5 * (
        negative_cosine_similarity(target_probs_a, pred_probs_a)
        + negative_cosine_similarity(target_probs_b, pred_probs_b)
    )

    # Local consistency: predictions should collapse onto a single connected
    # component per foreground class. Every batch element already plays both
    # the "sample 1" and "sample 2" role once (via `perm`), so a single mean
    # over the batch already realizes the paper's symmetric 0.5*(term+term).
    cleaned_targets = largest_component_targets(probs)
    loss_con_local = negative_cosine_similarity(probs, cleaned_targets)

    total = (
        args.lambda_unmix * loss_unmix
        + args.lambda_mix * loss_mix
        + args.lambda_con_global * loss_con_global
        + args.lambda_con_local * loss_con_local
    )
    components = {
        "unmix": loss_unmix.item(),
        "mix": loss_mix.item(),
        "con_global": loss_con_global.item(),
        "con_local": loss_con_local.item(),
        "labeled_voxels": labeled_voxels.item(),
    }
    return total, components


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "vnet_3d",
        "training_method": "cyclemix",
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
    if expected_model.get("n_filters") != args.n_filters:
        raise ValueError("resume checkpoint n_filters does not match")
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
        drop_last=True,  # CycleMix pairs samples in-batch; a size-1 final batch cannot pair.
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty (need at least 2 * batch_size training samples)")

    num_classes = train_dataset.num_classes
    ignore_index = train_dataset.ignore_index
    model = VNet3D(in_chns=1, class_num=num_classes, n_filters=args.n_filters).to(device)
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
        "dataset=%s classes=%d ignore=%d train=%d val=%d patch=%s device=%s",
        args.dataset, num_classes, ignore_index, len(train_indices), len(val_indices), args.patch_size, device,
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
