"""UNetCCT3D + DMSPS (Han et al., Medical Image Analysis 2024) on ScribbleBench.

Reimplements DMSPS's two-stage recipe on top of this repository's dual-decoder
``UNetCCT3D`` (shared 3D encoder, one clean decoder, one decoder fed
``dropout3d``-perturbed features -- exactly the DB-Net architecture DMSPS
uses) and ``ScribbleBench3DDataset``. See ``code/utils/dmsps.py`` for the loss
and label-expansion implementation, verified against the official
``HiLab-git/DMSPS`` source.

Stage 1 trains DB-Net directly on the sparse scribble with pCE + a dynamically
mixed soft pseudo-label consistency term. Stage 2 re-initializes DB-Net from
the stage-1 checkpoint and retrains it with the same loss, but the scribble is
first expanded with high-confidence, largest-connected-component predictions
from stage 1 (``--stage 2 --init_checkpoint <stage1 best.pth>``).

Only sparse (stage 1) or expanded (stage 2) labels contribute to
optimization. Dense training labels are accessed exclusively for model
selection on a patient-level holdout, exactly as in ``train_pce_3d.py``.
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
from torch.utils.data import DataLoader, Dataset, Subset
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
from networks.unet_cct_3d import UNetCCT3D  # noqa: E402
from train.common_3d import (  # noqa: E402
    atomic_torch_save,
    checkpoint_due,
    make_published_split,
    partial_cross_entropy,
    seed_everything,
    seed_worker,
    validate,
)
from utils.dmsps import (  # noqa: E402
    dual_branch_volume_probs,
    dynamic_mixed_pseudo_label,
    expand_labels,
    soft_pseudo_supervision_loss,
)

DEFAULTS = {
    "ACDC": {"patch_size": (16, 128, 128), "batch_size": 2, "tau": 0.1},
    "MSCMR": {"patch_size": (16, 128, 128), "batch_size": 2, "tau": 0.1},
    "WORD": {"patch_size": (64, 96, 96), "batch_size": 1, "tau": 0.3},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a dual-decoder 3D U-Net from ScribbleBench scribbles with DMSPS"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_CONFIGS))
    parser.add_argument("--stage", type=int, required=True, choices=(1, 2))
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        help="stage-1 best.pth to re-initialize DB-Net weights from (required for --stage 2)",
    )
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=3, type=int, default=None, metavar=("D", "H", "W"))
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

    # DMSPS-specific hyperparameters (paper defaults).
    parser.add_argument("--lambda_sps", type=float, default=8.0)
    parser.add_argument("--dropout_p", type=float, default=0.5, help="auxiliary-decoder feature dropout rate")
    parser.add_argument(
        "--tau", type=float, default=None,
        help="stage-2 normalized-entropy confidence threshold (default: dataset-specific)",
    )
    return parser.parse_args()


def validate_args(args):
    defaults = DEFAULTS[args.dataset]
    args.patch_size = tuple(args.patch_size or defaults["patch_size"])
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.tau = args.tau if args.tau is not None else defaults["tau"]
    args.feature_channels = tuple(args.feature_channels)
    if args.stage == 2 and not args.init_checkpoint:
        raise ValueError("--stage 2 requires --init_checkpoint pointing at a stage-1 best.pth")
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
    if not 0 < args.dropout_p < 1:
        raise ValueError("dropout_p must satisfy 0 < p < 1")
    if not 0 < args.tau <= 1:
        raise ValueError("tau must satisfy 0 < tau <= 1")
    return args


def dmsps_step(model, image, target, ignore_index, args):
    """One DMSPS forward pass; returns the total loss and its components."""
    main_logits, aux_logits = model(image, return_auxiliary=True)
    loss_pce_main, labeled_voxels = partial_cross_entropy(main_logits, target, ignore_index)
    loss_pce_aux, _ = partial_cross_entropy(aux_logits, target, ignore_index)
    loss_pce = 0.5 * (loss_pce_main + loss_pce_aux)

    probs_main = F.softmax(main_logits, dim=1)
    probs_aux = F.softmax(aux_logits, dim=1)
    alpha = float(np.random.uniform(0.0, 1.0))
    pseudo_target = dynamic_mixed_pseudo_label(probs_main, probs_aux, alpha)
    loss_sps = soft_pseudo_supervision_loss(probs_main, probs_aux, pseudo_target)

    total = loss_pce + args.lambda_sps * loss_sps
    components = {
        "pce": loss_pce.item(),
        "sps": loss_sps.item(),
        "alpha": alpha,
        "labeled_voxels": labeled_voxels.item(),
    }
    return total, components


class ExpandedLabelDataset(Dataset):
    """Swaps in a stage-2 expanded label for every case that has one."""

    def __init__(self, base_dataset, expanded_labels, transform):
        self.base_dataset = base_dataset
        self.expanded_labels = expanded_labels
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        sample = self.base_dataset[index]
        expanded = self.expanded_labels.get(sample["case"])
        if expanded is not None:
            sample = dict(sample)
            sample["label"] = expanded
        return self.transform(sample)


def build_stage2_labels(model, raw_dataset, train_indices, ignore_index, args, device):
    """Run stage-1 DB-Net over every training case and expand its scribble."""
    expanded_labels = {}
    model.eval()
    for index in tqdm(train_indices, desc="stage2 label expansion", leave=False):
        sample = raw_dataset[index]
        image = torch.from_numpy(sample["image"]).unsqueeze(0).unsqueeze(0).float()
        mean_probs = dual_branch_volume_probs(
            model=model,
            image=image,
            num_classes=raw_dataset.num_classes,
            patch_size=args.patch_size,
            device=device,
            overlap=args.val_overlap,
            sw_batch_size=args.sw_batch_size,
            use_amp=args.amp,
            max_accumulator_mb=args.max_accumulator_mb,
            temp_dir=args.temp_dir,
        )
        expanded_labels[sample["case"]] = expand_labels(
            sample["label"], mean_probs, ignore_index, args.tau
        )
    return expanded_labels


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_cct_3d",
        "training_method": "dmsps_stage{}".format(args.stage),
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[args.dataset]["class_names"]),
            "feature_chns": list(args.feature_channels),
            "perturbations": ["dropout"],
            "perturbation_dropout": args.dropout_p,
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
        or REPO_ROOT / "checkpoints" / "ScribbleBench_DMSPS" / args.dataset / "stage{}".format(args.stage)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    train_transform = RandomGenerator3D(args.patch_size, foreground_prob=args.foreground_crop_prob)
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
    model = UNetCCT3D(
        in_chns=1,
        class_num=num_classes,
        feature_chns=args.feature_channels,
        perturbations=("dropout",),
        perturbation_dropout=args.dropout_p,
    ).to(device)

    expanded_labels = {}
    if args.stage == 2:
        init_checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)
        logging.info("Initialized DB-Net from %s", args.init_checkpoint)
        expanded_labels = build_stage2_labels(model, raw_train_dataset, train_indices, ignore_index, args, device)
        num_expanded = sum(
            int(np.count_nonzero(label != ignore_index)) for label in expanded_labels.values()
        )
        logging.info("Stage-2 label expansion: %d cases, %d annotated voxels total", len(expanded_labels), num_expanded)

    train_dataset = ExpandedLabelDataset(raw_train_dataset, expanded_labels, train_transform)
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
        "dataset=%s stage=%d classes=%d ignore=%d train=%d val=%d patch=%s device=%s",
        args.dataset, args.stage, num_classes, ignore_index, len(train_indices), len(val_indices), args.patch_size, device,
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
                    loss, components = dmsps_step(model, image, target, ignore_index, args)
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
                        "iteration=%d/%d loss=%.6f pce=%.6f sps=%.6f alpha=%.3f lr=%.6g",
                        step, args.max_iterations, loss.item(), components["pce"], components["sps"],
                        components["alpha"], lr,
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
