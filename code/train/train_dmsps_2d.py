"""UNetCCT2D + DMSPS (Han et al., Medical Image Analysis 2024) on ACDC/
MSCMR's 2D slice-supervised protocol.

Reuses ``dmsps_step`` from ``train_dmsps_3d.py`` unchanged (dual-branch pCE +
dynamically mixed soft pseudo-label consistency is elementwise/softmax-only
math with no 3D-specific assumption, see ``code/utils/dmsps.py``), and
``UNetCCT2D`` (``networks/unet_2d.py``) is the same shared-encoder /
main-decoder / dropout-perturbed-auxiliary-decoder DB-Net architecture as
``UNetCCT3D``, one spatial dimension down.

Stage 1 trains DB-Net directly on the sparse scribble with pCE + a
dynamically mixed soft pseudo-label consistency term. Stage 2 re-initializes
DB-Net from the stage-1 checkpoint and retrains it with the same loss, but
the scribble is first expanded with high-confidence, largest-connected-
component predictions from stage 1 (``--stage 2 --init_checkpoint <stage1
best.pth>``) -- per *case* (its full slice stack, 26-connectivity across
slices, exactly like the 3D pipeline), using
``utils.dmsps.dual_branch_slice_stack_probs`` for the per-slice forward pass
instead of 3D sliding-window inference.

Only sparse (stage 1) or expanded (stage 2) labels contribute to
optimization. Dense training labels are accessed exclusively for model
selection on a patient-level holdout, exactly as in ``train_pce_2d.py``.
WORD stays a full-3D VNet pipeline, see ``train_dmsps_3d.py``.
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import RandomGenerator2D, ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNetCCT2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import atomic_torch_save, checkpoint_due, guard_fresh_output_dir, seed_everything, seed_worker  # noqa: E402
from train.train_dmsps_3d import dmsps_step  # noqa: E402
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402
from utils.dmsps import dual_branch_slice_stack_probs, expand_labels  # noqa: E402

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24, "tau": 0.1},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24, "tau": 0.1},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D dual-decoder U-Net from ScribbleBench scribbles with DMSPS"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
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
    if any(size % 16 for size in args.patch_size):
        raise ValueError("patch_size must be divisible by 16")
    if args.early_interval < 1 or args.late_interval < 1 or args.num_workers < 0:
        raise ValueError("early_interval/late_interval must be positive and num_workers non-negative")
    if args.late_phase_start < 0:
        raise ValueError("late_phase_start must be non-negative")
    if not 0 < args.dropout_p < 1:
        raise ValueError("dropout_p must satisfy 0 < p < 1")
    if not 0 < args.tau <= 1:
        raise ValueError("tau must satisfy 0 < tau <= 1")
    return args


class ExpandedLabel2DDataset(Dataset):
    """Swaps in a stage-2 expanded label for every slice whose case has one.

    2D counterpart of ``train_dmsps_3d.py``'s ``ExpandedLabelDataset``, built
    directly on ``ScribbleBench2DDataset``'s public per-volume caches instead
    of re-deriving per-slice access.
    """

    def __init__(self, base_dataset, expanded_labels, transform):
        self.base_dataset = base_dataset
        self.expanded_labels = expanded_labels
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        volume_index, slice_index = self.base_dataset.slice_index[index]
        case = self.base_dataset.cases[volume_index]
        expanded = self.expanded_labels.get(case)
        label = expanded[slice_index] if expanded is not None else self.base_dataset.labels[volume_index][slice_index]
        sample = {
            "image": self.base_dataset.images[volume_index][slice_index],
            "label": label,
            "idx": index,
            "case": case,
            "slice_index": slice_index,
            "spacing": self.base_dataset.spacings[volume_index],
            "num_classes": self.base_dataset.num_classes,
            "ignore_index": self.base_dataset.ignore_index,
            "is_scribble": self.base_dataset.is_scribble[volume_index],
        }
        return self.transform(sample)


def build_stage2_labels(model, raw_dataset, train_case_indices, ignore_index, args, device):
    """Run stage-1 DB-Net over every training case's full slice stack and
    expand its scribble, per case (26-connectivity across the case's slices,
    exactly matching the 3D pipeline's granularity)."""
    expanded_labels = {}
    model.eval()
    for volume_index in tqdm(train_case_indices, desc="stage2 label expansion", leave=False):
        mean_probs = dual_branch_slice_stack_probs(
            model=model,
            image=raw_dataset.images[volume_index],
            num_classes=raw_dataset.num_classes,
            patch_size=args.patch_size,
            device=device,
            use_amp=args.amp,
        )
        case = raw_dataset.cases[volume_index]
        expanded_labels[case] = expand_labels(raw_dataset.labels[volume_index], mean_probs, ignore_index, args.tau)
    return expanded_labels


def checkpoint_payload(model, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_cct_2d",
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
        args.output_dir
        or REPO_ROOT / "checkpoints" / "ScribbleBench_DMSPS" / args.dataset / "stage{}".format(args.stage)
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_fresh_output_dir(output_dir, args.resume)
    configure_logging(output_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    train_transform = RandomGenerator2D(args.patch_size)
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
    model = UNetCCT2D(
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
        expanded_labels = build_stage2_labels(model, raw_train_dataset, train_case_indices, ignore_index, args, device)
        num_expanded = sum(
            int(np.count_nonzero(label != ignore_index)) for label in expanded_labels.values()
        )
        logging.info("Stage-2 label expansion: %d cases, %d annotated slice-pixels total", len(expanded_labels), num_expanded)

    train_dataset = ExpandedLabel2DDataset(raw_train_dataset, expanded_labels, train_transform)
    train_slice_positions = raw_train_dataset.slice_positions_for_volumes(train_case_indices)
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
        "dataset=%s stage=%d classes=%d ignore=%d train_slices=%d val_cases=%d patch=%s device=%s",
        args.dataset, args.stage, num_classes, ignore_index, len(train_slice_positions), len(val_case_indices),
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
