"""UNet2D + ModelMix (Zhang & Patel, MICCAI 2024), jointly training ACDC and
MSCMR.

ModelMix always trains a *pair* of tasks together: two independently
initialized ``UNet2D`` models sharing the same architecture, periodically
cross-mixing one random encoder convolutional layer between them and
regularizing the mixed (virtual) model to agree with each task's own
individual model. ACDC and MSCMR are the only two ScribbleBench datasets
sharing a compatible (4-class cardiac) label space, matching the paper's own
primary experiment -- see ``code/utils/modelmix.py`` for the three
mechanisms (image-level mixup, model-level mixup, vicinal regularization),
verified against the official ``BWGZK/ModelMix`` repository. WORD has no
comparable "intrinsically related" partner dataset in this benchmark, so
ModelMix is not run on it.

Unlike every other method in this repository, this script has no
``--dataset`` flag: it always produces two checkpoints in one run,
``<output_dir>/ACDC/{best,last}.pth`` and ``<output_dir>/MSCMR/{best,last}.pth``,
each a plain ``UNet2D`` checkpoint evaluable directly with
``code/test/test_pce_2d.py`` (same as pCE/CycleMix/SDT-Net/VoxTrust-3D).

Only sparse labels in each task's own ``labelsTr`` contribute to
optimization. Dense training labels are accessed exclusively for model
selection on that task's own patient-level holdout, exactly as in
``train_pce_2d.py``.
"""

import argparse
import json
import logging
import math
import random
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

from dataloader.scribblebench_2d import RandomGenerator2D, ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import atomic_torch_save, checkpoint_due, partial_cross_entropy, seed_everything, seed_worker  # noqa: E402
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402
from utils.modelmix import (  # noqa: E402
    encoder_conv_layer_names,
    mix_invariance_loss,
    mix_one_encoder_layer,
    one_hot_scribble,
    random_rotate_image_and_label,
    rotate_back,
    sample_mix_ratio,
    soft_partial_cross_entropy,
    vicinal_regularization_loss,
)
from utils.sdtnet import soft_dice_loss  # noqa: E402

TASKS = ("ACDC", "MSCMR")
DEFAULTS = {"patch_size": (256, 256), "batch_size": 24}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Jointly train two UNet2D models for ACDC+MSCMR with ModelMix"
    )
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None, help="parent dir; ACDC/ and MSCMR/ subdirs are created")
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=None, help="shared by both tasks (encoders must match shape)")
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
    parser.add_argument("--resume_acdc", default=None)
    parser.add_argument("--resume_mscmr", default=None)
    return parser.parse_args()


def validate_args(args):
    args.patch_size = tuple(args.patch_size or DEFAULTS["patch_size"])
    args.batch_size = args.batch_size or DEFAULTS["batch_size"]
    args.feature_channels = tuple(args.feature_channels)
    if args.batch_size < 2:
        raise ValueError("ModelMix's image-level mixup pairs each sample with its batch-reversed counterpart; batch_size must be >= 2")
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
    return args


def modelmix_task_step(model_self, model_other, image, label, ignore_index, num_classes):
    """One task's full ModelMix contribution for a training iteration: own
    supervision + image-level mixup (sup + consistency) + model-level mixup
    (sup + vicinal-reg). See ``utils/modelmix.py``'s module docstring for the
    three mechanisms. Returns ``(total_loss, components)``; the caller sums
    this task's total with its partner task's total and backpropagates once,
    jointly updating both models (matching the source's single joint
    optimizer step over both backbones+heads).
    """
    logits = model_self(image)
    probs = F.softmax(logits, dim=1)
    loss_ce, labeled_pixels = partial_cross_entropy(logits, label, ignore_index)
    loss_dice = soft_dice_loss(probs, label, num_classes, ignore_index)
    loss_own = loss_ce + loss_dice

    # ---- Image-level mixup (Eq. 1) ----
    mix_ratio_img = sample_mix_ratio(image.shape[0], image.device)
    image_mixed = mix_ratio_img * image + (1.0 - mix_ratio_img) * torch.flip(image, dims=[0])
    label_onehot = one_hot_scribble(label, num_classes, ignore_index)
    label_onehot_mixed = mix_ratio_img * label_onehot + (1.0 - mix_ratio_img) * torch.flip(label_onehot, dims=[0])
    logits_mixed_image = model_self(image_mixed)
    probs_mixed_image = F.softmax(logits_mixed_image, dim=1)
    loss_image_mix_sup = soft_partial_cross_entropy(logits_mixed_image, label_onehot_mixed)
    loss_image_mix_consistency = mix_invariance_loss(probs_mixed_image, probs, mix_ratio_img)

    # ---- Model-level mixup: the ModelMix operator (Eq. 5-6) ----
    layer_name = random.choice(encoder_conv_layer_names(model_self.encoder))
    mix_ratio_model = float(np.random.beta(0.5, 0.5))
    mixed_encoder = mix_one_encoder_layer(model_self.encoder, model_other.encoder, layer_name, mix_ratio_model)

    rotated_image, rotated_label, angle = random_rotate_image_and_label(image, label)
    mixed_logits = model_self.decoder(mixed_encoder(rotated_image))
    mixed_probs = F.softmax(mixed_logits, dim=1)
    loss_model_mix_ce, _ = partial_cross_entropy(mixed_logits, rotated_label, ignore_index)
    loss_model_mix_dice = soft_dice_loss(mixed_probs, rotated_label, num_classes, ignore_index)
    loss_model_mix_sup = loss_model_mix_ce + loss_model_mix_dice

    mixed_probs_derotated = rotate_back(mixed_probs, angle)
    loss_vicinal_reg = vicinal_regularization_loss(mixed_probs_derotated, probs)

    total = loss_own + loss_image_mix_sup + loss_image_mix_consistency + loss_model_mix_sup + loss_vicinal_reg
    components = {
        "own": loss_own.item(),
        "image_mix_sup": loss_image_mix_sup.item(),
        "image_mix_consistency": loss_image_mix_consistency.item(),
        "model_mix_sup": loss_model_mix_sup.item(),
        "vicinal_reg": loss_vicinal_reg.item(),
        "labeled_pixels": labeled_pixels.item(),
    }
    return total, components


class _RestartingLoader:
    """Wraps a DataLoader iterator that transparently restarts (a fresh
    shuffled pass) whenever it runs out, so two tasks of different dataset
    size can be stepped in lockstep every iteration.
    """

    def __init__(self, loader):
        self.loader = loader
        self._iterator = iter(loader)

    def next_batch(self):
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            return next(self._iterator)


def checkpoint_payload(model, args, dataset_name, split, step, best_score):
    """Model weights + bookkeeping only -- deliberately no optimizer/scaler
    state. The two tasks share one joint optimizer (it holds both models'
    parameters at once), so splitting its state across two per-task
    checkpoint files would not round-trip cleanly; ``--resume_acdc``/
    ``--resume_mscmr`` restore model weights and the step/best-score
    bookkeeping, not exact optimizer momentum state.
    """
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "modelmix",
        "model_config": {
            "in_chns": 1,
            "class_num": len(DATASET_CONFIGS[dataset_name]["class_names"]),
            "feature_chns": list(args.feature_channels),
        },
        "data_config": {
            "dataset": dataset_name,
            "root_path": str(args.root_path) if args.root_path else None,
            "patch_size_hw": list(args.patch_size),
            "ignore_index": DATASET_CONFIGS[dataset_name]["ignore_index"],
        },
        "global_step": step,
        "best_val_mean_dice": best_score,
        "model_state_dict": model.state_dict(),
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, args, dataset_name, split):
    checkpoint = torch.load(path, map_location="cpu")
    expected_model = checkpoint.get("model_config", {})
    expected_data = checkpoint.get("data_config", {})
    if expected_model.get("feature_chns") != list(args.feature_channels):
        raise ValueError("resume checkpoint feature_channels do not match")
    if expected_data.get("dataset") != dataset_name:
        raise ValueError("resume checkpoint dataset does not match")
    if checkpoint.get("split") != split:
        raise ValueError("resume checkpoint train/val split does not match")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return int(checkpoint["global_step"]), float(checkpoint["best_val_mean_dice"])


def configure_logging(output_dir):
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


class TaskState:
    """Everything one task (ACDC or MSCMR) needs, mirroring the single-task
    scripts' setup but with its own checkpoint namespace."""

    def __init__(self, dataset_name, args, device):
        self.dataset_name = dataset_name
        train_transform = RandomGenerator2D(args.patch_size)
        self.train_dataset = ScribbleBench2DDataset(
            dataset_name, base_dir=args.root_path, split="train", sup_type="scribble", transform=train_transform
        )
        self.val_dataset = build_val_dataset(argparse.Namespace(dataset=dataset_name, root_path=args.root_path))

        train_case_indices, val_case_indices, train_groups, val_groups, protocol = resolve_case_split(
            self.train_dataset.cases, dataset_name
        )
        self.val_case_indices = val_case_indices
        self.split = {
            "protocol": protocol,
            "grouped_by_patient": True,
            "train_groups": train_groups,
            "val_groups": val_groups,
            "train_cases": [self.train_dataset.cases[index] for index in train_case_indices],
            "val_cases": [self.train_dataset.cases[index] for index in val_case_indices],
        }

        train_slice_positions = self.train_dataset.slice_positions_for_volumes(train_case_indices)
        generator = torch.Generator().manual_seed(args.seed)
        loader = DataLoader(
            Subset(self.train_dataset, train_slice_positions),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=True,  # ModelMix pairs samples with their batch-reversed counterpart.
        )
        if len(loader) == 0:
            raise RuntimeError(
                "{} training loader is empty (need at least 2 * batch_size training slices)".format(dataset_name)
            )
        self.loader = _RestartingLoader(loader)
        self.num_train_slices = len(train_slice_positions)

        self.num_classes = self.train_dataset.num_classes
        self.ignore_index = self.train_dataset.ignore_index
        self.model = UNet2D(in_chns=1, class_num=self.num_classes, feature_chns=args.feature_channels).to(device)

        self.output_dir = Path(
            args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_ModelMix"
        ).resolve() / dataset_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "split.json").open("w", encoding="utf-8") as handle:
            json.dump(self.split, handle, indent=2)

        self.best_score = -math.inf
        self.metrics_path = self.output_dir / "validation.jsonl"


def train(args):
    args = validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.amp and device.type != "cuda":
        logging.warning("AMP requested on %s; disabling AMP", device)
        args.amp = False

    tasks = {name: TaskState(name, args, device) for name in TASKS}
    configure_logging(tasks["ACDC"].output_dir.parent)

    optimizer = torch.optim.SGD(
        [parameter for task in tasks.values() for parameter in task.model.parameters()],
        lr=args.learning_rate, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    step, best_scores = 0, {name: -math.inf for name in TASKS}
    resume_paths = {"ACDC": args.resume_acdc, "MSCMR": args.resume_mscmr}
    for name, task in tasks.items():
        if resume_paths[name]:
            resumed_step, resumed_best = restore_checkpoint(resume_paths[name], task.model, args, name, task.split)
            step = max(step, resumed_step)
            best_scores[name] = resumed_best
            task.best_score = resumed_best
            logging.info("Resumed %s from %s at iteration %d", name, resume_paths[name], resumed_step)
    if step >= args.max_iterations:
        raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    logging.info(
        "ACDC: train_slices=%d val_cases=%d | MSCMR: train_slices=%d val_cases=%d | patch=%s device=%s",
        tasks["ACDC"].num_train_slices, len(tasks["ACDC"].val_case_indices),
        tasks["MSCMR"].num_train_slices, len(tasks["MSCMR"].val_case_indices),
        args.patch_size, device,
    )
    writer = SummaryWriter(str(tasks["ACDC"].output_dir.parent / "tensorboard"))
    last_eval_step = -1

    try:
        while step < args.max_iterations:
            for task in tasks.values():
                task.model.train()

            batch_acdc = tasks["ACDC"].loader.next_batch()
            batch_mscmr = tasks["MSCMR"].loader.next_batch()
            image_acdc = batch_acdc["image"].to(device, non_blocking=True)
            label_acdc = batch_acdc["label"].to(device, non_blocking=True).long()
            image_mscmr = batch_mscmr["image"].to(device, non_blocking=True)
            label_mscmr = batch_mscmr["label"].to(device, non_blocking=True).long()

            lr = args.learning_rate * (1.0 - step / args.max_iterations) ** 0.9
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            amp_context = torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
            with amp_context:
                loss_acdc, components_acdc = modelmix_task_step(
                    tasks["ACDC"].model, tasks["MSCMR"].model, image_acdc, label_acdc,
                    tasks["ACDC"].ignore_index, tasks["ACDC"].num_classes,
                )
                loss_mscmr, components_mscmr = modelmix_task_step(
                    tasks["MSCMR"].model, tasks["ACDC"].model, image_mscmr, label_mscmr,
                    tasks["MSCMR"].ignore_index, tasks["MSCMR"].num_classes,
                )
                total_loss = loss_acdc + loss_mscmr
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step += 1

            writer.add_scalar("train/total", total_loss.item(), step)
            for name, value in components_acdc.items():
                writer.add_scalar("train/ACDC/{}".format(name), value, step)
            for name, value in components_mscmr.items():
                writer.add_scalar("train/MSCMR/{}".format(name), value, step)
            writer.add_scalar("train/learning_rate", lr, step)
            if step % 20 == 0:
                logging.info(
                    "iteration=%d/%d total=%.6f acdc_own=%.6f mscmr_own=%.6f lr=%.6g",
                    step, args.max_iterations, total_loss.item(), components_acdc["own"], components_mscmr["own"], lr,
                )

            should_checkpoint = (
                checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                or step == args.max_iterations
            )
            if should_checkpoint:
                last_eval_step = step
                for name, task in tasks.items():
                    result = validate_2d(task.model, task.val_dataset, task.val_case_indices, args, device, task.num_classes)
                    score = result["mean_dice"]
                    if not math.isfinite(score):
                        raise RuntimeError("{} validation mean Dice is not finite".format(name))
                    writer.add_scalar("val/{}/mean_dice".format(name), score, step)
                    for class_id, class_score in result["per_class_dice"].items():
                        if math.isfinite(class_score):
                            writer.add_scalar("val/{}/dice_class_{}".format(name, class_id), class_score, step)
                    record = {
                        "global_step": step,
                        "mean_dice": result["mean_dice"],
                        "num_cases": result["num_cases"],
                        "per_class_dice": {
                            key: value if math.isfinite(value) else None
                            for key, value in result["per_class_dice"].items()
                        },
                    }
                    with task.metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, allow_nan=False) + "\n")
                    if score > best_scores[name]:
                        best_scores[name] = score
                        payload = checkpoint_payload(task.model, args, name, task.split, step, score)
                        atomic_torch_save(payload, task.output_dir / "best.pth")
                        logging.info("Saved %s best.pth: iteration=%d mean_dice=%.6f", name, step, score)
                    else:
                        logging.info("Validation %s: iteration=%d mean_dice=%.6f", name, step, score)
                    atomic_torch_save(
                        checkpoint_payload(task.model, args, name, task.split, step, best_scores[name]),
                        task.output_dir / "last.pth",
                    )
                    task.model.train()
    finally:
        writer.close()

    if last_eval_step != step:
        raise RuntimeError("final iteration was not validated; checkpoint invariant broken")
    for task in tasks.values():
        if not (task.output_dir / "best.pth").is_file() or not (task.output_dir / "last.pth").is_file():
            raise RuntimeError("expected best.pth and last.pth were not created for {}".format(task.dataset_name))
    logging.info(
        "Training complete. best_val_mean_dice: ACDC=%.6f MSCMR=%.6f", best_scores["ACDC"], best_scores["MSCMR"]
    )
    return {name: task.output_dir for name, task in tasks.items()}


if __name__ == "__main__":
    train(parse_args())
