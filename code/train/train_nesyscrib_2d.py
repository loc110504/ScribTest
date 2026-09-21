"""UNet2D + NeSy-Scrib on ACDC/MSCMR's 2D slice-supervised protocol.

NeSy-Scrib is a Mean-Teacher method (independently-EMA-updated teacher, no
Trust-Advantage EMA or distance-calibration machinery -- those stay
VoxTrust-3D's own extensions) whose pseudo-label reliability comes from two
independent signals: the teacher's softmax-entropy confidence, and how much
a small explicit anatomical rule bank had to *edit* the teacher's hard
prediction to make it consistent (``utils.nesyscrib.symbolic_repair``).
Unlabeled pixels the student learns from are weighted by the product of the
two, so a confidently wrong-but-anatomically-implausible prediction can
still be down-weighted even though nothing about scribble-level partial
cross-entropy alone would catch it.

See ``utils/nesyscrib.py``'s module docstring for why the repair operator is
a closed-form, class-specific, scribble-anchored projection rather than a
literal per-batch argmin optimization (KL-to-teacher + a non-differentiable
logic energy) or a single generic morphological filter (which would be
unsafe for MYO specifically -- it legitimately has one hole, the LV cavity).

ACDC/MSCMR only, no WORD/3D counterpart: the rule bank (RV/MYO/LV
connectivity, MYO's ring cavity, LV-MYO/RV-MYO adjacency) is specific to
cardiac short-axis anatomy and has no analog for WORD's 7 unrelated
abdominal organs -- the same restriction, and for the same kind of reason,
as DMPLS/Bayes-WSS's existing ACDC/MSCMR-only scope.

Deployed model: like VoxTrust-3D/EFFDNet, the checkpoint saves both nets
(``model_state_dict`` = EMA teacher, ``student_state_dict`` = student);
model selection and ``test_pce_2d.py``'s default evaluation target are both
the student (pass ``--eval_target teacher`` to evaluate the EMA teacher
instead).
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

from dataloader.scribblebench_2d import RandomGenerator2D, ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import (  # noqa: E402
    atomic_torch_save,
    checkpoint_due,
    guard_fresh_output_dir,
    partial_cross_entropy,
    seed_everything,
    seed_worker,
)
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402
from utils.ema_optim import WeightEMA  # noqa: E402
from utils.nesyscrib import (  # noqa: E402
    LV,
    MYO,
    RV,
    adjacency_violation,
    enclosure_violation,
    repair_reliability_map,
    symbolic_repair,
    teacher_confidence,
    weighted_pixel_ce_loss,
)
from utils.ramps import sigmoid_rampup  # noqa: E402

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net from ScribbleBench scribbles with NeSy-Scrib"
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

    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--rampup_frac", type=float, default=0.2)
    parser.add_argument("--pseudo_loss_weight", type=float, default=1.0, help="lambda_max")
    parser.add_argument("--noise_std", type=float, default=0.1, help="student input Gaussian noise std")
    parser.add_argument(
        "--repair_sigma", type=float, default=5.0,
        help="pixels; Gaussian decay radius of repair-distance distrust around an edited pixel",
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
    if any(size % 16 for size in args.patch_size):
        raise ValueError("patch_size must be divisible by 16")
    if args.early_interval < 1 or args.late_interval < 1 or args.num_workers < 0:
        raise ValueError("early_interval/late_interval must be positive and num_workers non-negative")
    if args.late_phase_start < 0:
        raise ValueError("late_phase_start must be non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("ema_decay must satisfy 0 < alpha < 1")
    if not 0.0 <= args.warmup_frac < 1.0:
        raise ValueError("warmup_frac must satisfy 0 <= warmup_frac < 1")
    if args.rampup_frac <= 0.0:
        raise ValueError("rampup_frac must be positive")
    if args.pseudo_loss_weight < 0:
        raise ValueError("pseudo_loss_weight must be non-negative")
    if args.noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if args.repair_sigma <= 0:
        raise ValueError("repair_sigma must be positive")
    return args


def nesyscrib_step(model, model_ema, batch, device, ignore_index, args):
    """One NeSy-Scrib training iteration: teacher hypothesis -> symbolic
    repair -> repair-aware reliability -> weighted student supervision.
    Returns ``(loss_scrib, loss_pseudo, diagnostics)``.
    """
    image = batch["image"].to(device, non_blocking=True)
    target = batch["label"].to(device, non_blocking=True).long()

    with torch.no_grad():
        teacher_logits = model_ema(image)
        teacher_prob = F.softmax(teacher_logits, dim=1)
        teacher_hard = teacher_prob.argmax(dim=1)

    scribble_np = target.detach().cpu().numpy()
    teacher_hard_np = teacher_hard.detach().cpu().numpy()

    repaired_np = np.empty_like(teacher_hard_np)
    reliability_np = np.empty(teacher_hard_np.shape, dtype=np.float32)
    edit_ratios, diagnostic_violations = [], []
    for sample_index in range(teacher_hard_np.shape[0]):
        pred_hard = teacher_hard_np[sample_index]
        scribble_label = scribble_np[sample_index]
        repaired, changed = symbolic_repair(pred_hard, scribble_label, ignore_index)

        myo_bool, lv_bool, rv_bool = repaired == MYO, repaired == LV, repaired == RV
        diag_violation = 1.0 - (1.0 - enclosure_violation(myo_bool, lv_bool)) * (
            1.0 - adjacency_violation(rv_bool, myo_bool)
        )
        reliability_np[sample_index] = repair_reliability_map(changed, diag_violation, args.repair_sigma)
        repaired_np[sample_index] = repaired

        foreground = pred_hard != 0
        edit_ratios.append(float(changed[foreground].mean()) if foreground.any() else 0.0)
        diagnostic_violations.append(diag_violation)

    repaired_target = torch.from_numpy(repaired_np).to(device).long()
    r_sym = torch.from_numpy(reliability_np).to(device)
    r_conf = teacher_confidence(teacher_prob)
    reliability = (r_conf * r_sym).clamp_min(1e-6)
    pseudo_weight_map = reliability * (target == ignore_index).float()

    noise = torch.randn_like(image) * args.noise_std
    if args.noise_std > 0:
        noise = noise.clamp(-2.0 * args.noise_std, 2.0 * args.noise_std)
    student_logits = model(image + noise)

    loss_scrib, labeled_pixels = partial_cross_entropy(student_logits, target, ignore_index)
    loss_pseudo = weighted_pixel_ce_loss(student_logits, repaired_target, pseudo_weight_map)

    diagnostics = {
        "labeled_pixels": labeled_pixels.item(),
        "mean_R": reliability.mean().item(),
        "mean_R_conf": r_conf.mean().item(),
        "mean_R_sym": r_sym.mean().item(),
        "edit_ratio": float(np.mean(edit_ratios)) if edit_ratios else 0.0,
        "mean_diagnostic_violation": float(np.mean(diagnostic_violations)) if diagnostic_violations else 0.0,
    }
    return loss_scrib, loss_pseudo, diagnostics


def create_model(num_classes, feature_channels, device):
    return UNet2D(in_chns=1, class_num=num_classes, feature_chns=feature_channels).to(device)


def checkpoint_payload(model, model_ema, optimizer, scaler, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "nesyscrib",
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
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, model_ema, optimizer, scaler, args, split):
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
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_NeSyScrib" / args.dataset
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
        drop_last=False,
    )
    if len(loader) == 0:
        raise RuntimeError("training loader is empty")

    num_classes = train_dataset.num_classes
    ignore_index = train_dataset.ignore_index

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

    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(args.resume, model, model_ema, optimizer, scaler, args, split)
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
                pseudo_active = step >= warmup_iters
                lr = args.learning_rate * (1.0 - step / args.max_iterations) ** 0.9
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                amp_context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16) if args.amp else nullcontext()
                )
                with amp_context:
                    loss_scrib, loss_pl, diagnostics = nesyscrib_step(
                        model, model_ema, batch, device, ignore_index, args
                    )
                    pseudo_weight = (
                        args.pseudo_loss_weight * sigmoid_rampup(step - warmup_iters, rampup_iters)
                        if pseudo_active
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
                        "iteration=%d/%d loss=%.6f scrib=%.6f pseudo=%.6f pseudo_w=%.4f lr=%.6g edit_ratio=%.4f",
                        step, args.max_iterations, loss.item(), loss_scrib.item(), loss_pl.item(), pseudo_weight, lr,
                        diagnostics["edit_ratio"],
                    )

                should_checkpoint = (
                    checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                    or step == args.max_iterations
                )
                if should_checkpoint:
                    # Selection is by the STUDENT's validation Dice, matching
                    # this repo's Mean-Teacher convention (VoxTrust-3D/EFFDNet);
                    # see checkpoint_payload() and this module's docstring.
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
                        payload = checkpoint_payload(
                            model, model_ema, optimizer, scaler, args, split, step, best_score
                        )
                        atomic_torch_save(payload, output_dir / "best.pth")
                        logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                    else:
                        logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                    model.train()
                    model_ema.train()
                    atomic_torch_save(
                        checkpoint_payload(model, model_ema, optimizer, scaler, args, split, step, best_score),
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
