"""Bayes-WSS (Zheng et al., MICCAI 2024, ``A Bayesian Approach to
Weakly-supervised Laparoscopic Image Segmentation``) on ACDC/MSCMR's 2D
slice-supervised protocol, verified against the official
``MoriLabNU/Bayesian_WSS`` source (``AutoLaparo/train.py``).

Two-stage training, exactly mirroring the official recipe:

Stage 1 -- learning ``p(x,y|z)``: a Bayesian dual-encoder/decoder CVAE
(``networks/bayes_wss_2d.BayesCVAE2D``) is trained on the scribble with the
ELBO decomposition (KL + reconstruction + partial-CE + a local-window
DenseCRF regularizer; see ``utils/bayes_wss.py`` for the loss terms and the
one documented deviation from the official compiled bilateral filter).

Stage 2 -- learning ``p(w|x,y)``: the frozen stage-1 CVAE's sample-averaged
prediction fills in every scribble-unlabeled pixel
(``utils.bayes_wss.merge_pseudo_labels``); a plain ``UNet2D`` (the officially
deployed ``BDL_MC_UNet``, architecturally identical to this repository's
``UNet2D``) is then trained with ordinary cross-entropy on that fully-dense
merged label. This ``UNet2D`` is what gets checkpointed and is the model
compared against every other method -- the CVAE is a training-only helper,
never saved. Consequently ``--resume`` only restores stage-2 (the student);
stage-1 CVAE pretraining always reruns first (deterministically, from the
same ``--seed``), which is redundant compute on a resumed run but keeps the
regenerated pseudo-labels identical to the interrupted run's. Because the checkpoint is a plain ``UNet2D`` (same
``model_name``/``model_config`` schema as ``train_pce_2d.py``), it is
evaluated directly with ``test_pce_2d.py``; there is no separate
``test_bayes_wss_2d.py``.

Following the paper's own optimizer choice (Sec. 3.1's implementation
details), both stages use Adam with a fixed learning rate (no polynomial
decay), unlike this benchmark's SGD-based methods. Only sparse scribble
labels and the CVAE's own pseudo-labels ever reach either network's
gradient; dense training labels are accessed exclusively for stage-2 model
selection on a patient-level holdout, exactly as in ``train_pce_2d.py``. Only
ACDC/MSCMR (2D) are implemented for Bayes-WSS in this benchmark.
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataloader.scribblebench_2d import RandomGenerator2D, ScribbleBench2DDataset  # noqa: E402
from dataloader.scribblebench_3d import DATASET_CONFIGS  # noqa: E402
from networks.bayes_wss_2d import BayesCVAE2D  # noqa: E402
from networks.unet_2d import UNet2D  # noqa: E402
from train.common_2d import validate_2d  # noqa: E402
from train.common_3d import atomic_torch_save, checkpoint_due, guard_fresh_output_dir, seed_everything, seed_worker  # noqa: E402
from train.train_pce_2d import build_val_dataset, resolve_case_split  # noqa: E402
from utils.bayes_wss import bayes_mean_softmax, bayes_wss_cvae_step, merge_pseudo_labels  # noqa: E402

SUPPORTED_DATASETS = ("ACDC", "MSCMR")
DEFAULTS = {
    "ACDC": {"patch_size": (256, 256), "batch_size": 24},
    "MSCMR": {"patch_size": (256, 256), "batch_size": 24},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a 2D U-Net from ScribbleBench scribbles with Bayes-WSS"
    )
    parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS)
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--cvae_iterations", type=int, default=15000,
        help="stage-1 CVAE (p(x,y|z)) pretraining budget",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=30000,
        help="stage-2 deployed UNet2D (p(w|x,y)) training budget",
    )
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patch_size", nargs=2, type=int, default=None, metavar=("H", "W"))
    parser.add_argument("--feature_channels", nargs="+", type=int, default=(16, 32, 64, 128, 256))
    parser.add_argument("--latent_dim", type=int, default=256, help="CVAE latent dimensionality")
    parser.add_argument("--base_lr", type=float, default=1e-4, help="Adam learning rate (both stages)")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sample_time", type=int, default=5, help="N: CVAE Monte Carlo sample count (paper Sec. 3.3)")
    parser.add_argument("--kl", type=float, default=1e-3, help="alpha: KL divergence weight")
    parser.add_argument("--recon", type=float, default=0.1, help="beta: image reconstruction weight")
    parser.add_argument("--crf", type=float, default=1e-8, help="gamma: local DenseCRF weight")
    parser.add_argument("--crf_sigma_rgb", type=float, default=15.0)
    parser.add_argument("--crf_sigma_xy", type=float, default=5.0)
    parser.add_argument("--crf_radius", type=int, default=5)
    parser.add_argument(
        "--early_interval", type=int, default=5000,
        help="stage-2 eval+checkpoint cadence for iterations <= --late_phase_start",
    )
    parser.add_argument(
        "--late_interval", type=int, default=1000,
        help="stage-2 eval+checkpoint cadence for iterations > --late_phase_start",
    )
    parser.add_argument(
        "--late_phase_start", type=int, default=20000,
        help="iteration at which the finer --late_interval cadence begins",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true", help="mixed precision for stage-2 validation inference")
    parser.add_argument("--resume", default=None, help="resume stage-2 from a last.pth checkpoint")
    return parser.parse_args()


def validate_args(args):
    defaults = DEFAULTS[args.dataset]
    args.patch_size = tuple(args.patch_size or defaults["patch_size"])
    args.batch_size = args.batch_size or defaults["batch_size"]
    args.feature_channels = tuple(args.feature_channels)
    if args.cvae_iterations < 0 or args.max_iterations < 1 or args.batch_size < 1:
        raise ValueError("cvae_iterations must be non-negative, max_iterations/batch_size must be positive")
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
    if args.sample_time < 1 or args.latent_dim < 1 or args.crf_radius < 1:
        raise ValueError("sample_time/latent_dim/crf_radius must be positive")
    return args


def checkpoint_payload(model, optimizer, args, split, step, best_score):
    return {
        "schema_version": 1,
        "model_name": "unet_2d",
        "training_method": "bayes_wss",
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
        "split": split,
        "args": vars(args),
    }


def restore_checkpoint(path, model, optimizer, args, split):
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
    return int(checkpoint["global_step"]), float(checkpoint["best_val_mean_dice"])


def configure_logging(output_dir):
    handlers = [logging.StreamHandler(), logging.FileHandler(output_dir / "train.log")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def train(args):
    args = validate_args(args)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    output_dir = Path(
        args.output_dir or REPO_ROOT / "checkpoints" / "ScribbleBench_BayesWSS" / args.dataset
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

    num_classes = train_dataset.num_classes
    ignore_index = train_dataset.ignore_index
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

    logging.info(
        "dataset=%s classes=%d ignore=%d train_slices=%d val_cases=%d patch=%s device=%s",
        args.dataset, num_classes, ignore_index, len(train_slice_positions), len(val_case_indices),
        args.patch_size, device,
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    batches = cycle(loader)

    # ---------------- Stage 1: learn p(x, y | z) with the CVAE. ----------------
    cvae = BayesCVAE2D(
        in_chns=1, class_num=num_classes, patch_size=args.patch_size,
        feature_chns=args.feature_channels, latent_dim=args.latent_dim,
    ).to(device)
    cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    cvae.train()
    for cvae_step in tqdm(range(1, args.cvae_iterations + 1), desc="stage1 CVAE", disable=args.cvae_iterations == 0):
        batch = next(batches)
        image = batch["image"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True).long()
        cvae_optimizer.zero_grad(set_to_none=True)
        loss, components = bayes_wss_cvae_step(cvae, image, target, ignore_index, args)
        loss.backward()
        cvae_optimizer.step()
        writer.add_scalar("stage1/total", loss.item(), cvae_step)
        for name, value in components.items():
            writer.add_scalar("stage1/{}".format(name), value, cvae_step)
        if cvae_step % 20 == 0:
            logging.info(
                "stage1 iteration=%d/%d loss=%.6f pce=%.6f crf=%.6g recon=%.6f kl=%.6f",
                cvae_step, args.cvae_iterations, loss.item(), components["pce"], components["crf"],
                components["recon"], components["kl"],
            )
    cvae.eval()
    logging.info("Stage 1 complete: CVAE frozen for pseudo-label generation")

    # ---------------- Stage 2: learn p(w | x, y) with a plain UNet2D. ----------------
    student = UNet2D(in_chns=1, class_num=num_classes, feature_chns=args.feature_channels).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    step, best_score = 0, -math.inf
    if args.resume:
        step, best_score = restore_checkpoint(args.resume, student, optimizer, args, split)
        logging.info("Resumed %s at iteration %d", args.resume, step)
        if step >= args.max_iterations:
            raise ValueError("resume checkpoint already reached max_iterations; increase --max_iterations")

    metrics_path = output_dir / "validation.jsonl"
    last_eval_step = -1
    student.train()
    try:
        while step < args.max_iterations:
            batch = next(batches)
            image = batch["image"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True).long()
            with torch.no_grad():
                _, _, _, y_logits = cvae(image, sample_time=args.sample_time)
                pseudo_argmax = bayes_mean_softmax(y_logits).argmax(dim=1)
            merged_label = merge_pseudo_labels(target, pseudo_argmax, ignore_index)

            optimizer.zero_grad(set_to_none=True)
            logits = student(image)
            loss = F.cross_entropy(logits, merged_label)
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar("stage2/ce", loss.item(), step)
            if step % 20 == 0:
                logging.info("stage2 iteration=%d/%d loss=%.6f", step, args.max_iterations, loss.item())

            should_checkpoint = (
                checkpoint_due(step, args.late_phase_start, args.early_interval, args.late_interval)
                or step == args.max_iterations
            )
            if should_checkpoint:
                result = validate_2d(student, val_dataset, val_case_indices, args, device, num_classes)
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
                    payload = checkpoint_payload(student, optimizer, args, split, step, best_score)
                    atomic_torch_save(payload, output_dir / "best.pth")
                    logging.info("Saved best.pth: iteration=%d mean_dice=%.6f", step, score)
                else:
                    logging.info("Validation: iteration=%d mean_dice=%.6f", step, score)
                student.train()
                atomic_torch_save(
                    checkpoint_payload(student, optimizer, args, split, step, best_score),
                    output_dir / "last.pth",
                )
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
