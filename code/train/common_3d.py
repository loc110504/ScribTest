"""Infrastructure shared by the scribble-supervised 3D training scripts.

Split resolution, checkpointing, seeding and dense-label validation are
identical across ``train_pce_3d.py``, ``train_cyclemix_3d.py`` and
``train_dmsps_3d.py``.  This module centralizes that logic so the three
scripts only differ in their loss/forward pass.
"""

import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train.legacy_splits import published_groups
from utils.sliding_window_3d import sliding_window_predict


def partial_cross_entropy(logits, target, ignore_index):
    """Mean CE over annotated voxels only; unlabeled voxels have no gradient.

    With uniform random cropping (``foreground_crop_prob=0``, matching the
    official CycleMix/DMSPS training recipes), a patch can legitimately
    contain zero annotated voxels. That patch contributes zero loss/gradient
    rather than aborting the run -- ``logits.sum() * 0.0`` keeps the value in
    the autograd graph so ``.backward()`` stays valid for callers (e.g.
    CycleMix) that combine this with other loss terms.
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


def checkpoint_due(step, late_phase_start, early_interval, late_interval):
    """Shared eval+checkpoint cadence for every scribble-supervised 3D script.

    Coarse ``early_interval`` up to ``late_phase_start``, then finer
    ``late_interval`` after it -- e.g. every 5000 iterations for the first
    20k of a 30k-iteration run, then every 1000 for the remaining 10k, so
    expensive sliding-window validation runs less often early on and more
    often as training approaches convergence. The caller is still expected
    to also checkpoint unconditionally at ``step == max_iterations``.
    """
    if step <= 0:
        return False
    interval = early_interval if step <= late_phase_start else late_interval
    return step % interval == 0


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
