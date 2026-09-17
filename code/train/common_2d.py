"""Infrastructure shared by the scribble-supervised 2D training scripts.

ACDC and MSCMR train as independent 2D slices (anisotropic spacing), stitched
back into a full volume only at validation/test time -- see
``dataloader/scribblebench_2d.py`` for the training-time slice dataset and
augmentation. This module holds the piece specific to that stitching:
per-slice resize-then-stitch inference and Dice-only checkpoint selection.
Split resolution, checkpointing and seeding are dimension-agnostic and are
reused directly from ``train.common_3d`` rather than duplicated here.
"""

import math
from contextlib import nullcontext

import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from train.common_3d import finite_mean
from utils.sliding_window_3d import extract_logits


@torch.inference_mode()
def predict_volume_2d(model, image, patch_size, device, use_amp=False):
    """Per-slice resize-then-stitch inference for one ``[D, H, W]`` case.

    Matches ``dataloader.scribblebench_2d.RandomGenerator2D``'s nearest-
    neighbor resize convention and the WSL4MIS/DMSPS ``val_2D.py``
    ``test_single_volume`` pattern: every slice is independently resized to
    ``patch_size``, argmax'd, and resized back to its native resolution.
    """
    patch_size = tuple(int(value) for value in patch_size)
    depth, height, width = image.shape
    prediction = np.zeros((depth, height, width), dtype=np.int64)
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    model.eval()
    for index in range(depth):
        resized = zoom(image[index], (patch_size[0] / height, patch_size[1] / width), order=0)
        tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with amp_context:
            logits = extract_logits(model(tensor))
        out = torch.argmax(torch.softmax(logits, dim=1), dim=1)[0].cpu().numpy()
        prediction[index] = zoom(out, (height / patch_size[0], width / patch_size[1]), order=0)
    return prediction


@torch.inference_mode()
def validate_2d(model, dataset, indices, args, device, num_classes):
    """Select checkpoints using dense masks from the training holdout only.

    2D counterpart of ``train.common_3d.validate``: identical Dice-only
    checkpoint-selection convention (HD95/ASSD stay evaluator-only), the only
    difference is per-slice resize-then-stitch inference (``predict_volume_2d``)
    instead of 3D sliding-window inference.
    """
    model.eval()
    case_scores = []
    class_scores = {class_id: [] for class_id in range(1, num_classes)}
    for index in tqdm(indices, desc="validation", leave=False):
        sample = dataset[index]
        prediction = predict_volume_2d(model, sample["image"], args.patch_size, device, use_amp=args.amp)
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
