"""3D building blocks for DMSPS (Han et al., Medical Image Analysis 2024).

Reimplements the dual-branch dynamically-mixed soft pseudo-label supervision
loss and the uncertainty-guided stage-2 label expansion, verified against the
official ``HiLab-git/DMSPS`` repository. The dual-decoder network required by
the method (shared encoder, one clean decoder, one decoder fed
``dropout3d(p=0.5)``-perturbed features at every encoder stage) is this
repository's existing ``UNetCCT3D`` (``networks/unet_cct_3d.py``); no new
network code is needed.
"""

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from utils.entropy_utils import normalized_entropy
from utils.sliding_window_3d import _allocate, _gaussian, _scan_starts
from scipy.ndimage import label as connected_components

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.int8)


def dynamic_mixed_pseudo_label(probs_main, probs_aux, alpha):
    """Eq. 1: ``p_hat = alpha * p1 + (1 - alpha) * p2``, detached."""
    if probs_main.shape != probs_aux.shape:
        raise ValueError("probs_main and probs_aux must have matching shapes")
    mixed = alpha * probs_main + (1.0 - alpha) * probs_aux
    return mixed.detach()


def soft_pseudo_supervision_loss(probs_main, probs_aux, pseudo_target):
    """Eq. 2-3 ``L_SPS``.

    The official implementation feeds already-softmaxed probabilities into
    ``nn.CrossEntropyLoss``, which applies ``log_softmax`` a second time. This
    is intentional and verified against the source (not a paraphrase bug): it
    must be kept exactly as-is to match the published method, rather than
    "corrected" to a plain soft cross-entropy ``-sum(target * log(p))``.
    """
    ce = torch.nn.CrossEntropyLoss()
    return 0.5 * (ce(probs_main, pseudo_target) + ce(probs_aux, pseudo_target))


def expand_labels(scribble, mean_probs, ignore_index, tau):
    """Stage-2 uncertainty-guided label expansion (Eq. 5-8).

    Args:
        scribble: ``[D, H, W]`` int64 array, the original sparse scribble
            (``ignore_index`` where unlabeled).
        mean_probs: ``[C, D, H, W]`` float32 array, ``p_bar = 0.5*(p1+p2)``
            averaged over the full volume.
        ignore_index: unlabeled class id.
        tau: normalized-entropy threshold below which a voxel is "confident".

    Returns:
        ``[D, H, W]`` int64 expanded label: original scribble is preserved
        wherever annotated; elsewhere, for every class, only the single
        largest 26-connected component of that class's confident region is
        kept as new pseudo-supervision.
    """
    if mean_probs.ndim != 4:
        raise ValueError("mean_probs must be [C, D, H, W]")
    num_classes = mean_probs.shape[0]
    entropy = -(mean_probs * np.log(np.clip(mean_probs, 1e-6, None))).sum(axis=0)
    entropy = entropy / np.log(num_classes)
    confident = entropy < tau
    class_map = np.argmax(mean_probs, axis=0)

    expanded = np.full(scribble.shape, ignore_index, dtype=np.int64)
    for class_id in range(num_classes):
        candidate = confident & (class_map == class_id)
        if not candidate.any():
            continue
        labeled, num_components = connected_components(candidate, structure=_STRUCTURE_26)
        if num_components == 0:
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        expanded[labeled == largest] = class_id

    scribbled = scribble != ignore_index
    expanded = np.where(scribbled, scribble, expanded)
    return expanded.astype(np.int64)


@torch.inference_mode()
def dual_branch_volume_probs(
    model,
    image,
    num_classes,
    patch_size,
    device,
    overlap=0.5,
    sw_batch_size=1,
    use_amp=False,
    max_accumulator_mb=1024,
    temp_dir=None,
):
    """Sliding-window ``p_bar = 0.5*(softmax(main), softmax(aux))`` for one volume.

    Mirrors ``sliding_window_3d.sliding_window_predict`` but accumulates the
    averaged two-decoder probability map instead of single-branch logits,
    which stage-2 label expansion needs.
    """
    patch_size = tuple(int(value) for value in patch_size)
    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, C, D, H, W]")
    original_shape = tuple(int(value) for value in image.shape[2:])
    axis_padding = []
    for current, target in zip(original_shape, patch_size):
        total = max(target - current, 0)
        axis_padding.append((total // 2, total - total // 2))
    d_pad, h_pad, w_pad = axis_padding
    image = F.pad(
        image.cpu().float(), (w_pad[0], w_pad[1], h_pad[0], h_pad[1], d_pad[0], d_pad[1])
    )
    spatial_shape = tuple(int(value) for value in image.shape[2:])
    starts = _scan_starts(spatial_shape, patch_size, overlap)
    importance = _gaussian(patch_size, sigma_scale=0.125)
    scores, counts, temporary = _allocate(
        (num_classes,) + spatial_shape, spatial_shape, max_accumulator_mb, temp_dir
    )
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and device.type == "cuda"
        else nullcontext()
    )
    model.eval()
    try:
        for offset in range(0, len(starts), sw_batch_size):
            batch_starts = starts[offset : offset + sw_batch_size]
            patches = torch.stack(
                [
                    image[0, :, d : d + patch_size[0], h : h + patch_size[1], w : w + patch_size[2]]
                    for d, h, w in batch_starts
                ]
            ).to(device, non_blocking=True)
            with amp_context:
                main_logits, aux_logits = model(patches, return_auxiliary=True)
                mean_probs = 0.5 * (F.softmax(main_logits, dim=1) + F.softmax(aux_logits, dim=1))
            if tuple(mean_probs.shape) != (len(batch_starts), num_classes, *patch_size):
                raise ValueError("Unexpected model output shape: {}".format(tuple(mean_probs.shape)))
            mean_probs = mean_probs.float().cpu().numpy()
            for batch_index, (d, h, w) in enumerate(batch_starts):
                region = (
                    slice(d, d + patch_size[0]),
                    slice(h, h + patch_size[1]),
                    slice(w, w + patch_size[2]),
                )
                scores[(slice(None),) + region] += mean_probs[batch_index] * importance[None]
                counts[region] += importance
        probs = np.asarray(scores) / np.asarray(counts)[None]
        crop = tuple(
            slice(before, before + size) for (before, _), size in zip(axis_padding, original_shape)
        )
        return np.ascontiguousarray(probs[(slice(None),) + crop])
    finally:
        del scores, counts
        if temporary is not None:
            temporary.cleanup()


__all__ = [
    "dynamic_mixed_pseudo_label",
    "soft_pseudo_supervision_loss",
    "expand_labels",
    "dual_branch_volume_probs",
    "normalized_entropy",
]
