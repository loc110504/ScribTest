"""3D building blocks for CycleMix (Zhang & Zhuang, CVPR 2022).

Reimplements the four-loss CycleMix framework (unmix pCE, mix pCE, global
mix-invariance consistency, local connectivity consistency) for volumetric
patches. The official repository (``BWGZK/CycleMix``) builds its "increment"
mix with Puzzle Mix (graph-cut + optimal transport over 2D slices); that
solver has no 3D equivalent and would be prohibitively expensive over a full
volume. Puzzle Mix is one instantiation of the paper's general mix operator
``M(a1, a2) = (1-z) * a1 + z * a2`` (Eq. 1-2); we instead sample ``z`` as an
axis-aligned cuboid indicator, i.e. 3D CutMix, which the paper explicitly
allows ("applicable to other mixup strategies, such as MixUp, CutMix"). The
four-loss objective and the increment/decrement (mix-then-occlude) mechanism
are otherwise implemented exactly as described.
"""

import numpy as np
import torch
from scipy.ndimage import label as connected_components

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.int8)


def sample_cuboid_mask(patch_size, frac_range, rng=None):
    """Sample a random axis-aligned cuboid boolean mask over ``patch_size``.

    Each axis independently gets a box whose fraction of that axis length is
    drawn from ``frac_range`` (``(low, high)``, both in ``(0, 1]``).
    """
    if rng is None:
        rng = np.random
    low, high = frac_range
    mask = np.zeros(patch_size, dtype=bool)
    slices = []
    for axis_size in patch_size:
        frac = rng.uniform(low, high)
        box = max(1, min(axis_size, int(round(axis_size * frac))))
        start = rng.randint(0, axis_size - box + 1) if axis_size > box else 0
        slices.append(slice(start, start + box))
    mask[tuple(slices)] = True
    return mask


def sample_batch_cuboid_masks(batch_size, patch_size, frac_range, device, rng=None):
    """Return a ``[B, 1, D, H, W]`` bool mask, one independent box per sample."""
    masks = np.stack(
        [sample_cuboid_mask(patch_size, frac_range, rng=rng) for _ in range(batch_size)]
    )
    return torch.from_numpy(masks).unsqueeze(1).to(device)


def mix_images(image_a, image_b, mask):
    """``M(a, b)``: take ``b`` inside ``mask``, ``a`` elsewhere."""
    return torch.where(mask, image_b, image_a)


def mix_labels(label_a, label_b, mask):
    """Label counterpart of :func:`mix_images`; ``mask`` is ``[B, 1, ...]``."""
    return torch.where(mask.squeeze(1), label_b, label_a)


def occlude(image, label, mask, ignore_index):
    """Decrement step: blank ``image`` and drop supervision inside ``mask``."""
    occluded_image = image.masked_fill(mask, 0.0)
    occluded_label = label.masked_fill(mask.squeeze(1), ignore_index)
    return occluded_image, occluded_label


def negative_cosine_similarity(p, q, dim=1, eps=1e-8):
    """Mean per-voxel negative cosine similarity (Eq. 9-13 ``L_ncs``)."""
    if p.shape != q.shape:
        raise ValueError("negative_cosine_similarity expects matching shapes, got {} and {}".format(p.shape, q.shape))
    p = p / p.norm(dim=dim, keepdim=True).clamp_min(eps)
    q = q / q.norm(dim=dim, keepdim=True).clamp_min(eps)
    return -(p * q).sum(dim=dim).mean()


def largest_component_targets(probs):
    """Keep the largest connected component per foreground class (Eq. 13 ``C(.)``).

    ``probs`` is a softmax probability map ``[B, C, D, H, W]``. Returns a
    detached one-hot tensor of the same shape where, for every sample and
    every foreground class, only the single largest 26-connected component of
    the class's ``argmax`` mask survives; everything else (including smaller
    components of the same class) is reassigned to background.
    """
    batch_size, num_classes = probs.shape[0], probs.shape[1]
    argmax = probs.detach().argmax(dim=1).cpu().numpy()
    cleaned = np.zeros_like(argmax)
    for b in range(batch_size):
        for class_id in range(1, num_classes):
            class_mask = argmax[b] == class_id
            if not class_mask.any():
                continue
            labeled, num_components = connected_components(class_mask, structure=_STRUCTURE_26)
            if num_components == 0:
                continue
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # background component
            largest = int(np.argmax(sizes))
            cleaned[b][labeled == largest] = class_id
    cleaned_t = torch.from_numpy(cleaned).to(probs.device)
    one_hot = torch.nn.functional.one_hot(cleaned_t.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 4, 1, 2, 3).contiguous().to(probs.dtype)
    return one_hot
