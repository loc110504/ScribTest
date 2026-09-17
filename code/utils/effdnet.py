"""EFFDNet: Enhanced Foreground Feature Discrimination Network (Liu et al.,
MICCAI 2025) building blocks, verified against the official
``Aurora-003-web/EFFDNet`` repository.

Mean-Teacher framework (student + an *independently initialized* EMA
teacher -- the source calls ``create_model()`` twice with no state-dict copy,
unlike this repo's SDT-Net/VoxTrust-3D, which is why ``update_ema_variables``
below is its own function rather than a reuse of ``utils.ema_optim.WeightEMA``)
with two additional losses on top of partial cross-entropy and a dense
teacher-pseudo-label cross-entropy:

1. Foreground-Background Separation Loss (FBSL): a modified SupCon-style
   contrastive loss over a coarse grid of aggregated features, pulling
   foreground-labeled grid cells together and background-labeled ones
   together while separating the two groups.
2. Foreground Augmentation with Diverse Context (FADC): a batch-level
   copy-paste augmentation that swaps each sample's own annotated bounding
   box for a resized crop of another sample's annotated region.

Both are shape-agnostic over 2D ``[B, C, H, W]`` / 3D ``[B, C, D, H, W]``
(dispatched on tensor rank), matching this repo's convention elsewhere
(``utils/{cyclemix,dmsps,sdtnet,voxtrust3d}.py``).
"""

import random

import torch
import torch.nn.functional as F


def update_ema_variables(model, ema_model, alpha, global_step):
    """Classic Mean-Teacher EMA update (Tarvainen & Valpola, 2017) with a
    warm-up ramp on alpha, verified against the official EFFDNet source.

    Unlike ``utils.ema_optim.WeightEMA``, there is no extra weight decay
    applied to the student, and alpha ramps from 0 up to its target value
    over the first few steps (``alpha_t = min(1 - 1/(t+1), alpha)``) instead
    of being fixed from step 0.
    """
    alpha = min(1.0 - 1.0 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1.0 - alpha)


def _adaptive_avg_pool(x, output_size):
    if x.ndim == 4:
        return F.adaptive_avg_pool2d(x, output_size)
    if x.ndim == 5:
        return F.adaptive_avg_pool3d(x, output_size)
    raise ValueError("Expected a 4D [B,C,H,W] or 5D [B,C,D,H,W] tensor, got ndim={}".format(x.ndim))


def _adaptive_max_pool(x, output_size):
    if x.ndim == 4:
        return F.adaptive_max_pool2d(x, output_size)
    if x.ndim == 5:
        return F.adaptive_max_pool3d(x, output_size)
    raise ValueError("Expected a 4D [B,1,H,W] or 5D [B,1,D,H,W] tensor, got ndim={}".format(x.ndim))


def foreground_background_region_labels(label, ignore_index, num_regions):
    """Eq. 4: per-grid-cell foreground/background region label.

    Args:
        label: ``[B, H, W]`` (2D) or ``[B, D, H, W]`` (3D) int64 scribble labels.
        num_regions: ``K`` -- grid resolution per spatial axis (``K^2`` cells
            for 2D, ``K^3`` for 3D). Uses adaptive pooling rather than the
            source's fixed ``ceil(size/K)`` block slicing, so it also works
            cleanly when a spatial size is not evenly divisible by ``K``.

    Returns:
        Bool tensor ``[B, K, K]`` or ``[B, K, K, K]``: ``True`` iff the cell
        contains at least one foreground-class (``label > 0``, excluding
        ``ignore_index``) scribble pixel -- matches the source's
        "zero the ignore class, then any positive class value makes the
        cell's sum > 0" rule exactly, since background is class ``0``.
    """
    if label.ndim not in (3, 4):
        raise ValueError("label must be [B,H,W] or [B,D,H,W], got ndim={}".format(label.ndim))
    is_foreground = ((label > 0) & (label != ignore_index)).float().unsqueeze(1)
    region_shape = (num_regions,) * (label.ndim - 1)
    pooled = _adaptive_max_pool(is_foreground, region_shape)
    return pooled.squeeze(1) > 0


def foreground_background_separation_loss(feature, label, ignore_index, num_regions=8, temperature=0.07):
    """FBSL (Eq. 3-5): modified SupCon-style contrastive loss over grid-
    aggregated features.

    Verified against the official repo's ``SupConLoss``: for an anchor-
    positive pair ``(i, p)``, the denominator sums only that pair's own
    similarity term plus every *negative* pair's term (excludes other
    positives) -- ``-log(exp(s_ip) / (exp(s_ip) + sum_negatives(exp(s_in))))``,
    not the vanilla SupCon paper's "sum over all others" denominator.
    Reimplemented here with a plain pairwise cosine-similarity matrix instead
    of the source's conv2d-as-batched-dot-product trick, for an identical
    numeric result with more readable code.

    Args:
        feature: ``[B, C, H, W]`` or ``[B, C, D, H, W]`` features from the
            layer preceding the segmentation head (e.g. ``UNet2D``/``VNet3D``
            with ``return_features=True``, ``features["decoder"][-1]``).
        label: ``[B, H, W]`` or ``[B, D, H, W]`` scribble labels. Only used to
            derive each grid cell's region label; its own spatial resolution
            need not match ``feature``'s.
    """
    if feature.ndim not in (4, 5):
        raise ValueError("feature must be [B,C,H,W] or [B,C,D,H,W], got ndim={}".format(feature.ndim))
    spatial_ndim = feature.ndim - 2
    region_shape = (num_regions,) * spatial_ndim
    pooled = _adaptive_avg_pool(feature, region_shape)  # [B, C, *region_shape]

    channels = pooled.shape[1]
    normalized = pooled.reshape(pooled.shape[0], channels, -1)  # [B, C, K^d]
    normalized = normalized.permute(0, 2, 1).reshape(-1, channels)  # [B*K^d, C]
    normalized = F.normalize(normalized, p=2, dim=1)

    region_label = foreground_background_region_labels(label, ignore_index, num_regions)
    region_label = region_label.reshape(-1)  # [B*K^d]

    similarity = torch.matmul(normalized, normalized.t()) / temperature  # [N, N]
    n = similarity.shape[0]
    same_label = region_label.unsqueeze(0) == region_label.unsqueeze(1)
    self_mask = torch.eye(n, dtype=torch.bool, device=feature.device)
    positive_mask = (same_label & ~self_mask).float()
    negative_mask = (~same_label).float()

    exp_sim = torch.exp(similarity)
    negative_sum = (exp_sim * negative_mask).sum(dim=1, keepdim=True)
    log_prob = torch.log(exp_sim / (exp_sim + negative_sum + 1e-12) + 1e-12)

    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1e-12)
    return -mean_log_prob_pos.mean()


def _annotated_bbox(label, ignore_index):
    """Tight bounding box enclosing every annotated (non-``ignore_index``)
    pixel of a single ``[H, W]``/``[D, H, W]`` label array -- every annotated
    class, not just foreground, matching the source's ``label < ignore_index``
    locator (verified against source; the paper text describes it as a
    "foreground bounding box", but the released code's mask covers any
    annotated scribble, background included).

    Returns a tuple of ``(start, stop)`` per axis (``stop`` exclusive), or
    ``None`` if the label has no annotation at all.
    """
    mask = label != ignore_index
    if not bool(mask.any()):
        return None
    coords = torch.nonzero(mask, as_tuple=False)
    mins = coords.amin(dim=0)
    maxs = coords.amax(dim=0) + 1
    return tuple(zip(mins.tolist(), maxs.tolist()))


def _resize_nearest(x, size):
    return F.interpolate(x, size=size, mode="nearest")


def foreground_augmentation_diverse_context(image, label, pseudo_label, ignore_index):
    """FADC (Eq. 6-8): swap each sample's own annotated bounding-box region
    for a resized crop of another (randomly chosen) sample's own annotated
    region, keeping the receiving sample's background outside that box
    untouched. Increases foreground context diversity without changing
    scribble/pseudo-label geometry relative to the receiving sample.

    Shape-agnostic over spatial rank: ``image`` is ``[B, C, *spatial]``,
    ``label``/``pseudo_label`` are ``[B, *spatial]``, ``spatial`` being
    ``(H, W)`` or ``(D, H, W)``. A sample with no annotation at all (or when
    no sample in the batch has one) is left unchanged, matching the source's
    size-guarded skip.
    """
    batch_size = image.shape[0]
    bboxes = [_annotated_bbox(label[i], ignore_index) for i in range(batch_size)]
    crops = []
    for i, bbox in enumerate(bboxes):
        if bbox is None:
            crops.append(None)
            continue
        slices = tuple(slice(start, stop) for start, stop in bbox)
        crops.append((image[i][(slice(None),) + slices], label[i][slices], pseudo_label[i][slices]))
    donor_candidates = [i for i, crop in enumerate(crops) if crop is not None]

    out_image, out_label, out_pseudo = image.clone(), label.clone(), pseudo_label.clone()
    if not donor_candidates:
        return out_image, out_label, out_pseudo

    for i in range(batch_size):
        if bboxes[i] is None:
            continue
        target_shape = tuple(stop - start for start, stop in bboxes[i])
        donor_index = random.choice(donor_candidates)
        donor_image, donor_label, donor_pseudo = crops[donor_index]

        resized_image = _resize_nearest(donor_image.unsqueeze(0), target_shape)[0]
        resized_label = (
            _resize_nearest(donor_label.float().unsqueeze(0).unsqueeze(0), target_shape).round().long()[0, 0]
        )
        resized_pseudo = (
            _resize_nearest(donor_pseudo.float().unsqueeze(0).unsqueeze(0), target_shape).round().long()[0, 0]
        )

        slices = tuple(slice(start, stop) for start, stop in bboxes[i])
        out_image[(i, slice(None)) + slices] = resized_image
        out_label[(i,) + slices] = resized_label
        out_pseudo[(i,) + slices] = resized_pseudo
    return out_image, out_label, out_pseudo
