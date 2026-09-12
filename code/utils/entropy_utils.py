import math

import torch


def normalized_entropy(probs, eps=1e-8, keepdim=True):
    ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
    ent = ent / math.log(probs.shape[1])
    ent = ent.clamp(0.0, 1.0)
    if not keepdim:
        ent = ent.squeeze(1)
    return ent


def build_uncertain_mask(
    entropy,
    scribble,
    mode="quantile",
    top_ratio=0.35,
    threshold=0.5,
    ignore_index=4,
):
    if entropy.dim() != 4 or entropy.size(1) != 1:
        raise ValueError(f"entropy must be [B,1,H,W], got {entropy.shape}")
    if scribble.dim() != 3:
        raise ValueError(f"scribble must be [B,H,W], got {scribble.shape}")

    unlabeled = (scribble == ignore_index).unsqueeze(1)

    if mode == "fixed":
        return (entropy >= threshold) & unlabeled

    if mode != "quantile":
        raise ValueError(f"Unknown uncertain mode: {mode}")

    mask = torch.zeros_like(entropy, dtype=torch.bool)
    for b in range(entropy.shape[0]):
        valid = unlabeled[b:b + 1]
        vals = entropy[b:b + 1][valid]
        if vals.numel() == 0:
            continue
        q = torch.quantile(vals.detach(), 1.0 - top_ratio)
        mask[b:b + 1] = (entropy[b:b + 1] >= q) & valid
    return mask


def teacher_confidence_mask(selected_probs, threshold=0.55):
    conf = selected_probs.max(dim=1, keepdim=True)[0]
    return conf >= threshold
