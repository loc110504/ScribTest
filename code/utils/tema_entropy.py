import math

import torch


def normalized_entropy(probs, eps=1e-8, keepdim=True):
    """
    probs: Tensor[B,C,H,W]
    returns: Tensor[B,1,H,W] in [0,1]
    """
    ent = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1, keepdim=True)
    ent = ent / math.log(probs.shape[1])
    ent = ent.clamp(0.0, 1.0)
    if not keepdim:
        ent = ent.squeeze(1)
    return ent


def build_uncertain_mask(
    entropy,
    scribble,
    mode='quantile',
    top_ratio=0.35,
    threshold=0.5,
    ignore_index=4,
):
    """
    entropy: Tensor[B,1,H,W]
    scribble: Tensor[B,H,W]
    returns BoolTensor[B,1,H,W]
    """
    unlabeled = (scribble == ignore_index).unsqueeze(1)

    if mode == 'fixed':
        return (entropy >= threshold) & unlabeled

    if mode != 'quantile':
        raise ValueError(f'Unknown uncertain mode: {mode}')

    ratio = float(max(0.0, min(1.0, top_ratio)))
    if ratio <= 0.0:
        return torch.zeros_like(unlabeled, dtype=torch.bool)
    if ratio >= 1.0:
        return unlabeled

    mask = torch.zeros_like(unlabeled, dtype=torch.bool)
    for b in range(entropy.shape[0]):
        ent_b = entropy[b, 0]
        unl_b = unlabeled[b, 0]
        vals = ent_b[unl_b]
        if vals.numel() == 0:
            continue
        q = torch.quantile(vals.detach(), 1.0 - ratio)
        mask[b, 0] = (ent_b >= q) & unl_b
    return mask


def teacher_confidence_mask(probs, threshold=0.55):
    return probs.max(dim=1, keepdim=True)[0] >= threshold
