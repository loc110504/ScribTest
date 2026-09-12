import torch
import torch.nn.functional as F


def weighted_soft_ce_loss(logits, soft_targets, mask, weight=None, eps=1e-8):
    mask_f = mask.float()
    if mask_f.sum() < 1:
        return logits.sum() * 0.0

    log_probs = F.log_softmax(logits, dim=1)
    per_pixel = -(soft_targets * log_probs).sum(dim=1, keepdim=True)
    if weight is not None:
        per_pixel = per_pixel * weight.float()
    return (per_pixel * mask_f).sum() / (mask_f.sum() + eps)


def weighted_feature_consistency_loss(feat_s, feat_t, mask, weight=None, eps=1e-8):
    mask_f = mask.float()
    if mask_f.shape[-2:] != feat_s.shape[-2:]:
        mask_f = F.interpolate(mask_f, size=feat_s.shape[-2:], mode="nearest")

    if weight is None:
        weight_f = torch.ones_like(mask_f)
    else:
        weight_f = weight.float()
        if weight_f.shape[-2:] != feat_s.shape[-2:]:
            weight_f = F.interpolate(weight_f, size=feat_s.shape[-2:], mode="nearest")

    if mask_f.sum() < 1:
        return feat_s.sum() * 0.0

    denom = mask_f.sum() + eps
    l1_map = torch.abs(feat_s - feat_t).mean(dim=1, keepdim=True)
    l1 = (l1_map * mask_f * weight_f).sum() / denom

    cos_map = F.cosine_similarity(feat_s, feat_t, dim=1).unsqueeze(1)
    cos_loss_map = 1.0 - cos_map
    cos = (cos_loss_map * mask_f * weight_f).sum() / denom
    return 0.5 * (l1 + cos)


def masked_symmetric_kl_loss(probs_a, probs_b, mask, eps=1e-8):
    mask_f = mask.float()
    if mask_f.sum() < 1:
        return probs_a.sum() * 0.0

    log_a = torch.log(probs_a.clamp_min(eps))
    log_b = torch.log(probs_b.clamp_min(eps))
    kl_ab = (probs_a.detach() * (log_a.detach() - log_b)).sum(dim=1, keepdim=True)
    kl_ba = (probs_b.detach() * (log_b.detach() - log_a)).sum(dim=1, keepdim=True)

    loss_map = 0.5 * (kl_ab + kl_ba)
    return (loss_map * mask_f).sum() / (mask_f.sum() + eps)
