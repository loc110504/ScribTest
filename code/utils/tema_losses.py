import torch
import torch.nn.functional as F


def weighted_soft_ce_loss(logits, soft_targets, mask, weight=None, eps=1e-8):
    mask_f = mask.float()
    if mask_f.sum() < 1:
        return logits.sum() * 0.0

    log_probs = F.log_softmax(logits, dim=1)
    loss_map = -(soft_targets * log_probs).sum(dim=1, keepdim=True)
    if weight is not None:
        loss_map = loss_map * weight.float()
    return (loss_map * mask_f).sum() / (mask_f.sum() + eps)


def weighted_feature_consistency_loss(feat_s, feat_t, mask, weight=None, eps=1e-8):
    mask_f = mask.float()
    if mask_f.shape[-2:] != feat_s.shape[-2:]:
        mask_f = F.interpolate(mask_f, size=feat_s.shape[-2:], mode='nearest')

    if mask_f.sum() < 1:
        return feat_s.sum() * 0.0

    if weight is None:
        weight_f = torch.ones_like(mask_f)
    else:
        weight_f = weight.float()
        if weight_f.shape[-2:] != feat_s.shape[-2:]:
            weight_f = F.interpolate(weight_f, size=feat_s.shape[-2:], mode='nearest')

    denom = mask_f.sum() + eps
    l1_map = torch.abs(feat_s - feat_t).mean(dim=1, keepdim=True)
    l1 = (l1_map * mask_f * weight_f).sum() / denom

    cos_map = F.cosine_similarity(feat_s, feat_t, dim=1).unsqueeze(1)
    cos = ((1.0 - cos_map) * mask_f * weight_f).sum() / denom

    return 0.5 * (l1 + cos)
