import math

import torch
import torch.nn.functional as F


def js_divergence_map(probs_a, probs_b, eps=1e-8, normalize=True):
    """returns Tensor[B,1,H,W]"""
    pa = probs_a.clamp_min(eps)
    pb = probs_b.clamp_min(eps)
    m = 0.5 * (pa + pb)
    kl_am = (pa * (torch.log(pa) - torch.log(m.clamp_min(eps)))).sum(dim=1, keepdim=True)
    kl_bm = (pb * (torch.log(pb) - torch.log(m.clamp_min(eps)))).sum(dim=1, keepdim=True)
    js = 0.5 * (kl_am + kl_bm)
    if normalize:
        js = js / math.log(2.0)
    return js.clamp(0.0, 1.0)


def temporal_teacher_arbitration(
    probs_fast,
    probs_slow,
    entropy_fast,
    entropy_slow,
    disagreement,
    boundary=None,
    gamma_boundary=0.5,
    gamma_disagree=0.5,
    gamma_stable=0.5,
    eps=1e-8,
):
    """returns dict with selected_probs, selected_weight, select_fast, R_fast, R_slow"""
    if boundary is None:
        boundary = torch.zeros_like(disagreement)

    boundary = boundary.clamp(0.0, 1.0)
    disagreement = disagreement.clamp(0.0, 1.0)

    r_fast = (1.0 - entropy_fast) * (1.0 + gamma_boundary * boundary) * (1.0 + gamma_disagree * disagreement)
    r_slow = (1.0 - entropy_slow) * (1.0 + gamma_boundary * (1.0 - boundary)) * (1.0 + gamma_stable * (1.0 - disagreement))

    select_fast = r_fast > r_slow
    selected_probs = torch.where(select_fast, probs_fast, probs_slow)
    selected_weight = torch.where(select_fast, r_fast, r_slow)
    selected_weight = selected_weight / (selected_weight.detach().mean() + eps)
    selected_weight = selected_weight.clamp(0.0, 3.0)

    return {
        'selected_probs': selected_probs,
        'selected_weight': selected_weight,
        'select_fast': select_fast,
        'R_fast': r_fast,
        'R_slow': r_slow,
    }


def select_teacher_feature(feat_fast, feat_slow, select_fast):
    mask = F.interpolate(select_fast.float(), size=feat_fast.shape[-2:], mode='nearest') > 0.5
    return torch.where(mask, feat_fast, feat_slow)
