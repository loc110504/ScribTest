import torch
import torch.nn.functional as F

from utils.entropy_utils import normalized_entropy


def region_wise_teacher_selection(probs_l, probs_g, boundary, gamma=0.5, eps=1e-8):
    h_l = normalized_entropy(probs_l)
    h_g = normalized_entropy(probs_g)

    r_l = (1.0 - h_l) * (1.0 + gamma * boundary)
    r_g = (1.0 - h_g) * (1.0 + gamma * (1.0 - boundary))

    select_local = r_l > r_g
    selected_probs = torch.where(select_local, probs_l, probs_g)
    selected_weight = torch.where(select_local, r_l, r_g)

    selected_weight = selected_weight / (selected_weight.detach().mean() + eps)
    selected_weight = selected_weight.clamp(0.0, 3.0)
    selected_hard = selected_probs.argmax(dim=1)

    return {
        "selected_probs": selected_probs,
        "selected_hard": selected_hard,
        "selected_weight": selected_weight,
        "select_local": select_local,
        "R_l": r_l,
        "R_g": r_g,
        "entropy_l": h_l,
        "entropy_g": h_g,
    }


def select_teacher_feature(feat_l, feat_g, select_local, target_size=None):
    if target_size is None:
        target_size = feat_l.shape[-2:]
    mask = F.interpolate(select_local.float(), size=target_size, mode="nearest") > 0.5
    return torch.where(mask, feat_l, feat_g)
