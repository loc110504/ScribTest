import numpy as np
import torch


def build_dual_branch_quantum_pseudo_label(
    prob_u,
    prob_m,
    Q_u,
    Q_m,
    label,
    ignore_index,
    agree_thresh=0.70,
    disagree_thresh=0.80,
    margin_thresh=0.10,
    tau_q=0.50,
    pseudo_mask_mode="unlabeled",
    eps=1e-8,
):
    prob_u = prob_u.detach()
    prob_m = prob_m.detach()
    Q_u = Q_u.detach()
    Q_m = Q_m.detach()

    if label.dim() == 4:
        label = label.squeeze(1)

    if pseudo_mask_mode == "unlabeled":
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == "all":
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError("Unsupported pseudo_mask_mode: {}".format(pseudo_mask_mode))

    conf_u, pred_u = torch.max(prob_u, dim=1)
    conf_m, pred_m = torch.max(prob_m, dim=1)

    same_pred = pred_u == pred_m
    diff_pred = pred_u != pred_m

    min_conf = torch.minimum(conf_u, conf_m)
    max_conf = torch.maximum(conf_u, conf_m)
    margin = torch.abs(conf_u - conf_m)

    reliable_agree = same_pred & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = diff_pred & (max_conf >= disagree_thresh) & (margin >= margin_thresh) & candidate_mask

    mean_pseudo = 0.5 * (prob_u + prob_m)
    choose_u = (conf_u > conf_m).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_u, prob_u, prob_m)

    soft_pseudo = torch.where(
        reliable_disagree.unsqueeze(1),
        high_conf_pseudo,
        mean_pseudo,
    )
    soft_pseudo = soft_pseudo / (soft_pseudo.sum(dim=1, keepdim=True) + eps)

    pseudo_conf, pseudo_pred = torch.max(soft_pseudo, dim=1)
    qconf_u, qpred_u = torch.max(Q_u, dim=1)
    qconf_m, qpred_m = torch.max(Q_m, dim=1)

    branch_reliable = reliable_agree | reliable_disagree

    quantum_reliable_u = (pseudo_pred == qpred_u) & (qconf_u >= tau_q) & candidate_mask
    quantum_reliable_m = (pseudo_pred == qpred_m) & (qconf_m >= tau_q) & candidate_mask

    R_u = branch_reliable & quantum_reliable_u
    R_m = branch_reliable & quantum_reliable_m

    soft_pseudo_u = soft_pseudo
    soft_pseudo_m = soft_pseudo

    return {
        "soft_pseudo_u": soft_pseudo_u.detach(),
        "soft_pseudo_m": soft_pseudo_m.detach(),
        "pseudo_u": torch.argmax(soft_pseudo_u, dim=1).detach(),
        "pseudo_m": torch.argmax(soft_pseudo_m, dim=1).detach(),
        "R_u": R_u.detach(),
        "R_m": R_m.detach(),
        "reliable_agree": reliable_agree.detach(),
        "reliable_disagree": reliable_disagree.detach(),
        "branch_reliable": branch_reliable.detach(),
        "candidate_mask": candidate_mask.detach(),
        "conf_u": conf_u.detach(),
        "conf_m": conf_m.detach(),
        "min_conf": min_conf.detach(),
        "max_conf": max_conf.detach(),
        "margin": margin.detach(),
        "pseudo_conf": pseudo_conf.detach(),
        "qconf_u": qconf_u.detach(),
        "qconf_m": qconf_m.detach(),
        "agreement_u": (pseudo_pred == qpred_u).detach(),
        "agreement_m": (pseudo_pred == qpred_m).detach(),
    }


def build_quantum_guided_reliable_masks(
    prob_u,
    prob_m,
    Q_u,
    Q_m,
    label,
    ignore_index,
    tau_u,
    tau_m,
    tau_q,
):
    conf_u, pseudo_u = torch.max(prob_u.detach(), dim=1)
    conf_m, pseudo_m = torch.max(prob_m.detach(), dim=1)
    qconf_u, qlabel_u = torch.max(Q_u.detach(), dim=1)
    qconf_m, qlabel_m = torch.max(Q_m.detach(), dim=1)

    unknown = label == ignore_index
    agreement_u = pseudo_u == qlabel_u
    agreement_m = pseudo_m == qlabel_m

    R_u = agreement_u & (conf_u > tau_u) & (qconf_u > tau_q) & unknown
    R_m = agreement_m & (conf_m > tau_m) & (qconf_m > tau_q) & unknown

    return {
        "pseudo_u": pseudo_u,
        "pseudo_m": pseudo_m,
        "R_u": R_u,
        "R_m": R_m,
        "conf_u": conf_u,
        "conf_m": conf_m,
        "qconf_u": qconf_u,
        "qconf_m": qconf_m,
        "agreement_u": agreement_u,
        "agreement_m": agreement_m,
    }


def safe_masked_mean(values, mask):
    if mask.sum() < 1:
        return np.nan
    return float(values[mask].mean().item())


def summarize_reliable_masks(pseudo_info):
    R_u = pseudo_info["R_u"]
    R_m = pseudo_info["R_m"]
    agreement_u = pseudo_info["agreement_u"]
    agreement_m = pseudo_info["agreement_m"]
    stats = {
        "reliable_ratio_u": R_u.float().mean(),
        "reliable_ratio_m": R_m.float().mean(),
        "agreement_ratio_u": agreement_u.float().mean(),
        "agreement_ratio_m": agreement_m.float().mean(),
        "conf_u_mean": safe_masked_mean(pseudo_info["conf_u"], R_u),
        "conf_m_mean": safe_masked_mean(pseudo_info["conf_m"], R_m),
        "qconf_u_mean": safe_masked_mean(pseudo_info["qconf_u"], R_u),
        "qconf_m_mean": safe_masked_mean(pseudo_info["qconf_m"], R_m),
    }
    if "reliable_agree" in pseudo_info:
        stats["reliable_agree_ratio"] = pseudo_info["reliable_agree"].float().mean()
    if "reliable_disagree" in pseudo_info:
        stats["reliable_disagree_ratio"] = pseudo_info["reliable_disagree"].float().mean()
    if "branch_reliable" in pseudo_info:
        stats["branch_reliable_ratio"] = pseudo_info["branch_reliable"].float().mean()
    if "pseudo_conf" in pseudo_info:
        stats["pseudo_conf_u_mean"] = safe_masked_mean(pseudo_info["pseudo_conf"], R_u)
        stats["pseudo_conf_m_mean"] = safe_masked_mean(pseudo_info["pseudo_conf"], R_m)
    if "margin" in pseudo_info:
        stats["margin_u_mean"] = safe_masked_mean(pseudo_info["margin"], R_u)
        stats["margin_m_mean"] = safe_masked_mean(pseudo_info["margin"], R_m)
    return stats


def masked_label_accuracy(pred, target, mask):
    if mask.sum() < 1:
        return np.nan
    return float((pred[mask] == target[mask]).float().mean().item())
