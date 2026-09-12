from typing import Dict

import torch
import torch.nn.functional as F


def unpack_model_output(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Model output must be a tensor or tuple/list with tensor first")

    if output.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], received {tuple(output.shape)}")

    return output


def evidential_prediction(
    raw_output: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    if raw_output.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W], received {tuple(raw_output.shape)}")

    if raw_output.shape[1] != int(num_classes):
        raise ValueError(
            f"Output channels={raw_output.shape[1]} but num_classes={num_classes}"
        )

    evidence = F.softplus(raw_output)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(eps)
    prob = alpha / strength
    uncertainty = float(num_classes) / strength

    return {
        "evidence": evidence,
        "alpha": alpha,
        "strength": strength,
        "prob": prob,
        "uncertainty": uncertainty,
    }


def partial_ce_from_prob(
    prob: torch.Tensor,
    label: torch.Tensor,
    ignore_index: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if prob.ndim != 4:
        raise ValueError("prob must be [B, C, H, W]")

    if label.ndim != 3:
        raise ValueError("label must be [B, H, W]")

    valid_mask = label != int(ignore_index)

    if valid_mask.sum().item() == 0:
        return prob.new_zeros(())

    num_classes = prob.shape[1]
    safe_label = label.clamp(0, num_classes - 1).long()
    gt_prob = prob.gather(dim=1, index=safe_label.unsqueeze(1)).squeeze(1)
    loss_map = -torch.log(gt_prob.clamp_min(eps))
    return loss_map[valid_mask].mean()


def masked_soft_ce_from_prob(
    student_prob: torch.Tensor,
    target_prob: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_prob.shape != target_prob.shape:
        raise ValueError("student_prob and target_prob must have identical shapes")

    target_prob = target_prob.detach()
    log_prob = torch.log(student_prob.clamp_min(eps))
    ce_map = -(target_prob * log_prob).sum(dim=1, keepdim=True)

    if mask is None:
        return ce_map.mean()

    mask = mask.detach().float()

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [B, 1, H, W]")

    if mask.sum().item() < 1:
        return student_prob.new_zeros(())

    return (ce_map * mask).sum() / (mask.sum() + eps)


def asymmetric_uncertainty_mse_loss(
    student_uncertainty: torch.Tensor,
    teacher_uncertainty: torch.Tensor,
    reliable_mask: torch.Tensor | None,
    margin: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    if student_uncertainty.shape != teacher_uncertainty.shape:
        raise ValueError("Student and Teacher uncertainty shapes must match")

    if student_uncertainty.ndim != 4:
        raise ValueError("Uncertainty maps must be [B, 1, H, W]")

    if student_uncertainty.shape[1] != 1:
        raise ValueError("Uncertainty maps must have one channel")

    if margin < 0:
        raise ValueError("margin must be non-negative")

    teacher_target = teacher_uncertainty.detach()
    positive_gap = F.relu(student_uncertainty - teacher_target - float(margin))
    loss_map = positive_gap.pow(2)

    if reliable_mask is None:
        return loss_map.mean()

    mask = reliable_mask.detach().float()

    if mask.shape != student_uncertainty.shape:
        raise ValueError("reliable_mask must match [B, 1, H, W]")

    if mask.sum().item() < 1:
        return student_uncertainty.new_zeros(())

    return (loss_map * mask).sum() / (mask.sum() + eps)


def segmentation_prob_from_output(
    raw_output: torch.Tensor,
    num_classes: int,
    evidential: bool = True,
) -> torch.Tensor:
    raw_output = unpack_model_output(raw_output)
    if evidential:
        return evidential_prediction(raw_output, num_classes)["prob"]
    return torch.softmax(raw_output, dim=1)


def build_mt_confidence_pseudo_label(
    student_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    label: torch.Tensor,
    agree_thresh: float = 0.7,
    disagree_thresh: float = 0.8,
    margin_thresh: float = 0.1,
    ignore_index: int = 4,
    pseudo_mask_mode: str = "unlabeled",
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    student_prob = student_prob.detach()
    teacher_prob = teacher_prob.detach()

    conf_s, pred_s = torch.max(student_prob, dim=1)
    conf_t, pred_t = torch.max(teacher_prob, dim=1)

    if pseudo_mask_mode == "unlabeled":
        candidate_mask = label == ignore_index
    elif pseudo_mask_mode == "all":
        candidate_mask = torch.ones_like(label, dtype=torch.bool)
    else:
        raise ValueError(f"Unsupported pseudo_mask_mode: {pseudo_mask_mode}")

    same_pred = pred_s == pred_t
    diff_pred = ~same_pred

    min_conf = torch.minimum(conf_s, conf_t)
    max_conf = torch.maximum(conf_s, conf_t)
    margin = torch.abs(conf_s - conf_t)

    reliable_agree = same_pred & (min_conf >= agree_thresh) & candidate_mask
    reliable_disagree = (
        diff_pred
        & (max_conf >= disagree_thresh)
        & (margin >= margin_thresh)
        & candidate_mask
    )

    mean_pseudo = 0.5 * (student_prob + teacher_prob)
    choose_student = (conf_s > conf_t).unsqueeze(1)
    high_conf_pseudo = torch.where(choose_student, student_prob, teacher_prob)

    soft_pseudo_label = torch.where(
        reliable_disagree.unsqueeze(1),
        high_conf_pseudo,
        mean_pseudo,
    )
    soft_pseudo_label = soft_pseudo_label / (soft_pseudo_label.sum(dim=1, keepdim=True) + eps)

    reliable_mask = (reliable_agree | reliable_disagree).float().unsqueeze(1)
    pseudo_conf = torch.maximum(conf_s, conf_t).unsqueeze(1)

    return {
        "soft_pseudo_label": soft_pseudo_label.detach(),
        "reliable_mask": reliable_mask.detach(),
        "reliable_agree": reliable_agree.detach(),
        "reliable_disagree": reliable_disagree.detach(),
        "agreement_ratio": reliable_agree.float().mean().detach(),
        "disagreement_ratio": reliable_disagree.float().mean().detach(),
        "reliable_ratio": reliable_mask.mean().detach(),
        "pseudo_conf": pseudo_conf.detach(),
    }
