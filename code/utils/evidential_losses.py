import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def evidence_from_logits(logits, activation="relu"):
    """
    Convert raw model logits to non-negative evidence.

    Args:
        logits: Tensor [B, C, H, W]
        activation: one of ["relu", "softplus", "exp"]

    Returns:
        evidence: Tensor [B, C, H, W]
    """
    if activation == "relu":
        evidence = F.relu(logits)
    elif activation == "softplus":
        evidence = F.softplus(logits)
    elif activation == "exp":
        evidence = torch.exp(torch.clamp(logits, min=-10.0, max=10.0))
    else:
        raise ValueError(f"Unsupported evidence activation: {activation}")

    evidence = torch.nan_to_num(evidence, nan=0.0, posinf=1e4, neginf=0.0)
    evidence = torch.clamp(evidence, min=0.0, max=1e4)
    return evidence


def dirichlet_params_from_logits(logits, activation="relu", eps=1e-6):
    """
    Build evidence, Dirichlet parameters, posterior probability,
    uncertainty map and belief mass from segmentation logits.

    Args:
        logits: Tensor [B, C, H, W]
        activation: evidence activation
        eps: numerical stability value

    Returns:
        evidence: Tensor [B, C, H, W]
        alpha: Tensor [B, C, H, W]
        prob: Tensor [B, C, H, W]
        uncertainty: Tensor [B, 1, H, W]
        belief: Tensor [B, C, H, W]
    """
    num_classes = logits.shape[1]

    evidence = evidence_from_logits(logits, activation=activation)

    alpha = evidence + 1.0
    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0)
    alpha = torch.clamp(alpha, min=1.0 + eps, max=1e4)

    s = alpha.sum(dim=1, keepdim=True)
    s = torch.nan_to_num(s, nan=float(num_classes), posinf=1e4, neginf=float(num_classes))
    s = torch.clamp(s, min=float(num_classes) * (1.0 + eps), max=1e4)

    prob = alpha / s
    prob = torch.nan_to_num(prob, nan=1.0 / float(num_classes), posinf=1.0, neginf=0.0)
    prob = torch.clamp(prob, min=eps, max=1.0)

    uncertainty = float(num_classes) / s
    uncertainty = torch.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=eps)
    uncertainty = torch.clamp(uncertainty, min=eps, max=1.0 - eps)

    belief = evidence / s
    belief = torch.nan_to_num(belief, nan=0.0, posinf=1.0, neginf=0.0)
    belief = torch.clamp(belief, min=0.0, max=1.0)

    return evidence, alpha, prob, uncertainty, belief


def one_hot_target(target, num_classes, ignore_index=None):
    """
    Convert target [B, H, W] to one-hot [B, C, H, W].

    Ignored pixels become all-zero vectors.

    Args:
        target: Tensor [B, H, W]
        num_classes: int
        ignore_index: optional int

    Returns:
        y_onehot: Tensor [B, C, H, W]
        valid_mask: Tensor [B, 1, H, W]
    """
    if ignore_index is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
        safe_target = target
    else:
        valid_mask = target != ignore_index
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

    safe_target = safe_target.clamp(min=0, max=num_classes - 1)

    y_onehot = F.one_hot(safe_target.long(), num_classes=num_classes)
    y_onehot = y_onehot.permute(0, 3, 1, 2).float()

    valid_mask = valid_mask.unsqueeze(1).float()
    y_onehot = y_onehot * valid_mask

    return y_onehot, valid_mask


def evidential_ce_loss(alpha, target, num_classes, ignore_index=None, eps=1e-6):
    """
    Evidential cross entropy:

    LCE = sum_c y_c * (digamma(S) - digamma(alpha_c))

    Args:
        alpha: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        num_classes: int
        ignore_index: optional int

    Returns:
        scalar Tensor
    """
    y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0)
    alpha = torch.clamp(alpha, min=1.0 + eps, max=1e4)

    s = alpha.sum(dim=1, keepdim=True)
    s = torch.clamp(s, min=float(num_classes) * (1.0 + eps), max=1e4)

    loss_map = torch.sum(
        y_onehot * (torch.digamma(s) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )

    loss_map = torch.nan_to_num(loss_map, nan=0.0, posinf=1e4, neginf=0.0)

    loss = (loss_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

    return loss


def evidential_dice_loss(
    prob,
    target,
    num_classes,
    ignore_index=None,
    smooth=1e-5,
    include_background=False,
):
    """
    Soft Dice loss using posterior probability prob = alpha / S.

    Args:
        prob: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        num_classes: int
        include_background:
            False -> compute class 1..C-1
            True  -> compute class 0..C-1

    Returns:
        scalar Tensor
    """
    y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

    prob = torch.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = torch.clamp(prob, min=0.0, max=1.0)

    if include_background:
        class_ids = range(num_classes)
    else:
        class_ids = range(1, num_classes)

    dice_losses = []

    for c in class_ids:
        p = prob[:, c:c + 1] * valid_mask
        y = y_onehot[:, c:c + 1] * valid_mask

        intersection = (p * y).sum()
        denominator = p.sum() + y.sum()

        dice = (2.0 * intersection + smooth) / (denominator + smooth)
        dice = torch.nan_to_num(dice, nan=0.0, posinf=1.0, neginf=0.0)
        dice = torch.clamp(dice, min=0.0, max=1.0)

        dice_losses.append(1.0 - dice)

    if len(dice_losses) == 0:
        return prob.new_tensor(0.0)

    loss = torch.stack(dice_losses).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1.0, neginf=0.0)

    return loss


def evidential_kl_loss(
    alpha,
    target,
    num_classes,
    ignore_index=None,
    annealing_coef=1.0,
    eps=1e-6,
):
    """
    KL[D(p|alpha_tilde) || D(p|1)]

    alpha_tilde = y + (1 - y) * alpha

    This suppresses evidence for incorrect classes.

    Args:
        alpha: Tensor [B, C, H, W]
        target: Tensor [B, H, W]
        num_classes: int
        ignore_index: optional int
        annealing_coef: float

    Returns:
        scalar Tensor
    """
    y_onehot, valid_mask = one_hot_target(target, num_classes, ignore_index)

    alpha = torch.nan_to_num(alpha, nan=1.0, posinf=1e4, neginf=1.0)
    alpha = torch.clamp(alpha, min=1.0 + eps, max=1e4)

    alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha

    # Important:
    # For ignored pixels, do not let lgamma/digamma run on arbitrary alpha.
    # Set alpha_tilde to uniform Dirichlet parameters, then mask them out.
    alpha_tilde = torch.where(
        valid_mask.bool(),
        alpha_tilde,
        torch.ones_like(alpha_tilde),
    )

    alpha_tilde = torch.nan_to_num(alpha_tilde, nan=1.0, posinf=1e4, neginf=1.0)
    alpha_tilde = torch.clamp(alpha_tilde, min=1.0 + eps, max=1e4)

    s_alpha = alpha_tilde.sum(dim=1, keepdim=True)
    s_alpha = torch.clamp(s_alpha, min=float(num_classes) * (1.0 + eps), max=1e4)

    ln_b = torch.lgamma(s_alpha) - torch.lgamma(alpha_tilde).sum(dim=1, keepdim=True)

    ln_b_uni = torch.lgamma(
        torch.tensor(float(num_classes), device=alpha.device, dtype=alpha.dtype)
    )

    digamma_sum = torch.digamma(s_alpha)
    digamma_alpha = torch.digamma(alpha_tilde)

    kl_map = ln_b - ln_b_uni + (
        (alpha_tilde - 1.0) * (digamma_alpha - digamma_sum)
    ).sum(dim=1, keepdim=True)

    kl_map = torch.nan_to_num(kl_map, nan=0.0, posinf=1e4, neginf=0.0)

    kl = (kl_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    kl = torch.nan_to_num(kl, nan=0.0, posinf=1e4, neginf=0.0)

    return float(annealing_coef) * kl


def calibrated_evidential_uncertainty_loss(
    logits,
    alpha,
    belief,
    uncertainty,
    target,
    num_classes,
    epoch,
    max_epoch,
    alpha0=1.0,
    ignore_index=None,
    eps=1e-6,
):
    """
    Calibrated Evidential Uncertainty loss.

    Correct pixels should become certain.
    Incorrect pixels should become uncertain.

    Args:
        logits: Tensor [B, C, H, W]
        alpha: Tensor [B, C, H, W]
        belief: Tensor [B, C, H, W]
        uncertainty: Tensor [B, 1, H, W]
        target: Tensor [B, H, W]
        num_classes: int
        epoch: current epoch
        max_epoch: total epoch count

    Returns:
        scalar Tensor
    """
    _, valid_mask = one_hot_target(target, num_classes, ignore_index)

    pred = torch.argmax(logits, dim=1)

    if ignore_index is None:
        valid_bool = torch.ones_like(target, dtype=torch.bool)
    else:
        valid_bool = target != ignore_index

    correct = ((pred == target) & valid_bool).unsqueeze(1)
    incorrect = ((pred != target) & valid_bool).unsqueeze(1)

    alpha_t = alpha0 * math.exp(-float(epoch) / max(float(max_epoch), 1.0))

    # Keep both correct and incorrect branches active.
    alpha_t = min(max(alpha_t, 0.05), 0.95)

    u = torch.nan_to_num(uncertainty, nan=1.0, posinf=1.0, neginf=eps)
    u = torch.clamp(u, min=eps, max=1.0 - eps)

    belief_sum = belief.sum(dim=1, keepdim=True)
    belief_sum = torch.nan_to_num(belief_sum, nan=0.0, posinf=1.0, neginf=0.0)
    belief_sum = torch.clamp(belief_sum, min=0.0, max=1.0)

    loss_correct = -alpha_t * belief_sum * torch.log(1.0 - u)
    loss_incorrect = -(1.0 - alpha_t) * (1.0 - belief_sum) * torch.log(u)

    loss_correct = torch.nan_to_num(loss_correct, nan=0.0, posinf=1e4, neginf=0.0)
    loss_incorrect = torch.nan_to_num(loss_incorrect, nan=0.0, posinf=1e4, neginf=0.0)

    loss_map = torch.zeros_like(u)
    loss_map = loss_map + torch.where(
        correct,
        loss_correct,
        torch.zeros_like(loss_correct),
    )
    loss_map = loss_map + torch.where(
        incorrect,
        loss_incorrect,
        torch.zeros_like(loss_incorrect),
    )

    loss_map = torch.nan_to_num(loss_map, nan=0.0, posinf=1e4, neginf=0.0)

    loss = (loss_map * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)
    loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)

    return loss


class EvidentialSegmentationLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_kl=0.2,
        lambda_dice=1.0,
        ignore_index=None,
        activation="relu",
        alpha0=1.0,
        kl_annealing_epochs=50,
        include_background=False,
        debug=False,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.lambda_dice = lambda_dice
        self.ignore_index = ignore_index
        self.activation = activation
        self.alpha0 = alpha0
        self.kl_annealing_epochs = kl_annealing_epochs
        self.include_background = include_background
        self.debug = debug

    def _check_finite(self, name, value, logits=None, alpha=None, uncertainty=None):
        if not self.debug:
            return

        if not torch.is_tensor(value):
            return

        if torch.isfinite(value).all():
            return

        print(f"[NaN/Inf detected] {name}")

        if logits is not None:
            print(
                "logits:",
                "min=", torch.nan_to_num(logits.detach()).min().item(),
                "max=", torch.nan_to_num(logits.detach()).max().item(),
                "mean=", torch.nan_to_num(logits.detach()).mean().item(),
            )

        if alpha is not None:
            print(
                "alpha:",
                "min=", torch.nan_to_num(alpha.detach()).min().item(),
                "max=", torch.nan_to_num(alpha.detach()).max().item(),
                "mean=", torch.nan_to_num(alpha.detach()).mean().item(),
            )

        if uncertainty is not None:
            print(
                "uncertainty:",
                "min=", torch.nan_to_num(uncertainty.detach()).min().item(),
                "max=", torch.nan_to_num(uncertainty.detach()).max().item(),
                "mean=", torch.nan_to_num(uncertainty.detach()).mean().item(),
            )

        raise FloatingPointError(f"{name} is not finite")

    def forward(self, logits, target, epoch, max_epoch):
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        logits = torch.clamp(logits, min=-20.0, max=20.0)

        evidence, alpha, prob, uncertainty, belief = dirichlet_params_from_logits(
            logits,
            activation=self.activation,
        )

        loss_ce = evidential_ce_loss(
            alpha=alpha,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        loss_ceu = calibrated_evidential_uncertainty_loss(
            logits=logits,
            alpha=alpha,
            belief=belief,
            uncertainty=uncertainty,
            target=target,
            num_classes=self.num_classes,
            epoch=epoch,
            max_epoch=max_epoch,
            alpha0=self.alpha0,
            ignore_index=self.ignore_index,
        )

        annealing_coef = min(
            1.0,
            float(epoch + 1) / float(max(self.kl_annealing_epochs, 1)),
        )

        loss_kl = evidential_kl_loss(
            alpha=alpha,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            annealing_coef=annealing_coef,
        )

        loss_dice = evidential_dice_loss(
            prob=prob,
            target=target,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
        )

        self._check_finite("loss_ce", loss_ce, logits, alpha, uncertainty)
        self._check_finite("loss_ceu", loss_ceu, logits, alpha, uncertainty)
        self._check_finite("loss_kl", loss_kl, logits, alpha, uncertainty)
        self._check_finite("loss_dice", loss_dice, logits, alpha, uncertainty)

        total_loss = (
            loss_ce
            + loss_ceu
            + self.lambda_kl * loss_kl
            + self.lambda_dice * loss_dice
        )

        self._check_finite("total_loss", total_loss, logits, alpha, uncertainty)

        # Do not silently hide instability in debug mode.
        # In normal mode, avoid TensorBoard crash from rare non-finite values.
        if not self.debug:
            total_loss = torch.nan_to_num(total_loss, nan=0.0, posinf=1e4, neginf=0.0)

        loss_dict = {
            "loss_total": total_loss.detach(),
            "loss_ece": loss_ce.detach(),
            "loss_ceu": loss_ceu.detach(),
            "loss_kl": loss_kl.detach(),
            "loss_dice": loss_dice.detach(),
            "uncertainty_mean": uncertainty.detach().mean(),
            "evidence_mean": evidence.detach().mean(),
        }

        aux_dict = {
            "prob": prob.detach(),
            "uncertainty": uncertainty.detach(),
            "evidence": evidence.detach(),
            "alpha": alpha.detach(),
        }

        return total_loss, loss_dict, aux_dict