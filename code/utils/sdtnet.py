"""3D building blocks for SDT-Net (Nguyen et al. 2026, "Scribble-Supervised
Medical Image Segmentation with Dynamic Teacher Switching and Hierarchical
Consistency").

Reimplements the dual-teacher/single-student framework -- Dynamic Teacher
Switching (DTS), Pick Reliable Pixels (PRP) pseudo-labeling, and Hierarchical
Consistency (HiCo) feature alignment -- for volumetric patches, ported from
this repository's own 2D reference implementation
(``code/train/train_sdtnet_2d.py``), which is treated as the ground truth for
method details the paper leaves ambiguous.

Two behaviors were deliberately NOT copied from the 2D reference, because
they are implementation defects rather than part of the method the paper
describes (verified empirically, not just by reading):

1. ``utils/ema_optim.py``'s ``WeightEMA.__init__`` copies the *teacher's*
   initial weights onto the *student* (``param.data.copy_(ema_param.data)``,
   with ``param`` the student's tensor). Called twice in the 2D script (once
   per teacher), this silently discards the student's own initialization in
   favor of teacher 2's, which contradicts the framework's premise of three
   independently initialized networks. :class:`TeacherEMA` below reproduces
   the ``.step()`` update exactly but does not reproduce this constructor
   side effect.
2. ``utils/losses.py``'s ``pDLoss``, called as in the 2D script with an
   ``ignore_index`` and an externally unsqueezed integer target, mixes
   *different batch elements* into the same intersection/union sum: its
   ``ignore_mask`` keeps its pre-one-hot shape ``[B, 1, ...]`` while the
   per-class score/target tensors are ``[B, ...]``, and multiplying them
   broadcasts to ``[B, B, ...]`` instead of ``[B, ...]`` (confirmed by
   printing the intermediate shape). :func:`soft_dice_loss` below is a
   corrected, shape-safe replacement with the same "mean soft Dice over all
   classes, ignored voxels excluded" semantics the paper describes.
"""

import torch
import torch.nn.functional as F


def pick_reliable_pixels(probs, threshold, ignore_index):
    """Eq. 5 (Pick Reliable Pixels): keep a pixel's argmax class only if its
    own probability strictly exceeds ``threshold``; otherwise mark it
    ``ignore_index``. Matches ``utils/pick_reliable_pixels.py``'s strict
    ``>`` comparison (not the paper text's ``>=``) and generalizes it from a
    hardcoded 4-class map to any number of classes.

    Args:
        probs: ``[B, C, D, H, W]`` softmax probabilities.
    Returns:
        ``[B, D, H, W]`` integer pseudo-label.
    """
    confidence, predicted = probs.max(dim=1)
    return torch.where(confidence > threshold, predicted, torch.full_like(predicted, ignore_index))


def feature_consistency_loss(student_feature, teacher_feature):
    """Eq. 7: ``L_feat = 0.5 * (L1(FS, FT) + (1 - cos(FS, FT)))``.

    Cosine similarity is computed on each sample's fully flattened feature
    vector (matching ``F.cosine_similarity(f.flatten(1), ...)`` in the 2D
    reference), then averaged over the batch.
    """
    if student_feature.shape != teacher_feature.shape:
        raise ValueError(
            "feature_consistency_loss expects matching shapes, got {} and {}".format(
                tuple(student_feature.shape), tuple(teacher_feature.shape)
            )
        )
    l1 = F.l1_loss(student_feature, teacher_feature)
    cosine = F.cosine_similarity(student_feature.flatten(1), teacher_feature.flatten(1), dim=1).mean()
    return 0.5 * (l1 + (1.0 - cosine))


def soft_dice_loss(probs, target, num_classes, ignore_index):
    """Mean soft Dice loss (Eq. 6's ``L_Dice`` term) over all ``num_classes``
    (background included, matching ``pDLoss(num_classes, ...)`` in the 2D
    reference), restricted to voxels where ``target != ignore_index``.

    Args:
        probs: ``[B, C, D, H, W]`` (3D) or ``[B, C, H, W]`` (2D ACDC/MSCMR
            slices) softmax probabilities.
        target: ``[B, D, H, W]`` or ``[B, H, W]`` integer labels, matching
            ``probs``'s spatial rank (may contain ``ignore_index``).
    """
    if probs.ndim not in (4, 5) or target.shape != probs.shape[:1] + probs.shape[2:]:
        raise ValueError(
            "Expected probs [B,C,H,W] or [B,C,D,H,W] and a matching target, got {} and {}".format(
                tuple(probs.shape), tuple(target.shape)
            )
        )
    spatial_ndim = probs.ndim - 2
    smooth = 1e-5
    valid = (target != ignore_index).float()
    safe_target = torch.where(target == ignore_index, torch.zeros_like(target), target)
    permute_order = (0, spatial_ndim + 1) + tuple(range(1, spatial_ndim + 1))
    one_hot = F.one_hot(safe_target.long(), num_classes=num_classes).permute(*permute_order).float()

    losses = []
    for class_id in range(num_classes):
        score = probs[:, class_id] * valid
        truth = one_hot[:, class_id] * valid
        intersect = (score * truth).sum()
        denom = (score * score).sum() + (truth * truth).sum()
        losses.append(1.0 - (2.0 * intersect + smooth) / (denom + smooth))
    return torch.stack(losses).mean()


class TeacherEMA:
    """Exponential moving average of a teacher's weights toward the student.

    Reproduces ``utils/ema_optim.py``'s ``WeightEMA.step()`` exactly --
    ``teacher = alpha*teacher + (1-alpha)*student``, plus its "customized
    weight decay" side effect that also shrinks the student's own parameters
    on every step -- without the constructor's teacher-to-student copy (see
    the module docstring for why that part is not reproduced).
    """

    def __init__(self, student, teacher, alpha=0.99, student_weight_decay=0.02 * 0.01):
        self.student_params = list(student.state_dict().values())
        self.teacher_params = list(teacher.state_dict().values())
        self.alpha = alpha
        self.student_weight_decay = student_weight_decay

    @torch.no_grad()
    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for student_param, teacher_param in zip(self.student_params, self.teacher_params):
            if teacher_param.dtype == torch.float32:
                teacher_param.mul_(self.alpha)
                teacher_param.add_(student_param * one_minus_alpha)
                student_param.mul_(1.0 - self.student_weight_decay)
