"""ModelMix: A New Model-Mixup Strategy to Minimize Vicinal Risk across
Tasks for Few-scribble based Cardiac Segmentation (Zhang & Patel, MICCAI
2024), verified against the official ``BWGZK/ModelMix`` repository.

ModelMix always trains a *pair* of tasks jointly: two encoder+decoder models
(same encoder architecture, no shared weights). Besides each task's own
scribble supervision, three extra mechanisms regularize the pair:

1. Image-level mixup (Eq. 1, ``L_inv``): each image is linearly blended with
   another sample from the same task (its batch-reversed counterpart --
   verified against the source, which does *not* implement the paper text's
   cutout-then-mix description), and the mixed image's own prediction is (a)
   supervised against the correspondingly blended one-hot label and (b)
   regularized to match the same blend of the two individual (detached)
   predictions.
2. Model-level mixup (Eq. 5, the method's namesake): one randomly selected
   *encoder* convolutional layer's weight (and bias) are linearly blended
   between the two tasks' own encoders; every other layer is untouched. That
   virtual encoder, paired with a task's own decoder, is supervised on that
   task's own (rotated, for a nontrivial invariance test) scribble label.
3. Vicinal regularization (Eq. 6): the virtual (mixed) model's prediction,
   rotated back to the original orientation, is regularized via negative
   cosine similarity against that task's own individual (non-mixed) model's
   prediction on the same (non-rotated) image.

This module holds the task-agnostic building blocks; ``train_modelmix_2d.py``
wires them into the joint ACDC+MSCMR training loop -- the only task pair in
this benchmark sharing a 4-class cardiac label space, matching the paper's
own primary experiment. WORD (abdominal CT) has no comparable "intrinsically
related" partner dataset here, so ModelMix is not run on it.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate


def sample_mix_ratio(batch_size, device, concentration=0.5):
    """``Beta(concentration, concentration)``-distributed mixing ratio,
    shaped ``[B, 1, 1, 1]`` for broadcasting against a batch of images."""
    ratio = np.random.beta(concentration, concentration, size=(batch_size, 1, 1, 1))
    return torch.from_numpy(ratio).float().to(device)


def one_hot_scribble(label, num_classes, ignore_index):
    """``[B, H, W]`` int64 scribble -> ``[B, C, H, W]`` one-hot float, zeroed
    across every channel wherever the label is ``ignore_index`` (no
    supervision contributed by an unannotated pixel, before or after
    mixing).
    """
    safe_label = torch.where(label == ignore_index, torch.zeros_like(label), label)
    one_hot = F.one_hot(safe_label, num_classes=num_classes).permute(0, 3, 1, 2).float()
    valid = (label != ignore_index).unsqueeze(1).float()
    return one_hot * valid


def soft_partial_cross_entropy(logits, soft_target):
    """Cross-entropy against a possibly-mixed soft one-hot target (Eq. 1's
    ``ce_mixed_loss``), normalized by the target's total annotation mass so
    a pixel only partially covered by the mix still contributes
    proportionally -- the soft-target counterpart of
    ``train.common_3d.partial_cross_entropy``'s "average per unit of
    supervision" convention.
    """
    log_prob = F.log_softmax(logits, dim=1)
    mass = soft_target.sum()
    if mass.item() == 0:
        return logits.sum() * 0.0
    return -(soft_target * log_prob).sum() / mass


def mix_invariance_loss(probs_mixed_image, probs, mix_ratio):
    """Eq. 1's ``L_inv``, called ``mix_consistency`` in the source: the mixed
    image's own prediction should match the same blend (its own batch-flip,
    with the same ``mix_ratio`` used to build the mixed image) of the two
    individual (detached) predictions. Unlike ``vicinal_regularization_loss``
    below, this uses ``1 - cos`` (verified against the source's
    ``mix_consistency`` term), not ``-cos``.

    Args:
        probs_mixed_image: softmax probabilities of the model's prediction
            on the *mixed* image.
        probs: softmax probabilities of the model's prediction on the
            original (unmixed) image.
        mix_ratio: the same ``[B, 1, 1, 1]`` ratio used to build the mixed
            image (see ``sample_mix_ratio``).
    """
    blended = mix_ratio * probs.detach() + (1.0 - mix_ratio) * torch.flip(probs.detach(), dims=[0])
    return 1.0 - F.cosine_similarity(probs_mixed_image.flatten(1), blended.flatten(1), dim=1).mean()


def vicinal_regularization_loss(mixed_probs_derotated, probs):
    """Eq. 6: negative cosine similarity between the de-rotated mixed
    (virtual) model's prediction and the individual model's own (detached)
    prediction on the same image -- verified against the source's
    ``invariant_loss_mix13`` (``-cos``, not ``1 - cos``).
    """
    return -F.cosine_similarity(mixed_probs_derotated.flatten(1), probs.detach().flatten(1), dim=1).mean()


def encoder_conv_layer_names(encoder):
    """Dotted names of every ``Conv2d``/``Conv3d`` submodule inside an
    encoder -- the eligible layers for model-mixup (Eq. 5). Matches the
    source's "any encoder parameter" pool (there, filtered by excluding the
    decoder's "Up"-named layers; here, trivially every layer already belongs
    to the encoder since it is a separate submodule from the decoder).
    """
    names = [
        name
        for name, module in encoder.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Conv3d))
    ]
    if not names:
        raise ValueError("encoder has no Conv2d/Conv3d submodules to mix")
    return names


def mix_one_encoder_layer(encoder_a, encoder_b, layer_name, mix_ratio):
    """Eq. 5: a virtual encoder -- a deep copy of ``encoder_a`` with just
    ``layer_name``'s weight (and bias, if present) replaced by
    ``mix_ratio*encoder_a + (1-mix_ratio)*encoder_b``'s value at that same
    layer. Every other layer keeps ``encoder_a``'s own parameters.

    Verified against the source's own ``backbone_mix = deepcopy(backbone1)``
    plus a raw ``.data =`` assignment for the mixed layer: the returned
    encoder is a disposable copy with no autograd connection back to
    ``encoder_a``/``encoder_b``'s real parameters (the blend is written
    under ``torch.no_grad()``, matching the source's ungraphed ``.data``
    write). This is intentional, not an oversight -- a training step that
    backprops through this virtual encoder's forward pass therefore never
    updates either encoder's real weights directly; only whichever *decoder*
    is called on top of it (a real, non-copied module) receives gradient,
    training the decoder to be robust to blended encoder features. The
    encoder itself is still updated every step, just through the *other*
    loss terms that call it directly (own supervision, image-level mixup).
    """
    mixed = copy.deepcopy(encoder_a)
    mixed_layer = mixed.get_submodule(layer_name)
    layer_a = encoder_a.get_submodule(layer_name)
    layer_b = encoder_b.get_submodule(layer_name)
    with torch.no_grad():
        mixed_layer.weight.copy_(mix_ratio * layer_a.weight + (1.0 - mix_ratio) * layer_b.weight)
        if mixed_layer.bias is not None:
            mixed_layer.bias.copy_(mix_ratio * layer_a.bias + (1.0 - mix_ratio) * layer_b.bias)
    return mixed


def random_rotate_image_and_label(image, label):
    """A single random angle in ``[0, 360)`` applied to both ``image`` and
    ``label`` (nearest-neighbor, label-safe), matching the source's per-
    iteration full-rotation augmentation for the model-mixup branch.
    """
    angle = float(np.random.uniform(0.0, 360.0))
    rotated_image = rotate(image, angle, interpolation=InterpolationMode.NEAREST)
    rotated_label = rotate(
        label.unsqueeze(1).float(), angle, interpolation=InterpolationMode.NEAREST
    ).squeeze(1).long()
    return rotated_image, rotated_label, angle


def rotate_back(tensor, angle):
    """Undo ``random_rotate_image_and_label``'s rotation on a prediction map
    (nearest-neighbor, matching the source's de-rotation of the mixed
    model's output before comparing it to the individual model's output).
    """
    return rotate(tensor, -angle, interpolation=InterpolationMode.NEAREST)
