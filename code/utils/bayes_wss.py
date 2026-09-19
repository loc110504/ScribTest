"""Losses for Bayes-WSS (Zheng et al., MICCAI 2024, ``A Bayesian Approach to
Weakly-supervised Laparoscopic Image Segmentation``), verified against the
official ``MoriLabNU/Bayesian_WSS`` source (``AutoLaparo/train.py``).

The method has two training stages, both reimplemented here:

Stage 1 -- learning ``p(x,y|z)`` (``bayes_wss_cvae_step``): the CVAE
(``networks/bayes_wss_2d.BayesCVAE2D``) is trained to jointly maximize the
ELBO (paper Eq. 6) via

    L = L_pCE(mean_softmax(y), s) + L_crf(mean_softmax(y)) + beta*L_recon + alpha*L_kl

exactly the official loss composition (``loss = loss_pce + loss_crf +
args.recon * loss_recon + args.kl * loss_kl``), with ``y`` sampled
``sample_time`` times and softmax-averaged before both ``L_pCE`` and
``L_crf`` (Eq. 4's Monte Carlo estimate of ``p(y|x,z)``).

Stage 2 -- learning ``p(w|x,y)`` (``merge_pseudo_labels`` +
plain cross-entropy on a *plain* ``UNet2D``, done directly in
``train/train_bayes_wss_2d.py``): the frozen stage-1 CVAE's averaged,
argmax'd prediction fills in every scribble-unlabeled pixel (``mask``);
scribble-annotated pixels are kept as-is. The resulting fully-dense merged
label supervises a plain UNet2D with ordinary cross-entropy -- this is the
network Bayes-WSS actually deploys (``BDL_MC_UNet`` in the official source,
architecturally identical to this repository's ``UNet2D``), so
``test_pce_2d.py`` evaluates a Bayes-WSS checkpoint directly, exactly like
pCE/CycleMix/EFFDNet/SDT-Net/ModelMix.

Deviation from the official source, clearly flagged: the official DenseCRF
loss (``utils/crfloss/pytorch_deeplab_v3_plus/DenseCRFLoss.py``) calls a
compiled SWIG/C++ permutohedral-lattice bilateral filter
(``wrapper/bilateralfilter``) with an effectively unbounded receptive field.
That native extension is not part of this repository's pure-PyTorch,
CPU-testable stack (no other method here needs a compiled dependency), so
``local_dense_crf_loss`` instead computes the same functional form -- Eq. 8,
``sum_c y_c^T K (1 - y_c)`` with a Gaussian RGB+XY bilateral kernel -- over a
bounded local neighborhood via ``unfold``, fully differentiable in pure
PyTorch. This trades the permutohedral lattice's long-range support for a
finite window (``--crf_radius``); the RGB bandwidth (``--crf_sigma_rgb``)
keeps the paper's default (15, over a 0-255 intensity scale), while the XY
bandwidth (``--crf_sigma_xy``) is scaled down to match the bounded window
rather than the paper's near-global 100 (after its own 0.5 downsampling),
since a large sigma with a small window would just degrade to an unweighted
box average.
"""

import torch
import torch.nn.functional as F


def bayes_kl_loss(mu, log_var):
    """Paper Eq. 6 KL term: ``KL[q(z|x) || N(0, I)]``, matching the official
    ``torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)``."""
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1), dim=0)


def bayes_reconstruction_loss(x_gen, image):
    """Paper Eq. 6 reconstruction term: MSE between ``p(x|z)``'s mean
    reconstruction (averaged over the sample dimension, matching the
    official ``x_gen = torch.mean(x_gen, dim=0)``) and the input image."""
    if x_gen.ndim != 5:
        raise ValueError("x_gen must be [sample_time, B, C, H, W], got ndim={}".format(x_gen.ndim))
    return F.mse_loss(x_gen.mean(dim=0), image)


def bayes_mean_softmax(y_logits):
    """Mean-of-softmax over the sample dimension (Eq. 4's MC estimate of
    ``p(y|x,z)``), matching the official ``y = mean(softmax(y, dim=2), dim=0)``."""
    if y_logits.ndim != 5:
        raise ValueError("y_logits must be [sample_time, B, C, H, W], got ndim={}".format(y_logits.ndim))
    return F.softmax(y_logits, dim=2).mean(dim=0)


def bayes_pce_loss(mean_probs, target, ignore_index):
    """Partial NLL loss on the sample-averaged probability map, matching the
    official ``F.nll_loss(torch.log(y + 1e-12), label_batch, ignore_index=...)``."""
    if mean_probs.shape[0] != target.shape[0] or mean_probs.shape[2:] != target.shape[1:]:
        raise ValueError("mean_probs [B,C,H,W] and target [B,H,W] shapes are incompatible")
    log_probs = torch.log(mean_probs.clamp_min(1e-12))
    return F.nll_loss(log_probs, target, ignore_index=ignore_index)


def _spatial_offsets(radius, device, dtype):
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dy, dx = torch.meshgrid(coords, coords, indexing="ij")
    return (dy.pow(2) + dx.pow(2)).reshape(-1)  # [K*K]


def local_dense_crf_loss(image, probs, weight, sigma_rgb=15.0, sigma_xy=5.0, radius=5, intensity_scale=255.0):
    """Bounded-window approximation of the official permutohedral-lattice
    DenseCRF loss (Eq. 8); see this module's docstring for the deviation.

    Args:
        image: ``[B, 1, H, W]`` grayscale image in ``[0, 1]``.
        probs: ``[B, C, H, W]`` softmax segmentation probabilities.
        weight: scalar multiplier (paper's ``gamma``).
        sigma_rgb/sigma_xy: Gaussian bandwidths over intensity (scaled by
            ``intensity_scale``, matching the official ``* 255.0``) and pixel
            offset, respectively.
        radius: half-width of the local neighborhood window (``2r+1`` square).
    """
    if image.ndim != 4 or image.shape[1] != 1:
        raise ValueError("image must be [B, 1, H, W]")
    if probs.ndim != 4 or probs.shape[0] != image.shape[0] or probs.shape[2:] != image.shape[2:]:
        raise ValueError("probs must be [B, C, H, W] matching image's batch/spatial size")
    batch, num_classes, height, width = probs.shape
    kernel = 2 * radius + 1
    num_pixels = height * width
    scaled_image = image * intensity_scale

    image_patches = F.unfold(scaled_image, kernel_size=kernel, padding=radius).view(batch, kernel * kernel, num_pixels)
    center_intensity = scaled_image.view(batch, 1, num_pixels)
    intensity_sq_diff = (image_patches - center_intensity).pow(2)

    spatial_sq_dist = _spatial_offsets(radius, image.device, image.dtype)  # [K*K]
    weights = torch.exp(
        -spatial_sq_dist.view(1, -1, 1) / (2.0 * sigma_xy ** 2) - intensity_sq_diff / (2.0 * sigma_rgb ** 2)
    )
    center_offset = (kernel * kernel) // 2
    weights[:, center_offset, :] = 0.0  # exclude the self-pairing term

    probs_patches = F.unfold(probs, kernel_size=kernel, padding=radius).view(
        batch, num_classes, kernel * kernel, num_pixels
    )
    center_probs = probs.view(batch, num_classes, 1, num_pixels)
    pairwise = weights.unsqueeze(1) * center_probs * (1.0 - probs_patches)
    loss = pairwise.sum(dim=2).sum(dim=1).mean()
    return weight * loss


def merge_pseudo_labels(scribble, pseudo_argmax, ignore_index):
    """Stage-2 label fusion (paper Sec. 2.2's ``y = (1-mask)*s + mask*Y``):
    keeps the scribble wherever annotated, fills in the CVAE's pseudo-label
    argmax elsewhere. Returns a fully-dense label with no ``ignore_index``."""
    if scribble.shape != pseudo_argmax.shape:
        raise ValueError("scribble and pseudo_argmax must have matching shapes")
    unlabeled = scribble == ignore_index
    return torch.where(unlabeled, pseudo_argmax, scribble)


def bayes_wss_cvae_step(cvae, image, target, ignore_index, args):
    """One stage-1 (``p(x,y|z)``) forward pass; returns loss and components."""
    mu, log_var, x_gen, y_logits = cvae(image, sample_time=args.sample_time)
    loss_kl = bayes_kl_loss(mu, log_var)
    loss_recon = bayes_reconstruction_loss(x_gen, image)
    mean_probs = bayes_mean_softmax(y_logits)
    loss_pce = bayes_pce_loss(mean_probs, target, ignore_index)
    loss_crf = local_dense_crf_loss(
        image, mean_probs, weight=args.crf, sigma_rgb=args.crf_sigma_rgb, sigma_xy=args.crf_sigma_xy,
        radius=args.crf_radius,
    )
    total = loss_pce + loss_crf + args.recon * loss_recon + args.kl * loss_kl
    components = {
        "pce": loss_pce.item(),
        "crf": loss_crf.item(),
        "recon": loss_recon.item(),
        "kl": loss_kl.item(),
    }
    return total, components


__all__ = [
    "bayes_kl_loss",
    "bayes_reconstruction_loss",
    "bayes_mean_softmax",
    "bayes_pce_loss",
    "local_dense_crf_loss",
    "merge_pseudo_labels",
    "bayes_wss_cvae_step",
]
