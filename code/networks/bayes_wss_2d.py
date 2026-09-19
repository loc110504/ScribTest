"""Training-time Bayesian dual-encoder/decoder CVAE for Bayes-WSS (Zheng et
al., MICCAI 2024, ``A Bayesian Approach to Weakly-supervised Laparoscopic
Image Segmentation``), verified against the official ``MoriLabNU/Bayesian_WSS``
source (``networks/Bayesian_UNet.py``'s ``BDL_L_UNet``).

The paper models the joint distribution ``p(x, y | z)`` with a conditional
VAE: encoder ``e1``/decoder ``d1`` reconstruct the image from a sampled
latent ``z`` (``p(x|z)``), while a second encoder ``e2`` and decoder ``d2``
consume the same ``z`` concatenated onto its own bottleneck features to
predict the segmentation (``p(y|x,z)``). Both encoders/decoders reuse this
repository's ``Encoder2D``/``Decoder2D`` (``networks/unet_2d.py``) building
blocks -- the official source's own ``Encoder``/``Decoder`` are themselves
"borrowed from HiLab-git/PyMIC", the same lineage as ``unet_2d.py`` -- rather
than duplicating them.

Unlike DMSPS/DMPLS's dual-decoder network (needed at both train and test
time), this CVAE is training-only: only its segmentation branch's *targets*
(the sampled, softmax-averaged ``p(y|x,z)``) are used, to supervise a
second, plain ``UNet2D`` (the official source's ``BDL_MC_UNet``, which is
architecturally identical to this repository's ``UNet2D`` -- same
``Encoder2D``/``Decoder2D`` with feature_chns ``[16,32,64,128,256]``,
dropout schedule ``[0.05,0.1,0.2,0.3,0.5]``, transpose-conv upsampling). That
plain ``UNet2D`` is what gets deployed and checkpointed by
``train/train_bayes_wss_2d.py``, exactly like every other single-network
method in this benchmark (evaluated with the shared ``test_pce_2d.py``).

The official ``fc_mu``/``fc_var``/``transform`` linear layers hardcode the
flattened bottleneck size for AutoLaparo's ``240x480`` inputs
(``256 * 15 * 30``, i.e. ``256 * (240/16) * (480/16)``). This module instead
derives that size from ``patch_size`` (ACDC/MSCMR resize to ``256x256``, so
the bottleneck is ``256 * 16 * 16``), which is mathematically the same
computation with a different input resolution, not a behavior change.
"""

import torch
import torch.nn as nn

from networks.unet_2d import DEFAULT_DROPOUT, DEFAULT_FEATURE_CHNS, Decoder2D, Encoder2D


class BayesCVAE2D(nn.Module):
    """``BDL_L_UNet`` equivalent: models ``p(x|z)`` and ``p(y|x,z)`` jointly."""

    def __init__(
        self,
        in_chns,
        class_num,
        patch_size,
        feature_chns=DEFAULT_FEATURE_CHNS,
        dropout=DEFAULT_DROPOUT,
        latent_dim=256,
    ):
        super().__init__()
        if len(feature_chns) != 5:
            raise ValueError("BayesCVAE2D expects 5 feature_chns stages")
        patch_h, patch_w = (int(value) for value in patch_size)
        if patch_h % 16 or patch_w % 16:
            raise ValueError("patch_size must be divisible by 16 (4 encoder maxpools)")
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.latent_dim = int(latent_dim)
        self.bottleneck_shape = (feature_chns[4], patch_h // 16, patch_w // 16)
        flat_dim = feature_chns[4] * (patch_h // 16) * (patch_w // 16)

        # p(x|z): reconstruction branch.
        self.encoder1 = Encoder2D(self.in_chns, feature_chns, dropout)
        self.decoder1 = Decoder2D(feature_chns, class_num=self.in_chns)
        self.fc_mu = nn.Linear(flat_dim, self.latent_dim)
        self.fc_var = nn.Linear(flat_dim, self.latent_dim)
        self.transform = nn.Linear(self.latent_dim, flat_dim)

        # p(y|x,z): segmentation branch, bottleneck concatenated with z_map
        # (both feature_chns[4]-wide -- z_map is `transform(z)` reshaped back
        # to the bottleneck's spatial/channel shape, not latent_dim-wide).
        self.encoder2 = Encoder2D(self.in_chns, feature_chns, dropout)
        concat_feature_chns = tuple(feature_chns[:4]) + (feature_chns[4] * 2,)
        self.decoder2 = Decoder2D(concat_feature_chns, class_num=self.class_num)

    def reparameterize(self, mu, logvar, sample_time):
        batch, dim = mu.shape
        std = torch.exp(0.5 * logvar)
        if sample_time == 1:
            eps = torch.randn_like(std)
            return eps * std + mu
        samples = []
        for _ in range(sample_time):
            eps = torch.randn_like(std)
            samples.append((eps * std + mu).unsqueeze(0))
        return torch.cat(samples, dim=0).view(batch * sample_time, dim)

    def forward(self, x, sample_time=1):
        if x.ndim != 4 or x.shape[1] != self.in_chns:
            raise ValueError("BayesCVAE2D expects [B, {}, H, W], got {}".format(self.in_chns, tuple(x.shape)))
        batch, _, height, width = x.shape

        features1 = self.encoder1(x)
        flat = torch.flatten(features1[4], start_dim=1)
        mu = self.fc_mu(flat)
        log_var = self.fc_var(flat)
        z = self.reparameterize(mu, log_var, sample_time)
        z_map = self.transform(z).view(-1, *self.bottleneck_shape)

        recon_features = [feature.repeat(sample_time, 1, 1, 1) for feature in features1[:4]] + [z_map]
        x_gen = torch.sigmoid(self.decoder1(recon_features))
        x_gen = x_gen.view(sample_time, batch, self.in_chns, height, width)

        features2 = self.encoder2(x)
        seg_features = [feature.repeat(sample_time, 1, 1, 1) for feature in features2[:4]]
        bottleneck2 = features2[4].repeat(sample_time, 1, 1, 1)
        seg_features.append(torch.cat([z_map, bottleneck2], dim=1))
        y_logits = self.decoder2(seg_features)
        y_logits = y_logits.view(sample_time, batch, self.class_num, height, width)

        return mu, log_var, x_gen, y_logits


__all__ = ["BayesCVAE2D"]
