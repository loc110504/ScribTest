"""3D U-Net with Cross-Consistency Training auxiliary decoders."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_3d import (
    DEFAULT_STRIDES,
    Decoder3D,
    Encoder3D,
    initialize_3d_weights,
)


def feature_dropout_3d(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    maximum = attention.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1)
    threshold = maximum * float(np.random.uniform(0.7, 0.9))
    return x * (attention < threshold).to(dtype=x.dtype)


class FeatureNoise3D(nn.Module):
    def __init__(self, uniform_range=0.3):
        super().__init__()
        self.uniform_range = float(uniform_range)

    def forward(self, x):
        noise = torch.empty_like(x).uniform_(-self.uniform_range, self.uniform_range)
        return x * (1.0 + noise)


class UNetCCT3D(nn.Module):
    """Shared 3D encoder with a main decoder and perturbed auxiliary decoders.

    By default this mirrors the current repository's ``UNet_CCT`` contract and
    returns ``(main_logits, aux_logits)``.  Pass ``return_auxiliary=False`` to
    obtain only the main logits during inference.
    """

    SUPPORTED_PERTURBATIONS = ("dropout", "feature_noise", "feature_dropout")

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        feature_chns=(16, 32, 64, 128, 256),
        strides=DEFAULT_STRIDES,
        dropout=(0.0, 0.0, 0.1, 0.2, 0.3),
        perturbations=("dropout",),
        perturbation_dropout=0.5,
        noise_range=0.3,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        perturbations = tuple(perturbations)
        if not perturbations:
            raise ValueError("UNetCCT3D requires at least one auxiliary perturbation")
        invalid = set(perturbations) - set(self.SUPPORTED_PERTURBATIONS)
        if invalid:
            raise ValueError("Unsupported CCT perturbations: {}".format(sorted(invalid)))
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.perturbations = perturbations
        self.perturbation_dropout = float(perturbation_dropout)
        self.feature_noise = FeatureNoise3D(noise_range)
        self.encoder = Encoder3D(
            self.in_chns,
            feature_chns,
            strides=strides,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.main_decoder = Decoder3D(
            feature_chns,
            self.encoder.strides,
            self.class_num,
            norm=norm,
            activation=activation,
        )
        self.aux_decoders = nn.ModuleList(
            [
                Decoder3D(
                    feature_chns,
                    self.encoder.strides,
                    self.class_num,
                    norm=norm,
                    activation=activation,
                )
                for _ in perturbations
            ]
        )
        initialize_3d_weights(self)

    def _perturb(self, features, perturbation):
        if not self.training:
            return features
        if perturbation == "dropout":
            return [
                F.dropout3d(feature, p=self.perturbation_dropout, training=True)
                for feature in features
            ]
        if perturbation == "feature_noise":
            return [self.feature_noise(feature) for feature in features]
        if perturbation == "feature_dropout":
            return [feature_dropout_3d(feature) for feature in features]
        raise RuntimeError("Unknown perturbation: {}".format(perturbation))

    def forward(self, x, return_auxiliary=True):
        if x.ndim != 5:
            raise ValueError("UNetCCT3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        features = self.encoder(x)
        main_logits = self.main_decoder(features)
        if not return_auxiliary:
            return main_logits
        auxiliary = [
            decoder(self._perturb(features, perturbation))
            for decoder, perturbation in zip(self.aux_decoders, self.perturbations)
        ]
        return (main_logits, *auxiliary)


# The user-facing CTT spelling is retained as an alias.  The method implemented
# here is CCT (Cross-Consistency Training), matching this repository's 2D model.
UNetCTT3D = UNetCCT3D
