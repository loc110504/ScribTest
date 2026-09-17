"""2D U-Net, the standard backbone across ACDC/MSCMR scribble-supervision
literature (WSL4MIS, DMSPS, CycleMix, ScribFormer).

Faithful port of ``HiLab-git/WSL4MIS``'s ``code/networks/unet.py`` ``Encoder``/
``Decoder``/``UNet``/``UNet_CCT`` (BatchNorm2d + LeakyReLU, per-stage dropout
``[0.05, 0.1, 0.2, 0.3, 0.5]``, ``ConvTranspose2d`` upsampling -- the
``bilinear=False`` branch of the source), verified against the fetched
upstream file. Renamed to this repository's ``in_chns``/``class_num``/
``return_auxiliary`` conventions (matching ``unet_3d.py``/``unet_cct_3d.py``'s
public API) rather than HiLab's own argument names.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_FEATURE_CHNS = (16, 32, 64, 128, 256)
DEFAULT_DROPOUT = (0.05, 0.1, 0.2, 0.3, 0.5)


class ConvBlock2D(nn.Module):
    """Two 2D convolutions with batch norm and leaky relu."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv_conv(x)


class DownBlock2D(nn.Module):
    """Downsampling followed by ConvBlock2D."""

    def __init__(self, in_channels, out_channels, dropout_p):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock2D(in_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock2D(nn.Module):
    """Transpose-conv upsampling followed by ConvBlock2D."""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvBlock2D(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Encoder2D(nn.Module):
    def __init__(self, in_chns, feature_chns=DEFAULT_FEATURE_CHNS, dropout=DEFAULT_DROPOUT):
        super().__init__()
        if len(feature_chns) != 5:
            raise ValueError("Encoder2D expects 5 feature_chns stages")
        if len(dropout) != 5:
            raise ValueError("Encoder2D expects 5 dropout stages")
        self.in_conv = ConvBlock2D(in_chns, feature_chns[0], dropout[0])
        self.down1 = DownBlock2D(feature_chns[0], feature_chns[1], dropout[1])
        self.down2 = DownBlock2D(feature_chns[1], feature_chns[2], dropout[2])
        self.down3 = DownBlock2D(feature_chns[2], feature_chns[3], dropout[3])
        self.down4 = DownBlock2D(feature_chns[3], feature_chns[4], dropout[4])

    def forward(self, x):
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        return [x0, x1, x2, x3, x4]


class Decoder2D(nn.Module):
    def __init__(self, feature_chns=DEFAULT_FEATURE_CHNS, class_num=4):
        super().__init__()
        if len(feature_chns) != 5:
            raise ValueError("Decoder2D expects 5 feature_chns stages")
        self.up1 = UpBlock2D(feature_chns[4], feature_chns[3], feature_chns[3], dropout_p=0.0)
        self.up2 = UpBlock2D(feature_chns[3], feature_chns[2], feature_chns[2], dropout_p=0.0)
        self.up3 = UpBlock2D(feature_chns[2], feature_chns[1], feature_chns[1], dropout_p=0.0)
        self.up4 = UpBlock2D(feature_chns[1], feature_chns[0], feature_chns[0], dropout_p=0.0)
        self.out_conv = nn.Conv2d(feature_chns[0], class_num, kernel_size=3, padding=1)

    def forward(self, features, return_features=False):
        x0, x1, x2, x3, x4 = features
        decoder_features = []
        x = self.up1(x4, x3)
        decoder_features.append(x)
        x = self.up2(x, x2)
        decoder_features.append(x)
        x = self.up3(x, x1)
        decoder_features.append(x)
        x = self.up4(x, x0)
        decoder_features.append(x)
        logits = self.out_conv(x)
        if return_features:
            return logits, decoder_features
        return logits


class UNet2D(nn.Module):
    """2D U-Net returning raw segmentation logits at input resolution."""

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        feature_chns=DEFAULT_FEATURE_CHNS,
        dropout=DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.encoder = Encoder2D(self.in_chns, feature_chns, dropout)
        self.decoder = Decoder2D(feature_chns, self.class_num)

    def forward(self, x, return_features=False):
        if x.ndim != 4:
            raise ValueError("UNet2D expects [B, C, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        features = self.encoder(x)
        if return_features:
            logits, decoder_features = self.decoder(features, return_features=True)
            return logits, {"encoder": features, "decoder": decoder_features}
        return self.decoder(features)


def feature_dropout_2d(x):
    attention = torch.mean(x, dim=1, keepdim=True)
    maximum = attention.flatten(1).amax(dim=1).view(-1, 1, 1, 1)
    threshold = maximum * float(np.random.uniform(0.7, 0.9))
    return x * (attention < threshold).to(dtype=x.dtype)


class FeatureNoise2D(nn.Module):
    def __init__(self, uniform_range=0.3):
        super().__init__()
        self.uniform_range = float(uniform_range)

    def forward(self, x):
        noise = torch.empty_like(x).uniform_(-self.uniform_range, self.uniform_range)
        return x * (1.0 + noise)


class UNetCCT2D(nn.Module):
    """Shared 2D encoder with a main decoder and perturbed auxiliary decoders.

    Mirrors ``UNetCCT3D`` (``networks/unet_cct_3d.py``) exactly, one spatial
    dimension down: returns ``(main_logits, *aux_logits)`` by default, or
    only the main logits when ``return_auxiliary=False``.
    """

    SUPPORTED_PERTURBATIONS = ("dropout", "feature_noise", "feature_dropout")

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        feature_chns=DEFAULT_FEATURE_CHNS,
        dropout=DEFAULT_DROPOUT,
        perturbations=("dropout",),
        perturbation_dropout=0.5,
        noise_range=0.3,
    ):
        super().__init__()
        perturbations = tuple(perturbations)
        if not perturbations:
            raise ValueError("UNetCCT2D requires at least one auxiliary perturbation")
        invalid = set(perturbations) - set(self.SUPPORTED_PERTURBATIONS)
        if invalid:
            raise ValueError("Unsupported CCT perturbations: {}".format(sorted(invalid)))
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.perturbations = perturbations
        self.perturbation_dropout = float(perturbation_dropout)
        self.feature_noise = FeatureNoise2D(noise_range)
        self.encoder = Encoder2D(self.in_chns, feature_chns, dropout)
        self.main_decoder = Decoder2D(feature_chns, self.class_num)
        self.aux_decoders = nn.ModuleList(
            [Decoder2D(feature_chns, self.class_num) for _ in perturbations]
        )

    def _perturb(self, features, perturbation):
        if not self.training:
            return features
        if perturbation == "dropout":
            return [
                F.dropout2d(feature, p=self.perturbation_dropout, training=True)
                for feature in features
            ]
        if perturbation == "feature_noise":
            return [self.feature_noise(feature) for feature in features]
        if perturbation == "feature_dropout":
            return [feature_dropout_2d(feature) for feature in features]
        raise RuntimeError("Unknown perturbation: {}".format(perturbation))

    def forward(self, x, return_auxiliary=True):
        if x.ndim != 4:
            raise ValueError("UNetCCT2D expects [B, C, H, W], got {}".format(tuple(x.shape)))
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


# The user-facing CTT spelling is retained as an alias, matching unet_cct_3d.py.
UNetCTT2D = UNetCCT2D
