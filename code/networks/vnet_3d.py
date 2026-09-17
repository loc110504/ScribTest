"""3D VNet (Milletari et al., 2016), the standard 3D backbone for WORD-style
abdominal CT scribble-supervision benchmarks (DMSPS Sec. 4, the WORD paper).

Faithful port of ``HiLab-git/WSL4MIS``'s and ``HiLab-git/DMSPS``'s identical
``code/networks/vnet.py`` ``VNet`` (verified against both fetched sources),
split into a reusable ``VNetEncoder3D``/``VNetDecoder3D`` pair so
``VNetCCT3D`` below can share one encoder between a main decoder and
dropout-perturbed auxiliary decoders, exactly as ``unet_cct_3d.py`` does for
``UNet3D``. Renamed to this repository's ``in_chns``/``class_num``/
``return_auxiliary`` conventions; the additive (not concatenated) long skip
connections and the per-stage conv-layer counts (1, 2, 3, 3, 3) are preserved
unchanged from the source.
"""

import torch.nn as nn
import torch.nn.functional as F

from .unet_cct_3d import FeatureNoise3D, feature_dropout_3d


def _norm_layer_3d(normalization, channels):
    if normalization == "batchnorm":
        return nn.BatchNorm3d(channels)
    if normalization == "groupnorm":
        return nn.GroupNorm(num_groups=16, num_channels=channels)
    if normalization == "instancenorm":
        return nn.InstanceNorm3d(channels)
    if normalization == "none":
        return None
    raise ValueError("Unsupported VNet normalization: {}".format(normalization))


class VNetConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization="none"):
        super().__init__()
        ops = []
        for stage in range(n_stages):
            input_channel = n_filters_in if stage == 0 else n_filters_out
            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            norm = _norm_layer_3d(normalization, n_filters_out)
            if norm is not None:
                ops.append(norm)
            ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNetDownBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super().__init__()
        ops = [nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)]
        norm = _norm_layer_3d(normalization, n_filters_out)
        if norm is not None:
            ops.append(norm)
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNetUpBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super().__init__()
        ops = [nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)]
        norm = _norm_layer_3d(normalization, n_filters_out)
        if norm is not None:
            ops.append(norm)
        ops.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        return self.conv(x)


class VNetEncoder3D(nn.Module):
    def __init__(self, in_chns, n_filters=16, normalization="none", has_dropout=False, dropout_p=0.5):
        super().__init__()
        self.has_dropout = has_dropout
        self.block_one = VNetConvBlock(1, in_chns, n_filters, normalization)
        self.block_one_dw = VNetDownBlock(n_filters, 2 * n_filters, normalization=normalization)
        self.block_two = VNetConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_two_dw = VNetDownBlock(n_filters * 2, n_filters * 4, normalization=normalization)
        self.block_three = VNetConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_three_dw = VNetDownBlock(n_filters * 4, n_filters * 8, normalization=normalization)
        self.block_four = VNetConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_four_dw = VNetDownBlock(n_filters * 8, n_filters * 16, normalization=normalization)
        self.block_five = VNetConvBlock(3, n_filters * 16, n_filters * 16, normalization)
        self.dropout = nn.Dropout3d(p=dropout_p, inplace=False)

    def forward(self, x):
        x1 = self.block_one(x)
        x1_dw = self.block_one_dw(x1)
        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)
        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)
        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)
        x5 = self.block_five(x4_dw)
        if self.has_dropout:
            x5 = self.dropout(x5)
        return [x1, x2, x3, x4, x5]


class VNetDecoder3D(nn.Module):
    def __init__(self, class_num, n_filters=16, normalization="none", has_dropout=False, dropout_p=0.5):
        super().__init__()
        self.has_dropout = has_dropout
        self.block_five_up = VNetUpBlock(n_filters * 16, n_filters * 8, normalization=normalization)
        self.block_six = VNetConvBlock(3, n_filters * 8, n_filters * 8, normalization)
        self.block_six_up = VNetUpBlock(n_filters * 8, n_filters * 4, normalization=normalization)
        self.block_seven = VNetConvBlock(3, n_filters * 4, n_filters * 4, normalization)
        self.block_seven_up = VNetUpBlock(n_filters * 4, n_filters * 2, normalization=normalization)
        self.block_eight = VNetConvBlock(2, n_filters * 2, n_filters * 2, normalization)
        self.block_eight_up = VNetUpBlock(n_filters * 2, n_filters, normalization=normalization)
        self.block_nine = VNetConvBlock(1, n_filters, n_filters, normalization)
        self.out_conv = nn.Conv3d(n_filters, class_num, 1, padding=0)
        self.dropout = nn.Dropout3d(p=dropout_p, inplace=False)

    def forward(self, features, return_features=False):
        x1, x2, x3, x4, x5 = features
        decoder_features = []
        x5_up = self.block_five_up(x5) + x4
        x6 = self.block_six(x5_up)
        decoder_features.append(x6)
        x6_up = self.block_six_up(x6) + x3
        x7 = self.block_seven(x6_up)
        decoder_features.append(x7)
        x7_up = self.block_seven_up(x7) + x2
        x8 = self.block_eight(x7_up)
        decoder_features.append(x8)
        x8_up = self.block_eight_up(x8) + x1
        x9 = self.block_nine(x8_up)
        decoder_features.append(x9)
        if self.has_dropout:
            x9 = self.dropout(x9)
        logits = self.out_conv(x9)
        if return_features:
            return logits, decoder_features
        return logits


class VNet3D(nn.Module):
    """VNet returning raw segmentation logits at input resolution."""

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        n_filters=16,
        normalization="batchnorm",
        has_dropout=True,
    ):
        super().__init__()
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.encoder = VNetEncoder3D(self.in_chns, n_filters, normalization, has_dropout)
        self.decoder = VNetDecoder3D(self.class_num, n_filters, normalization, has_dropout)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError("VNet3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        features = self.encoder(x)
        if return_features:
            logits, decoder_features = self.decoder(features, return_features=True)
            return logits, {"encoder": features, "decoder": decoder_features}
        return self.decoder(features)


class VNetCCT3D(nn.Module):
    """Shared VNet encoder with a main decoder and perturbed auxiliary decoders.

    Mirrors ``UNetCCT3D`` (``networks/unet_cct_3d.py``) with a VNet backbone
    instead of the concatenation-skip 3D U-Net: returns ``(main_logits,
    *aux_logits)`` by default, or only the main logits when
    ``return_auxiliary=False``.
    """

    SUPPORTED_PERTURBATIONS = ("dropout", "feature_noise", "feature_dropout")

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        n_filters=16,
        normalization="batchnorm",
        has_dropout=True,
        perturbations=("dropout",),
        perturbation_dropout=0.5,
        noise_range=0.3,
    ):
        super().__init__()
        perturbations = tuple(perturbations)
        if not perturbations:
            raise ValueError("VNetCCT3D requires at least one auxiliary perturbation")
        invalid = set(perturbations) - set(self.SUPPORTED_PERTURBATIONS)
        if invalid:
            raise ValueError("Unsupported CCT perturbations: {}".format(sorted(invalid)))
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.perturbations = perturbations
        self.perturbation_dropout = float(perturbation_dropout)
        self.feature_noise = FeatureNoise3D(noise_range)
        self.encoder = VNetEncoder3D(self.in_chns, n_filters, normalization, has_dropout)
        self.main_decoder = VNetDecoder3D(self.class_num, n_filters, normalization, has_dropout)
        self.aux_decoders = nn.ModuleList(
            [
                VNetDecoder3D(self.class_num, n_filters, normalization, has_dropout)
                for _ in perturbations
            ]
        )

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
            raise ValueError("VNetCCT3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
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
