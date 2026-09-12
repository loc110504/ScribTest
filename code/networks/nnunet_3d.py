"""Standalone nnU-Net-style 3D PlainConvUNet.

This is a network component compatible with this repository.  It intentionally
does not claim to reproduce nnU-Net's dataset fingerprinting, planning,
preprocessing, training, ensembling or inference pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_3d import DEFAULT_STRIDES, _triple, initialize_3d_weights, make_activation, make_norm_3d


class StackedConvBlocks3D(nn.Module):
    def __init__(
        self,
        num_convs,
        in_channels,
        out_channels,
        first_stride=1,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        if num_convs < 1:
            raise ValueError("num_convs must be positive")
        use_bias = norm in (None, "none")
        blocks = []
        for index in range(num_convs):
            blocks.extend(
                [
                    nn.Conv3d(
                        in_channels if index == 0 else out_channels,
                        out_channels,
                        kernel_size=3,
                        stride=_triple(first_stride) if index == 0 else 1,
                        padding=1,
                        bias=use_bias,
                    ),
                    make_norm_3d(norm, out_channels),
                    make_activation(activation),
                ]
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


def _stage_values(value, length, name):
    if isinstance(value, int):
        return (value,) * length
    values = tuple(int(item) for item in value)
    if len(values) != length:
        raise ValueError("{} must contain {} entries".format(name, length))
    return values


class NNUNet3D(nn.Module):
    """nnU-Net-style PlainConvUNet with optional native-scale deep supervision."""

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        feature_chns=(32, 64, 128, 256, 320),
        strides=DEFAULT_STRIDES,
        n_conv_per_stage=2,
        n_conv_per_stage_decoder=2,
        deep_supervision=False,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        if len(feature_chns) != len(strides) + 1:
            raise ValueError("feature_chns must contain one more entry than strides")
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.deep_supervision = bool(deep_supervision)
        self.strides = tuple(_triple(stride) for stride in strides)
        encoder_convs = _stage_values(n_conv_per_stage, len(feature_chns), "n_conv_per_stage")
        decoder_convs = _stage_values(
            n_conv_per_stage_decoder, len(feature_chns) - 1, "n_conv_per_stage_decoder"
        )

        encoder = []
        for level, channels in enumerate(feature_chns):
            encoder.append(
                StackedConvBlocks3D(
                    encoder_convs[level],
                    self.in_chns if level == 0 else feature_chns[level - 1],
                    channels,
                    first_stride=1 if level == 0 else self.strides[level - 1],
                    norm=norm,
                    activation=activation,
                )
            )
        self.encoder = nn.ModuleList(encoder)

        upconvs = []
        decoder = []
        seg_heads = []
        for decoder_index, level in enumerate(range(len(feature_chns) - 1, 0, -1)):
            stride = self.strides[level - 1]
            out_channels = feature_chns[level - 1]
            upconvs.append(
                nn.ConvTranspose3d(
                    feature_chns[level], out_channels, kernel_size=stride, stride=stride
                )
            )
            decoder.append(
                StackedConvBlocks3D(
                    decoder_convs[decoder_index],
                    out_channels * 2,
                    out_channels,
                    norm=norm,
                    activation=activation,
                )
            )
            seg_heads.append(nn.Conv3d(out_channels, self.class_num, kernel_size=1))
        self.upconvs = nn.ModuleList(upconvs)
        self.decoder = nn.ModuleList(decoder)
        self.seg_heads = nn.ModuleList(seg_heads)
        initialize_3d_weights(self)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError("NNUNet3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        skips = []
        for stage in self.encoder:
            x = stage(x)
            skips.append(x)

        segmentation_outputs = []
        for up, stage, head, skip in zip(
            self.upconvs, self.decoder, self.seg_heads, reversed(skips[:-1])
        ):
            x = up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = stage(torch.cat((skip, x), dim=1))
            segmentation_outputs.append(head(x))

        if self.deep_supervision:
            return tuple(reversed(segmentation_outputs))
        return segmentation_outputs[-1]
