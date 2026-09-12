"""Memory-conscious 3D U-Net building blocks and segmentation network."""

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_STRIDES = ((1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2))


def _triple(value):
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError("Expected a 3D value, got {}".format(value))
    return tuple(int(item) for item in value)


def make_norm_3d(norm, channels):
    if norm == "instance":
        return nn.InstanceNorm3d(channels, affine=True)
    if norm == "batch":
        return nn.BatchNorm3d(channels)
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    if norm in (None, "none"):
        return nn.Identity()
    raise ValueError("Unsupported 3D normalization: {}".format(norm))


def make_activation(activation):
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope=1e-2, inplace=True)
    raise ValueError("Unsupported activation: {}".format(activation))


class ConvBlock3D(nn.Module):
    """Two 3D convolutions with normalization and activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        dropout_p=0.0,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        use_bias = norm in (None, "none")
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=_triple(stride),
                padding=1,
                bias=use_bias,
            ),
            make_norm_3d(norm, out_channels),
            make_activation(activation),
            nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias),
            make_norm_3d(norm, out_channels),
            make_activation(activation),
        )

    def forward(self, x):
        return self.block(x)


class Encoder3D(nn.Module):
    def __init__(
        self,
        in_channels,
        features,
        strides=DEFAULT_STRIDES,
        dropout=(0.0, 0.0, 0.0, 0.0, 0.0),
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        if len(features) != len(strides) + 1:
            raise ValueError("features must contain one more entry than strides")
        if len(dropout) != len(features):
            raise ValueError("dropout must have the same length as features")
        self.strides = tuple(_triple(stride) for stride in strides)
        stages = [
            ConvBlock3D(
                in_channels,
                features[0],
                dropout_p=dropout[0],
                norm=norm,
                activation=activation,
            )
        ]
        for stage, stride in enumerate(self.strides, start=1):
            stages.append(
                ConvBlock3D(
                    features[stage - 1],
                    features[stage],
                    stride=stride,
                    dropout_p=dropout[stage],
                    norm=norm,
                    activation=activation,
                )
            )
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stride,
        dropout_p=0.0,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        stride = _triple(stride)
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
        )
        self.conv = ConvBlock3D(
            out_channels + skip_channels,
            out_channels,
            dropout_p=dropout_p,
            norm=norm,
            activation=activation,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat((skip, x), dim=1))


class Decoder3D(nn.Module):
    def __init__(
        self,
        features,
        strides,
        class_num,
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        blocks = []
        for level in range(len(features) - 1, 0, -1):
            blocks.append(
                UpBlock3D(
                    features[level],
                    features[level - 1],
                    features[level - 1],
                    stride=strides[level - 1],
                    norm=norm,
                    activation=activation,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv3d(features[0], class_num, kernel_size=1)

    def forward(self, features, return_features=False):
        x = features[-1]
        decoder_features = []
        for block, skip in zip(self.blocks, reversed(features[:-1])):
            x = block(x, skip)
            decoder_features.append(x)
        logits = self.out_conv(x)
        if return_features:
            return logits, decoder_features
        return logits


def initialize_3d_weights(module, negative_slope=1e-2):
    for layer in module.modules():
        if isinstance(layer, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(
                layer.weight,
                a=negative_slope,
                mode="fan_out",
                nonlinearity="leaky_relu",
            )
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.InstanceNorm3d, nn.BatchNorm3d, nn.GroupNorm)):
            if layer.weight is not None:
                nn.init.ones_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class UNet3D(nn.Module):
    """3D U-Net returning raw segmentation logits at input resolution."""

    def __init__(
        self,
        in_chns=1,
        class_num=4,
        feature_chns=(16, 32, 64, 128, 256),
        strides=DEFAULT_STRIDES,
        dropout=(0.0, 0.0, 0.1, 0.2, 0.3),
        norm="instance",
        activation="leaky_relu",
    ):
        super().__init__()
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.encoder = Encoder3D(
            self.in_chns,
            feature_chns,
            strides=strides,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.decoder = Decoder3D(
            feature_chns,
            self.encoder.strides,
            self.class_num,
            norm=norm,
            activation=activation,
        )
        initialize_3d_weights(self)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError("UNet3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        features = self.encoder(x)
        if return_features:
            logits, decoder_features = self.decoder(features, return_features=True)
            return logits, {"encoder": features, "decoder": decoder_features}
        return self.decoder(features)
