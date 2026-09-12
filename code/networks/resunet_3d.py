"""Residual 3D U-Net for volumetric medical image segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet_3d import DEFAULT_STRIDES, _triple, initialize_3d_weights, make_activation, make_norm_3d


class ResidualBlock3D(nn.Module):
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
        stride = _triple(stride)
        use_bias = norm in (None, "none")
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=use_bias
        )
        self.norm1 = make_norm_3d(norm, out_channels)
        self.act1 = make_activation(activation)
        self.dropout = nn.Dropout3d(dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=use_bias)
        self.norm2 = make_norm_3d(norm, out_channels)
        if in_channels != out_channels or stride != (1, 1, 1):
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_norm_3d(norm, out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.out_act = make_activation(activation)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        return self.out_act(x + residual)


class ResidualUpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, stride, norm, activation):
        super().__init__()
        stride = _triple(stride)
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=stride, stride=stride
        )
        self.fuse = ResidualBlock3D(
            out_channels + skip_channels,
            out_channels,
            norm=norm,
            activation=activation,
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat((skip, x), dim=1))


class ResUNet3D(nn.Module):
    """Encoder-decoder 3D U-Net with residual blocks at every resolution."""

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
        if len(feature_chns) != len(strides) + 1:
            raise ValueError("feature_chns must contain one more entry than strides")
        if len(dropout) != len(feature_chns):
            raise ValueError("dropout must have the same length as feature_chns")
        self.in_chns = int(in_chns)
        self.class_num = int(class_num)
        self.strides = tuple(_triple(stride) for stride in strides)
        encoder = [
            ResidualBlock3D(
                self.in_chns,
                feature_chns[0],
                dropout_p=dropout[0],
                norm=norm,
                activation=activation,
            )
        ]
        for level, stride in enumerate(self.strides, start=1):
            encoder.append(
                ResidualBlock3D(
                    feature_chns[level - 1],
                    feature_chns[level],
                    stride=stride,
                    dropout_p=dropout[level],
                    norm=norm,
                    activation=activation,
                )
            )
        self.encoder = nn.ModuleList(encoder)

        decoder = []
        for level in range(len(feature_chns) - 1, 0, -1):
            decoder.append(
                ResidualUpBlock3D(
                    feature_chns[level],
                    feature_chns[level - 1],
                    feature_chns[level - 1],
                    self.strides[level - 1],
                    norm,
                    activation,
                )
            )
        self.decoder = nn.ModuleList(decoder)
        self.out_conv = nn.Conv3d(feature_chns[0], self.class_num, kernel_size=1)
        initialize_3d_weights(self)

    def forward(self, x, return_features=False):
        if x.ndim != 5:
            raise ValueError("ResUNet3D expects [B, C, D, H, W], got {}".format(tuple(x.shape)))
        if x.shape[1] != self.in_chns:
            raise ValueError("Expected {} input channels, got {}".format(self.in_chns, x.shape[1]))
        encoder_features = []
        for stage in self.encoder:
            x = stage(x)
            encoder_features.append(x)
        decoder_features = []
        for block, skip in zip(self.decoder, reversed(encoder_features[:-1])):
            x = block(x, skip)
            decoder_features.append(x)
        logits = self.out_conv(x)
        if return_features:
            return logits, {"encoder": encoder_features, "decoder": decoder_features}
        return logits
