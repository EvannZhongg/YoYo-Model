from __future__ import annotations

import torch
from torch import nn

from string_segmentation.semantic_model import ConvBlock, _FPNRefine, _group_count


class MultiTaskCenterlineFPN(nn.Module):
    """MobileNetV3-FPN with mask and geometry heads.

    Channels are ordered as ``mask_logit, centerline_logit, cos_2theta,
    sin_2theta``.  Keeping a tensor output makes the module usable by the
    existing CUDA inference path while the two supervision targets remain
    explicitly separated in the loss.
    """

    _FEATURE_INDICES = (1, 3, 6, 12, 16)
    _FEATURE_CHANNELS = (16, 24, 40, 112, 960)

    def __init__(self, decoder_channels: int = 32, pretrained_backbone: bool = False):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.encoder = mobilenet_v3_large(weights=weights).features
        self.lateral = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, decoder_channels, 1, bias=False),
                nn.GroupNorm(_group_count(decoder_channels), decoder_channels),
            )
            for channels in self._FEATURE_CHANNELS
        )
        self.refine = nn.ModuleList(_FPNRefine(decoder_channels) for _ in range(4))
        self.mask_head = nn.Sequential(_FPNRefine(decoder_channels), nn.Conv2d(decoder_channels, 1, 1))
        self.geometry_head = nn.Sequential(_FPNRefine(decoder_channels), nn.Conv2d(decoder_channels, 3, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self

    def _features(self, value: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        input_size = value.shape[-2:]
        selected = set(self._FEATURE_INDICES)
        features = []
        for index, layer in enumerate(self.encoder):
            value = layer(value)
            if index in selected:
                features.append(value)
        pyramid = self.lateral[-1](features[-1])
        for level in range(len(features) - 2, -1, -1):
            pyramid = nn.functional.interpolate(
                pyramid, size=features[level].shape[-2:], mode="bilinear", align_corners=False
            )
            pyramid = self.refine[level](pyramid + self.lateral[level](features[level]))
        return pyramid, input_size

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        pyramid, input_size = self._features(value)
        output = torch.cat((self.mask_head(pyramid), self.geometry_head(pyramid)), dim=1)
        return nn.functional.interpolate(output, size=input_size, mode="bilinear", align_corners=False)


class TinyMultiTaskUNet(nn.Module):
    """CPU-friendly four-channel smoke-test model."""

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c = [base_channels * m for m in (1, 2, 4, 8, 16)]
        self.enc1, self.enc2 = ConvBlock(3, c[0]), ConvBlock(c[0], c[1])
        self.enc3, self.enc4 = ConvBlock(c[1], c[2]), ConvBlock(c[2], c[3])
        self.bottleneck = ConvBlock(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.dec4, self.dec3 = ConvBlock(c[4] + c[3], c[3]), ConvBlock(c[3] + c[2], c[2])
        self.dec2, self.dec1 = ConvBlock(c[2] + c[1], c[1]), ConvBlock(c[1] + c[0], c[0])
        self.output = nn.Conv2d(c[0], 4, 1)

    @staticmethod
    def _up(value, skip):
        return nn.functional.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(value)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat((self._up(bottleneck, e4), e4), dim=1))
        d3 = self.dec3(torch.cat((self._up(d4, e3), e3), dim=1))
        d2 = self.dec2(torch.cat((self._up(d3, e2), e2), dim=1))
        d1 = self.dec1(torch.cat((self._up(d2, e1), e1), dim=1))
        return self.output(d1)


def build_model(
    architecture: str = "mobilenet_v3_fpn",
    base_channels: int = 16,
    pretrained_backbone: bool = False,
) -> nn.Module:
    architecture = str(architecture or "mobilenet_v3_fpn").strip().lower()
    if architecture == "mobilenet_v3_fpn":
        return MultiTaskCenterlineFPN(
            decoder_channels=max(16, int(base_channels) * 2),
            pretrained_backbone=bool(pretrained_backbone),
        )
    if architecture == "tiny_unet":
        return TinyMultiTaskUNet(base_channels=max(4, int(base_channels)))
    raise ValueError(f"Unsupported centerline architecture: {architecture}")
