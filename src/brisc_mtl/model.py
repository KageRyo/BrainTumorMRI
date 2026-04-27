from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import timm


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ConvNeXtUNetMultiTask(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int = 4,
        in_channels: int = 1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            features_only=True,
            pretrained=pretrained,
            in_chans=in_channels,
        )
        channels: List[int] = self.encoder.feature_info.channels()
        if len(channels) < 4:
            raise ValueError(f"Expected at least 4 feature levels from {backbone}, got {channels}")

        c1, c2, c3, c4 = channels[-4:]
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(c4, num_classes),
        )
        self.dec3 = DecoderBlock(c4, c3, 512)
        self.dec2 = DecoderBlock(512, c2, 256)
        self.dec1 = DecoderBlock(256, c1, 128)
        self.refine = nn.Sequential(
            ConvBlock(128, 64),
            nn.Conv2d(64, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.encoder(x)
        f1, f2, f3, f4 = features[-4:]
        class_logits = self.classifier(f4)
        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        mask_logits = self.refine(x)
        mask_logits = F.interpolate(mask_logits, size=input_size, mode="bilinear", align_corners=False)
        return {"class_logits": class_logits, "mask_logits": mask_logits}
