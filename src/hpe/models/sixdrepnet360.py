from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models.resnet import Bottleneck

from hpe.geometry.rotations import rotation_matrix_from_6d


class SixDRepNet360(nn.Module):
    """6DRepNet360 ResNet-50 architecture used by the published checkpoint."""

    def __init__(self, *, rotation_fp32: bool = True) -> None:
        super().__init__()
        # This flag adds no parameters or buffers; published state dicts stay compatible.
        self.rotation_fp32 = rotation_fp32
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.linear_reg = nn.Linear(512 * Bottleneck.expansion, 6)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * Bottleneck.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers: list[nn.Module] = [Bottleneck(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * Bottleneck.expansion
        layers.extend(Bottleneck(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.maxpool(self.relu(self.bn1(self.conv1(images))))
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        features = self.avgpool(features).flatten(1)
        poses = self.linear_reg(features)
        if self.rotation_fp32 and poses.dtype in (torch.float16, torch.bfloat16):
            poses = poses.float()
        return rotation_matrix_from_6d(poses)
