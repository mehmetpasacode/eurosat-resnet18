"""CIFAR-style ResNet-18 for small RGB images."""

from pathlib import Path

import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10, width=1.0, dropout=0.2):
        super().__init__()
        channels = [int(base * width) for base in (64, 128, 256, 512)]
        self.inplanes = channels[0]
        self.conv1 = conv3x3(3, channels[0])
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(self, block, channels, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers = [block(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers.extend(block(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            x = layer(x)
        x = self.avg(x).flatten(1)
        return self.fc(self.drop(x))


def resnet18_cifar(num_classes, width=1.0, dropout=0.2):
    return ResNet_CIFAR(BasicBlock, [2, 2, 2, 2], num_classes, width, dropout)


def load_model(out_dir="outputs", device=None):
    """Load a best checkpoint, including legacy checkpoints at default width."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        Path(out_dir) / "best.pt", map_location=device, weights_only=True
    )
    classes = checkpoint["classes"]
    model = resnet18_cifar(
        len(classes),
        width=checkpoint.get("width", 1.0),
        dropout=checkpoint.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), classes, device
