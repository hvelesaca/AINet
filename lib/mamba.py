# Añadir este nuevo archivo mamba.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba

class VisionMamba(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = MambaBlock(features[0], features[0])
        self.stage2 = MambaBlock(features[0], features[1], stride=2)
        self.stage3 = MambaBlock(features[1], features[2], stride=2)
        self.stage4 = MambaBlock(features[2], features[3], stride=2)

    def forward(self, x):
        x = self.stem(x)
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        return [c1, c2, c3, c4]

class MambaBlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.mamba = Mamba(
            d_model=out_dim,
            d_state=32,
            d_conv=4,
            expand=2
        )

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).permute(0, 2, 1)  # (B, L, C)
        x = self.mamba(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x
