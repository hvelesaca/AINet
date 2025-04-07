import torch
import torch.nn as nn
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download

class MambaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.mamba = Mamba(d_model=out_channels, d_state=64, d_conv=4, expand=2)
        self.residual = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.residual(x)
        x = self.conv(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        x_mamba = self.mamba(x_flat)
        x_mamba = x_mamba.permute(0, 2, 1).view(B, C, H, W)
        return nn.functional.relu(x_mamba + identity)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VisionMambaUNet(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )
        self.enc1 = MambaConvBlock(features[0], features[0])
        self.enc2 = MambaConvBlock(features[0], features[1], stride=2)
        self.enc3 = MambaConvBlock(features[1], features[2], stride=2)
        self.enc4 = MambaConvBlock(features[2], features[3], stride=2)

        self.dec3 = DecoderBlock(features[3], features[2])
        self.dec2 = DecoderBlock(features[2], features[1])
        self.dec1 = DecoderBlock(features[1], features[0])

        self.seg_head = nn.Conv2d(features[0], 1, kernel_size=1)

        if pretrained:
            self.load_pretrained_weights()

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        d3 = self.dec3(x4, x3)
        d2 = self.dec2(d3, x2)
        d1 = self.dec1(d2, x1)

        out = nn.functional.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.seg_head(out)

        return [out], out

    def load_pretrained_weights(self):
        try:
            model_path = hf_hub_download(
                repo_id="state-spaces/mamba-130m",
                filename="pytorch_model.bin",
                cache_dir="./pretrained_mamba"
            )
            pretrained_weights = torch.load(model_path, map_location='cpu')

            model_dict = self.state_dict()
            matched_weights = {k: v for k, v in pretrained_weights.items() if k in model_dict and v.shape == model_dict[k].shape}

            model_dict.update(matched_weights)
            self.load_state_dict(model_dict)

            print(f"✅ Pesos pre-entrenados cargados exitosamente: {len(matched_weights)}/{len(model_dict)} capas coinciden.")

        except Exception as e:
            print(f"❌ Error cargando pesos pre-entrenados: {e}")
            print("⚠️ Inicializando con pesos aleatorios...")
