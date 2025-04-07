import torch
import torch.nn as nn
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2

class MambaConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
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
        x_pooled = self.pool(x)
        x_flat = x_pooled.view(B, C, -1).permute(0, 2, 1)
        x_mamba = self.mamba(x_flat)
        x_mamba = x_mamba.permute(0, 2, 1).view(B, C, 16, 16)
        x_mamba = F.interpolate(x_mamba, size=(H, W), mode='bilinear', align_corners=False)
        return F.relu(x_mamba + identity)

class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
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
        attn = self.attention(torch.cat([x, skip], dim=1))
        skip = skip * attn
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class VisionMambaPVTUNet(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        # Backbone PVT preentrenado
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # Extraer canales del backbone PVT
        pvt_channels = [64, 128, 320, 512]

        # Bloques Mamba para procesar características del backbone
        self.enc1 = MambaConvBlock(pvt_channels[0], features[0])
        self.enc2 = MambaConvBlock(pvt_channels[1], features[1])
        self.enc3 = MambaConvBlock(pvt_channels[2], features[2])
        self.enc4 = MambaConvBlock(pvt_channels[3], features[3])

        # Decoder con atención
        self.dec3 = AttentionDecoderBlock(features[3], features[2])
        self.dec2 = AttentionDecoderBlock(features[2], features[1])
        self.dec1 = AttentionDecoderBlock(features[1], features[0])

        self.seg_head = nn.Conv2d(features[0], 1, kernel_size=1)

        if pretrained:
            self.load_pretrained_mamba_weights()

    def forward(self, x):
        # Backbone PVT
        pvt_feats = self.backbone.forward_features(x)
        x1, x2, x3, x4 = pvt_feats  # características multi-escala del backbone

        # Encoder con Mamba
        e1 = self.enc1(x1)
        e2 = self.enc2(x2)
        e3 = self.enc3(x3)
        e4 = self.enc4(x4)

        # Decoder con atención
        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = F.interpolate(d1, scale_factor=4, mode='bilinear', align_corners=False)
        out = self.seg_head(out)

        return [out], out

    def load_pretrained_mamba_weights(self):
        try:
            model_path = hf_hub_download(
                repo_id="state-spaces/mamba-130m",
                filename="pytorch_model.bin",
                cache_dir="./pretrained_mamba"
            )
            pretrained_weights = torch.load(model_path, map_location='cpu')

            model_dict = self.state_dict()
            matched_weights = {}

            # Carga parcial solo para bloques Mamba internos
            for name, param in pretrained_weights.items():
                for model_name in model_dict.keys():
                    if model_name.endswith(name) and param.shape == model_dict[model_name].shape:
                        matched_weights[model_name] = param
                        print(f"✅ Coincidencia encontrada: {model_name}")

            model_dict.update(matched_weights)
            self.load_state_dict(model_dict, strict=False)

            print(f"✅ Pesos pre-entrenados Mamba cargados parcialmente: {len(matched_weights)}/{len(model_dict)} capas coinciden.")

        except Exception as e:
            print(f"❌ Error cargando pesos pre-entrenados Mamba: {e}")
            print("⚠️ Inicializando bloques Mamba con pesos aleatorios...")

# Ejemplo de uso:
if __name__ == "__main__":
    model = VisionMambaPVTUNet(pretrained=True)
    x = torch.randn(1, 3, 256, 256)
    preds, _ = model(x)
    print(preds[0].shape)  # [1, 1, 256, 256]
