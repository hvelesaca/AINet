import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
import timm

#pvt_v2_variants = [#'pvt_v2_b0',#'pvt_v2_b1',#'pvt_v2_b2', # La que usaste#'pvt_v2_b3',#'pvt_v2_b4',#'pvt_v2_b5',#]
class PVTBackbone(nn.Module):
    def __init__(self, model_name="pvt_v2_b2", pretrained=True):
        super().__init__()
        
        try:
          # Crear el backbone usando timm
          self.backbone = timm.create_model(model_name, features_only=True, pretrained=pretrained)# ¡Importante para obtener skips!
          print(f"Modelo {model_name} cargado exitosamente.")

          # Obtener los canales de salida (importante para conectar a los encoders)
          self.out_channels = [64, 128, 320, 512]
          print(f"Canales de salida de {model_name}: {self.out_channels}")

        except Exception as e:
          print(f"Error al cargar {model_name}: {e}")
          print("Asegúrate de tener 'timm' instalado y que el nombre del modelo sea correcto.")

    def forward_features(self, x):
        return self.backbone(x)

# CBAM Attention Module
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel Attention
        avg_out = self.channel_attention(x)
        max_out = self.channel_attention(F.adaptive_max_pool2d(x, 1))
        x = x * (avg_out + max_out)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_attn

# Mamba Convolutional Block
class MambaConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, mamba_dim: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d((16, 16))
        self.mamba = Mamba(d_model=out_channels, d_state=mamba_dim, d_conv=4, expand=2)
        self.residual = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.residual(x)
        x = self.conv(x)
        B, C, H, W = x.shape
        x_pooled = self.pool(x).flatten(2).transpose(1, 2)
        x_mamba = self.mamba(x_pooled).transpose(1, 2).view(B, C, 16, 16)
        x_mamba = F.interpolate(x_mamba, size=(H, W), mode='bilinear', align_corners=False)
        return F.relu(x_mamba + identity)

# Attention Decoder Block with CBAM
class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.cbam = CBAM(out_channels * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.cbam(x)
        return self.conv(x)

# Modelo Completo con Deep Supervision y estructura U-Net
class CamouflageDetectionNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        
        self.backbone = PVTBackbone("pvt_v2_b2", pretrained=True)
        out_channels = self.backbone.out_channels  # [96, 192, 384, 768]

        self.encoders = nn.ModuleList([
            MambaConvBlock(pvt_channels[i], features[i]) for i in range(4)
        ])

        self.decoders = nn.ModuleList([
            AttentionDecoderBlock(features[3], features[2]),
            AttentionDecoderBlock(features[2], features[1]),
            AttentionDecoderBlock(features[1], features[0])
        ])

        self.seg_heads = nn.ModuleList([
            nn.Conv2d(features[i], 1, kernel_size=1) for i in range(3)
        ])
        
    def forward(self, x: torch.Tensor):
        skips = self.backbone.forward_features(x)
        enc_feats = [enc(skip) for enc, skip in zip(self.encoders, skips)]

        d3 = self.decoders[0](enc_feats[3], enc_feats[2])
        d2 = self.decoders[1](d3, enc_feats[1])
        d1 = self.decoders[2](d2, enc_feats[0])

        out1 = F.interpolate(self.seg_heads[0](d1), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_heads[1](d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.seg_heads[2](d3), size=x.shape[2:], mode='bilinear', align_corners=False)

        final_out = (out1 + out2 + out3) / 3
        return [out1, out2, out3], final_out
        
# Ejemplo de uso optimizado
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CamouflageDetectionNet(pretrained=True).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        outputs, final_output = model(x)
        print(final_output.shape)  # [1, 1, 256, 256]
