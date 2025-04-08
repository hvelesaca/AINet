import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
from lib.pvtv2 import pvt_v2_b2

# CBAM Attention Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_attn = self.sigmoid_channel(avg_out + max_out)
        x = x * channel_attn

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attn = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_attn

        return x

# Mamba Convolutional Block
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

# Attention Decoder Block with CBAM
class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.cbam = CBAM(out_channels * 2)
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
        x = self.cbam(x)
        return self.conv(x)

# Modelo Completo con Deep Supervision
class VisionMambaPVTUNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        path = './pretrained_pvt/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        pvt_channels = [64, 128, 320, 512]

        self.enc1 = MambaConvBlock(pvt_channels[0], features[0])
        self.enc2 = MambaConvBlock(pvt_channels[1], features[1])
        self.enc3 = MambaConvBlock(pvt_channels[2], features[2])
        self.enc4 = MambaConvBlock(pvt_channels[3], features[3])

        self.dec3 = AttentionDecoderBlock(features[3], features[2])
        self.dec2 = AttentionDecoderBlock(features[2], features[1])
        self.dec1 = AttentionDecoderBlock(features[1], features[0])

        # Deep supervision heads
        self.seg_head1 = nn.Conv2d(features[0], 1, kernel_size=1)
        self.seg_head2 = nn.Conv2d(features[1], 1, kernel_size=1)
        self.seg_head3 = nn.Conv2d(features[2], 1, kernel_size=1)

        if pretrained:
            self.load_pretrained_mamba_weights()

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone.forward_features(x)

        e1 = self.enc1(x1)
        e2 = self.enc2(x2)
        e3 = self.enc3(x3)
        e4 = self.enc4(x4)

        d3 = self.dec3(e4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out1 = F.interpolate(self.seg_head1(d1), scale_factor=4, mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_head2(d2), scale_factor=8, mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.seg_head3(d3), scale_factor=16, mode='bilinear', align_corners=False)

        final_out = (out1 + out2 + out3) / 3

        return [out1, out2, out3], final_out

    def load_pretrained_mamba_weights(self):
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
            self.load_state_dict(model_dict, strict=False)
            print(f"✅ Pesos Mamba cargados parcialmente: {len(matched_weights)} capas.")
        except Exception as e:
            print(f"❌ Error cargando pesos Mamba: {e}")

# Ejemplo de uso
if __name__ == "__main__":
    model = VisionMambaPVTUNet(pretrained=True)
    x = torch.randn(1, 3, 256, 256)
    outputs, final_output = model(x)
    print(final_output.shape)  # [1, 1, 256, 256]
