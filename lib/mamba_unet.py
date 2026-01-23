import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
import timm
from lib.pvtv2 import pvt_v2_b2

# Residual Block (Conv + InstanceNorm + LeakyReLU + Residual)
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x + residual
    
#https://github.com/Jongchan/attention-module/blob/c06383c514ab0032d044cc6fcd8c8207ea222ea7/MODELS/cbam.py
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class UMambaConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mamba_dim: int = 64, use_mamba: bool = True, use_cbam: bool = True):
        super().__init__()
        self.project_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

        self.layer_norm = nn.LayerNorm(out_channels)

        # Branch 1
        self.linear1_branch1 = nn.Linear(out_channels, out_channels)
        self.conv1d_branch1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.act_branch1 = nn.SiLU()
        
        # Mamba
        self.use_mamba = use_mamba
        if self.use_mamba:
            self.mamba = Mamba(d_model=out_channels, d_state=mamba_dim, d_conv=4, expand=2)

        # Branch 2
        self.linear1_branch2 = nn.Linear(out_channels, out_channels)
        self.act_branch2 = nn.SiLU()

        # Final projection
        self.linear2 = nn.Linear(out_channels, out_channels)

        # CBAM (opcional)
        self.use_cbam = use_cbam
        if self.use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # Step 0: Project input if in_channels != out_channels
        x = self.project_in(x)

        # Step 1: Two Residual Blocks
        x = self.res_block1(x)
        x = self.res_block2(x)

        # Step 2: Flatten and Permute
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        # Step 3: LayerNorm
        x = self.layer_norm(x)

        # Step 4: Split into two branches
        # Branch 1
        branch1 = self.linear1_branch1(x)
        branch1 = branch1.transpose(1, 2)
        branch1 = self.conv1d_branch1(branch1)
        branch1 = branch1.transpose(1, 2)
        branch1 = self.act_branch1(branch1)
        
        # Aplicar Mamba solo si está activo
        if self.use_mamba:
            branch1 = self.mamba(branch1)
        
        # Branch 2
        branch2 = self.linear1_branch2(x)
        branch2 = self.act_branch2(branch2)
        # Step 5: Hadamard Product
        x = branch1 * branch2
        # Step 6: Linear Projection
        x = self.linear2(x)
        # Step 7: Reshape back
        x = x.transpose(1, 2).view(B, -1, H, W)

        # Aplicar CBAM solo si está activo
        if self.use_cbam:
            x = self.cbam(x)

        return x

# Simple Decoder Block with UMamba (No CBAM)
class SimpleDecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.umamba_block = UMambaConvBlock(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concatenar skip connection
        x = self.umamba_block(x)
        return x

# --- Nuevo bloque para Feature Aggregation ---
class FeatureAggregation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        x = torch.cat(features, dim=1)
        x = self.conv1x1(x)
        x = self.conv3x3(x)
        return x

# --- Decoder Block modificado ---
class AdvancedDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.aggr = FeatureAggregation(in_channels + skip_channels, out_channels)
        self.umamba_block = UMambaConvBlock(out_channels, out_channels)

    def forward(self, x, skips):
        x = self.upsample(x)
        skips = [F.interpolate(feat, size=x.shape[2:], mode='bilinear', align_corners=False) for feat in skips]
        x = self.aggr([x] + skips)
        x = self.umamba_block(x)
        return x

# --- Modelo Completo ---
class CamouflageDetectionNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if pretrained:
            self._load_backbone_weights('/kaggle/input/pretrained_pvt_v2_b2/pytorch/default/1/pvt_v2_b2.pth')

        out_channels = [64, 128, 320, 512] 

        self.encoders = nn.ModuleList([
            UMambaConvBlock(out_channels[i], features[i]) for i in range(4)
        ])

        self.decoder3 = AdvancedDecoderBlock(features[3], features[2], features[2])
        self.decoder2 = AdvancedDecoderBlock(features[2], features[1], features[1])
        self.decoder1 = AdvancedDecoderBlock(features[1], features[0], features[0])

        self.final_decoder = UMambaConvBlock(features[0], features[0])

        self.seg_heads = nn.ModuleList([
            nn.Conv2d(features[2], 1, kernel_size=1),
            nn.Conv2d(features[1], 1, kernel_size=1),
            nn.Conv2d(features[0], 1, kernel_size=1),
            nn.Conv2d(features[0], 1, kernel_size=1),
        ])

    def forward(self, x: torch.Tensor):
        # Backbone features
        skips = self.backbone.forward_features(x)
        enc_feats = [enc(skip) for enc, skip in zip(self.encoders, skips)]

        # Decoder path
        d3 = self.decoder3(enc_feats[3], [enc_feats[2]])
        d2 = self.decoder2(d3, [enc_feats[1]])
        d1 = self.decoder1(d2, [enc_feats[0]])

        d0 = self.final_decoder(d1)

        # Deep supervision
        out3 = F.interpolate(self.seg_heads[0](d3), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_heads[1](d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_heads[2](d1), size=x.shape[2:], mode='bilinear', align_corners=False)
        out0 = F.interpolate(self.seg_heads[3](d0), size=x.shape[2:], mode='bilinear', align_corners=False)

        final_out = (out0 + out1 + out2 + out3) / 4

        return [out0, out1, out2, out3], final_out

    def _load_backbone_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("✅ Pesos backbone cargados correctamente.")
        except Exception as e:
            print(f"❌ Error cargando pesos backbone: {e}")


# Ejemplo de uso optimizado
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CamouflageDetectionNet(pretrained=True).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        outputs, final_output = model(x)
        print(final_output.shape)  # [1, 1, 256, 256]
