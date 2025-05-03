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
    def __init__(self, in_channels: int, out_channels: int, mamba_dim: int = 64, use_cbam: bool = True):
        super().__init__()
        self.project_in = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.res_block1 = ResidualBlock(out_channels)
        self.res_block2 = ResidualBlock(out_channels)

        self.layer_norm = nn.LayerNorm(out_channels)

        # Branch 1
        self.linear1_branch1 = nn.Linear(out_channels, out_channels)
        self.conv1d_branch1 = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.act_branch1 = nn.SiLU()
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

# --- Decoder Block ---
class DecoderBlock(nn.Module):
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

class Conv_Block(nn.Module):
    def __init__(self, channels):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(channels*3, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels*2, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(channels*2)

        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, input1, input2, input3):
        fuse = torch.cat((input1, input2, input3), 1)
        fuse = self.bn1(self.conv1(fuse))
        fuse = self.bn2(self.conv2(fuse))
        fuse = self.bn3(self.conv3(fuse))
        return fuse

    def initialize(self):
        weight_init(self)

# --- Modelo Completo ---
class CamouflageDetectionNet(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()
        if pretrained:
            self._load_backbone_weights('/kaggle/input/pretrained_pvt_v2_b2/pytorch/default/1/pvt_v2_b2.pth')

        out_channels = [64, 128, 320, 512] 
        channels=128
        
        self.encoders = nn.ModuleList([
            UMambaConvBlock(out_channels[i], features[i]) for i in range(4)
        ])

        self.decoder3 = DecoderBlock(features[3], features[2], features[2])
        self.decoder2 = DecoderBlock(features[2], features[1], features[1])
        self.decoder1 = DecoderBlock(features[1], features[0], features[0])

        self.conv_block = Conv_Block(channels)

        self.fuse1 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse2 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))
        self.fuse3 = nn.Sequential(nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),nn.BatchNorm2d(channels))

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

        enc1 = enc_feats[0]
        enc2 = enc_feats[1]
        enc3 = enc_feats[2]
        enc4 = enc_feats[3]

        if enc4.size()[2:] != enc3.size()[2:]:
            enc4 = F.interpolate(enc4, size=enc3.size()[2:], mode='bilinear')
        if enc2.size()[2:] != enc3.size()[2:]:
            enc2 = F.interpolate(enc2, size=enc3.size()[2:], mode='bilinear')

        print("size2: ", enc2.shape)
        print("size3: ", enc3.shape)
        print("size4: ", enc4.shape)
        
        enc5 = self.conv_block(enc2, enc3, enc4)

        enc4 = torch.cat((enc4, enc5),1)
        enc3 = torch.cat((enc3, enc5),1)
        enc2 = torch.cat((enc2, enc5),1)
        
        enc4 = F.relu(self.fuse1(enc4), inplace=True)
        enc3 = F.relu(self.fuse2(enc3), inplace=True)
        enc2 = F.relu(self.fuse3(enc2), inplace=True)
        
        # Decoder path
        d4 = self.decoder3(enc5, [enc4])
        d3 = self.decoder2(d4, [enc3])
        d2 = self.decoder1(d3, [enc2])
        d1 = torch.cat((d2, enc1),1)

        # Deep supervision
        out4 = F.interpolate(self.seg_heads[0](d4), size=x.shape[2:], mode='bilinear', align_corners=False)
        out3 = F.interpolate(self.seg_heads[1](d3), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_heads[2](d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_heads[3](d1), size=x.shape[2:], mode='bilinear', align_corners=False)

        final_out = (out1 + out2 + out3 + out4) / 4

        return [out1, out2, out3, out4], final_out

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
