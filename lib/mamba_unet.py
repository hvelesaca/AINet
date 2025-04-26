import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
import timm
from lib.pvtv2 import pvt_v2_b2

# --- Aggregation Block ---
class AggregationBlock(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, features):
        x = torch.cat(features, dim=1)
        return self.conv(x)

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

# Ejemplo de Bloque Encoder Alternativo (Mamba -> CBAM)
class Mamba_CBAMEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mamba_dim=64):
        super().__init__()
        self.mamba_block = MambaConvBlock(in_channels, out_channels, mamba_dim=mamba_dim)
        self.cbam = CBAM(out_channels) # CBAM sobre la salida de Mamba

    def forward(self, x):
        x = self.mamba_block(x)
        return self.cbam(x) # Aplicar CBAM después

# Ejemplo de Bloque Decoder Alternativo (Conv -> CBAM)
class Mamba_CBAMDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Combinar skip y upsample ANTES de la convolución principal
        self.conv_block = MambaConvBlock(out_channels * 2, out_channels) # O un bloque convolucional estándar
        self.cbam = CBAM(out_channels) # CBAM sobre la salida del bloque principal

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x) # Procesamiento principal
        return self.cbam(x)    # Refinamiento con CBAM
        
# Modelo Completo con Deep Supervision y estructura U-Net
class CamouflageDetectionNetAnt(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        self.backbone = pvt_v2_b2()  
        if pretrained:
            self._load_backbone_weights('/kaggle/input/pretrained_pvt_v2_b2/pytorch/default/1/pvt_v2_b2.pth')      
                
        out_channels = [64, 128, 320, 512] #self.backbone.out_channels 

        # --- Encoder Path ---
        self.encoder1 = Mamba_CBAMEncoderBlock(out_channels[0], features[0])
        self.encoder2 = Mamba_CBAMEncoderBlock(out_channels[1], features[1])
        self.encoder3 = Mamba_CBAMEncoderBlock(out_channels[2], features[2])
        self.encoder4 = Mamba_CBAMEncoderBlock(out_channels[3], features[3])

        # --- Decoder Path ---
        self.decoder3 = Mamba_CBAMDecoderBlock(features[3], features[2])
        self.decoder2 = Mamba_CBAMDecoderBlock(features[2], features[1])
        self.decoder1 = Mamba_CBAMDecoderBlock(features[1], features[0])

        # --- Segmentation Heads (Deep Supervision) ---
        #self.dropout = nn.Dropout2d(p=dropout_prob) # Capa de Dropout
        self.seg_head3 = nn.Conv2d(features[2], 1, kernel_size=1) # Output from decoder3
        self.seg_head2 = nn.Conv2d(features[1], 1, kernel_size=1) # Output from decoder2
        self.seg_head1 = nn.Conv2d(features[0], 1, kernel_size=1) # Output from decoder1
        
        #self.encoders = nn.ModuleList([
        #    MambaConvBlock(out_channels[i], features[i]) for i in range(4)
        #])

        #self.decoders = nn.ModuleList([
        #    AttentionDecoderBlock(features[3], features[2]),
        #    AttentionDecoderBlock(features[2], features[1]),
        #    AttentionDecoderBlock(features[1], features[0])
        #])

        #self.seg_heads = nn.ModuleList([
        #    nn.Conv2d(features[i], 1, kernel_size=1) for i in range(3)
        #])

        # Fusión jerárquica aprendida
        #self.fusion_mlp = nn.Sequential(
        #    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(8),
        #    #nn.ReLU(inplace=True),
        #    nn.PReLU(),
        #    nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        #)
        
    def forward(self, x: torch.Tensor):
        skips = self.backbone.forward_features(x)
        #enc_feats = [enc(skip) for enc, skip in zip(self.encoders, skips)]

        #d3 = self.decoders[0](enc_feats[3], enc_feats[2])
        #d2 = self.decoders[1](d3, enc_feats[1])
        #d1 = self.decoders[2](d2, enc_feats[0])

        #out1 = F.interpolate(self.seg_heads[0](d1), size=x.shape[2:], mode='bilinear', align_corners=False)
        #out2 = F.interpolate(self.seg_heads[1](d2), size=x.shape[2:], mode='bilinear', align_corners=False)
        #out3 = F.interpolate(self.seg_heads[2](d3), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # --- Encoder ---
        skips = self.backbone.forward_features(x) # Obtener features del backbone [s1, s2, s3, s4]
        # Procesar skips con MambaConvBlocks
        enc1_out = self.encoder1(skips[0])
        enc2_out = self.encoder2(skips[1])
        enc3_out = self.encoder3(skips[2])
        enc4_out = self.encoder4(skips[3]) # Bottleneck feature

        # --- Decoder ---
        # Pasar la salida del encoder anterior y el skip correspondiente
        dec3_out = self.decoder3(enc4_out, enc3_out) # Input: Bottleneck, Skip: Encoder 3 output
        dec2_out = self.decoder2(dec3_out, enc2_out) # Input: Decoder 3 out, Skip: Encoder 2 output
        dec1_out = self.decoder1(dec2_out, enc1_out) # Input: Decoder 2 out, Skip: Encoder 1 output
        
        # Generar salidas de segmentación en diferentes niveles del decoder
        # Interpolar todas a la dimensión de la entrada original
        out3 = F.interpolate(self.seg_head3(dec3_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_head2(dec2_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_head1(dec1_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Combinar las salidas (puedes elegir solo out1 o una combinación)
        final_out = (out1 + out2 + out3) / 3 # Promedio 

        #fusion_input = torch.cat([out1, out2, out3], dim=1)  # [B, 3, H, W]
        #final_out = self.fusion_mlp(fusion_input)            # [B, 1, H, W]

        # Devolver todas las salidas y la final combinada
        return [out1, out2, out3], final_out

    def _load_backbone_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("✅ Pesos backbone cargados correctamente.")
        except Exception as e:
            print(f"❌ Error cargando pesos backbone: {e}")


# --- MultiScale Attention (TA + F-TA + B-TA combined) ---
class MultiScaleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ta = CBAM(channels)
        self.f_ta = CBAM(channels)
        self.b_ta = CBAM(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x):
        ta_out = self.ta(x)
        fta_out = self.f_ta(x)
        bta_out = self.b_ta(x)
        x = torch.cat([ta_out, fta_out, bta_out], dim=1)
        return self.conv(x)

# --- Decoder Block with Mask Flow ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()
        self.mamba_cbam = CBAM(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.mamba_cbam(x)
        return x

# --- Full Network ---
class CamouflageDetectionNet(nn.Module):
    def __init__(self, encoder_channels=[64, 128, 320, 512], decoder_channels=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        
        # Assume encoder is defined elsewhere (PVTv2 + Mamba + CBAM backbone)
        self.encoder = YourEncoder()

        # Decoder Blocks
        self.decoder4 = DecoderBlock(encoder_channels[3], encoder_channels[2], decoder_channels[3])
        self.decoder3 = DecoderBlock(decoder_channels[3], encoder_channels[1], decoder_channels[2])
        self.decoder2 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[1])
        self.decoder1 = DecoderBlock(decoder_channels[1], 0, decoder_channels[0])

        # Multi-Scale Attention Modules
        self.msa4 = MultiScaleAttention(decoder_channels[3])
        self.msa3 = MultiScaleAttention(decoder_channels[2])
        self.msa2 = MultiScaleAttention(decoder_channels[1])
        self.msa1 = MultiScaleAttention(decoder_channels[0])

        # Deep Supervision Heads
        self.pred4 = nn.Conv2d(decoder_channels[3], 1, 1)
        self.pred3 = nn.Conv2d(decoder_channels[2], 1, 1)
        self.pred2 = nn.Conv2d(decoder_channels[1], 1, 1)
        self.pred1 = nn.Conv2d(decoder_channels[0], 1, 1)

        # Aggregation Block
        self.aggregation = AggregationBlock(4, 1)

    def forward(self, x):
        # Encoder
        e1, e2, e3, e4 = self.encoder(x)

        # Decoder with Mask Flows and Multi-Scale Attention
        d4 = self.decoder4(e4, e3)
        d4 = self.msa4(d4)
        p4 = self.pred4(d4)

        d3 = self.decoder3(d4, e2)
        d3 = self.msa3(d3)
        p3 = self.pred3(d3)

        d2 = self.decoder2(d3, e1)
        d2 = self.msa2(d2)
        p2 = self.pred2(d2)

        d1 = self.decoder1(d2, None)
        d1 = self.msa1(d1)
        p1 = self.pred1(d1)

        # Upsample all outputs to input size
        p4 = F.interpolate(p4, size=x.size()[2:], mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=x.size()[2:], mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=x.size()[2:], mode='bilinear', align_corners=True)
        p1 = F.interpolate(p1, size=x.size()[2:], mode='bilinear', align_corners=True)

        # Aggregation
        out = self.aggregation([p1, p2, p3, p4])

        return [p1, p2, p3, p4], out
    def _load_backbone_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location='cpu')
            self.backbone.load_state_dict(state_dict, strict=False)
            print("✅ Pesos backbone cargados correctamente.")
        except Exception as e:
            print(f"❌ Error cargando pesos backbone: {e}")  
            
# --- Placeholder Encoder ---
class YourEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = pvt_v2_b2()  
        #if pretrained:
        self._load_backbone_weights('/kaggle/input/pretrained_pvt_v2_b2/pytorch/default/1/pvt_v2_b2.pth')      
                
        out_channels = [64, 128, 320, 512] #self.backbone.out_channels 

        # --- Encoder Path ---
        self.stage1 = Mamba_CBAMEncoderBlock(out_channels[0], features[0])
        self.stage2 = Mamba_CBAMEncoderBlock(out_channels[1], features[1])
        self.stage3 = Mamba_CBAMEncoderBlock(out_channels[2], features[2])
        self.stage4 = Mamba_CBAMEncoderBlock(out_channels[3], features[3])

    def forward(self, x):
        e1 = self.stage1(x)
        e2 = self.stage2(e1)
        e3 = self.stage3(e2)
        e4 = self.stage4(e3)
        return e1, e2, e3, e4
    
            
# Ejemplo de uso optimizado
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CamouflageDetectionNet(pretrained=True).to(device)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256).to(device)
        outputs, final_output = model(x)
        print(final_output.shape)  # [1, 1, 256, 256]
