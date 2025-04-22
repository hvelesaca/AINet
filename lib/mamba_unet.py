import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from huggingface_hub import hf_hub_download
import timm

#convnextv2_base, convnextv2_small, convnextv2_tiny
# ConvNeXt Tiny devuelve 4 features: [96, 192, 384, 768]
# ConvNeXt Small devuelve 4 features: [96, 192, 384, 768]
# ConvNeXt Base devuelve 4 features: [128, 256, 512, 1024]
class ConvNeXtBackbone(nn.Module):
    def __init__(self, model_name="convnext_base", pretrained=True):
        super().__init__()
        try:
          self.backbone = timm.create_model(model_name, features_only=True, pretrained=pretrained)
          print(f"Modelo {model_name} cargado exitosamente.")

          self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]
          print(f"Canales de salida de {model_name}: {self.out_channels}")

        except Exception as e:
          print(f"Error al cargar {model_name}: {e}")
          print("Asegúrate de tener 'timm' instalado y que el nombre del modelo sea correcto.")

    def forward_features(self, x):
        return self.backbone(x)
        
        
#pvt_v2_variants = [#'pvt_v2_b0',#'pvt_v2_b1',#'pvt_v2_b2', # La que usaste#'pvt_v2_b3',#'pvt_v2_b4',#'pvt_v2_b5',#]
class PVTBackbone(nn.Module):
    def __init__(self, model_name="pvt_v2_b2", pretrained=True):
        super().__init__()
        
        try:
          # Crear el backbone usando timm
          self.backbone = timm.create_model(model_name, features_only=True, pretrained=pretrained)# ¡Importante para obtener skips!
          print(f"Modelo {model_name} cargado exitosamente.")

          # Obtener los canales de salida (importante para conectar a los encoders)
          #self.out_channels = [f['num_chs'] for f in self.backbone.feature_info]
          self.out_channels = [64, 128, 320, 512]
          print(f"Canales de salida de {model_name}: {self.out_channels}")

        except Exception as e:
          print(f"Error al cargar {model_name}: {e}")
          print("Asegúrate de tener 'timm' instalado y que el nombre del modelo sea correcto.")

    def forward_features(self, x):
        return self.backbone(x)

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


#https://github.com/Peachypie98/CBAM
class CBAM2(nn.Module):
    def __init__(self, channels, reduction: int = 16):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = reduction
        self.sam = SAM(bias=False)
        self.cam = CAM(channels=self.channels, r=self.r)

    def forward(self, x):
        output = self.cam(x)
        output = self.sam(output)
        return output + x

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = F.sigmoid(output) * x
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = F.sigmoid(output) * x 
        return output 
        
# CBAM Attention Module
class CBAMOrin(nn.Module):
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
            nn.ReLU(inplace=True),
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

class CBAM3(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.PReLU(),
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

class MambaConvBlockPRELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, mamba_dim: int = 64):
        super().__init__()
        self.prelu = nn.PReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
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
        return F.prelu(x_mamba + identity, self.prelu.weight)
        
class CBAM_MambaEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mamba_dim=64):
        super().__init__()
        self.cbam = CBAM(in_channels)
        self.mamba_block = MambaConvBlock(in_channels, out_channels, mamba_dim=mamba_dim)

    def forward(self, x):
        return self.cbam(x)
        #return self.mamba_block(x)

class CBAM_MambaDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.cbam = CBAM(out_channels * 2)
        self.mamba_block = MambaConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.cbam(x)
        #return self.mamba_block(x)

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
            #nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
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

class CamouflageDetectionNet(nn.Module):
    def __init__(self, features=[64, 128, 320, 512], pretrained=True, dropout_prob=0.1):
        super().__init__()
        
        self.backbone = PVTBackbone("pvt_v2_b2", pretrained=pretrained)
        #self.backbone = ConvNeXtBackbone("convnext_small", pretrained=True)

        out_channels = self.backbone.out_channels  # [64, 128, 320, 512]

        self.encoder1 = Mamba_CBAMEncoderBlock(out_channels[0], features[0])
        self.encoder2 = Mamba_CBAMEncoderBlock(out_channels[1], features[1])
        self.encoder3 = Mamba_CBAMEncoderBlock(out_channels[2], features[2])
        self.encoder4 = Mamba_CBAMEncoderBlock(out_channels[3], features[3])

        self.decoder3 = Mamba_CBAMDecoderBlock(features[3], features[2])
        self.decoder2 = Mamba_CBAMDecoderBlock(features[2], features[1])
        self.decoder1 = Mamba_CBAMDecoderBlock(features[1], features[0])

        # --- Capa de Dropout ---
        # Usar nn.Dropout2d para mapas de características
        # Se aplicará antes de cada seg_head si dropout_prob > 0
        #self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        
        # --- Deep Supervision Heads ---
        self.seg_head3 = nn.Conv2d(features[2], 1, kernel_size=1)
        self.seg_head2 = nn.Conv2d(features[1], 1, kernel_size=1)
        self.seg_head1 = nn.Conv2d(features[0], 1, kernel_size=1)
        
        # Fusión jerárquica aprendida
        #self.fusion_mlp = nn.Sequential(
        #    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(8),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        #)

        # --- Refinamiento final con Mamba ---
        #self.refine_mamba = MambaConvBlock(1, 1)

    def forward(self, x: torch.Tensor):
        # --- Encoder ---
        skips = self.backbone.forward_features(x)

        enc1_out = self.encoder1(skips[0])
        enc2_out = self.encoder2(skips[1])
        enc3_out = self.encoder3(skips[2])
        enc4_out = self.encoder4(skips[3])

        # --- Decoder ---
        dec3_out = self.decoder3(enc4_out, enc3_out)
        dec2_out = self.decoder2(dec3_out, enc2_out)
        dec1_out = self.decoder1(dec2_out, enc1_out)

        # --- Aplicar Dropout ANTES de las cabezas de segmentación ---
        # Dropout solo se activa durante model.train()
        #dec3_out = self.dropout(dec3_out)
        #dec2_out = self.dropout(dec2_out)
        #dec1_out = self.dropout(dec1_out)

        # --- Deep Supervision Heads ---
        out3 = F.interpolate(self.seg_head3(dec3_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_head2(dec2_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_head1(dec1_out), size=x.shape[2:], mode='bilinear', align_corners=False)

        # --- Fusión Jerárquica ---
        #fusion_input = torch.cat([out1, out2, out3], dim=1)  # [B, 3, H, W]
        #final_out = self.fusion_mlp(fusion_input)            # [B, 1, H, W]

        # Combinar las salidas (puedes elegir solo out1 o una combinación)
        final_out = (out1 + out2 + out3) / 3 # Promedio

        # --- Refinamiento final con Mamba ---
        #final_out = self.refine_mamba(final_out)

        return [out1, out2, out3], final_out


# Modelo Completo con Deep Supervision y estructura U-Net
class CamouflageDetectionNet2(nn.Module):
    def __init__(self, features=[64, 128, 256, 512], pretrained=True):
        super().__init__()
        
        self.backbone = PVTBackbone("pvt_v2_b2", pretrained=True)
        #self.backbone = ConvNeXtBackbone("convnext_small", pretrained=True)

        out_channels = self.backbone.out_channels  # [96, 192, 384, 768]

        # --- Encoder Path ---
        # Módulos Mamba que procesan las salidas del backbone
        self.encoder1 = MambaConvBlock(out_channels[0], features[0])
        self.encoder2 = MambaConvBlock(out_channels[1], features[1])
        self.encoder3 = MambaConvBlock(out_channels[2], features[2])
        self.encoder4 = MambaConvBlock(out_channels[3], features[3])

        # --- Decoder Path ---
        # Módulos Decoder que reciben la salida del nivel anterior y el skip connection
        self.decoder3 = AttentionDecoderBlock(features[3], features[2]) # Up(enc4) + enc3
        self.decoder2 = AttentionDecoderBlock(features[2], features[1]) # Up(dec3) + enc2
        self.decoder1 = AttentionDecoderBlock(features[1], features[0]) # Up(dec2) + enc1

        # --- Segmentation Heads (Deep Supervision) ---
        #self.dropout = nn.Dropout2d(p=dropout_prob) # Capa de Dropout
        self.seg_head3 = nn.Conv2d(features[2], 1, kernel_size=1) # Output from decoder3
        self.seg_head2 = nn.Conv2d(features[1], 1, kernel_size=1) # Output from decoder2
        self.seg_head1 = nn.Conv2d(features[0], 1, kernel_size=1) # Output from decoder1

        # Fusión jerárquica aprendida
        #self.fusion_mlp = nn.Sequential(
        #    nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, bias=False),
        #    nn.BatchNorm2d(8),
        #    nn.ReLU(inplace=True),
        #    nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1)
        #)
       
    def forward(self, x: torch.Tensor):
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

        # --- Fusión Jerárquica ---
        #fusion_input = torch.cat([out1, out2, out3], dim=1)  # [B, 3, H, W]
        #final_out = self.fusion_mlp(fusion_input)            # [B, 1, H, W]
        
        # Combinar las salidas (puedes elegir solo out1 o una combinación)
        final_out = (out1 + out2 + out3) / 3 # Promedio 

        # Devolver todas las salidas y la final combinada
        return [out1, out2, out3], final_out
    
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
