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
        #self.backbone = pvt_v2_b2()
        #if pretrained:
            # Intenta cargar los pesos, maneja la excepción si falla
        #    self._load_backbone_weights('./pretrained_pvt/pvt_v2_b2.pth') # Ajusta la ruta si es necesario

        self.backbone = PVTBackbone("pvt_v2_b3", pretrained=True)
        out_channels = self.backbone.out_channels  # [96, 192, 384, 768]

        # --- Encoder Path ---
        # Módulos Mamba que procesan las salidas del backbone
        self.encoder1 = MambaConvBlock(out_channels[0], features[0])
        self.encoder2 = MambaConvBlock(out_channels[1], features[1])
        self.encoder3 = MambaConvBlock(out_channels[2], features[2])
        self.encoder4 = MambaConvBlock(out_channels[3], features[3])

        # Canales de salida esperados del backbone pvt_v2_b2 en cada etapa
        #pvt_channels = [64, 128, 320, 512]

        # --- Decoder Path ---
        # Módulos Decoder que reciben la salida del nivel anterior y el skip connection
        self.decoder3 = AttentionDecoderBlock(features[3], features[2]) # Up(enc4) + enc3
        self.decoder2 = AttentionDecoderBlock(features[2], features[1]) # Up(dec3) + enc2
        self.decoder1 = AttentionDecoderBlock(features[1], features[0]) # Up(dec2) + enc1

        # --- Segmentation Heads (Deep Supervision) ---
        self.seg_head3 = nn.Conv2d(features[2], 1, kernel_size=1) # Output from decoder3
        self.seg_head2 = nn.Conv2d(features[1], 1, kernel_size=1) # Output from decoder2
        self.seg_head1 = nn.Conv2d(features[0], 1, kernel_size=1) # Output from decoder1

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

        # --- Deep Supervision Heads ---
        # Generar salidas de segmentación en diferentes niveles del decoder
        # Interpolar todas a la dimensión de la entrada original
        out3 = F.interpolate(self.seg_head3(dec3_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out2 = F.interpolate(self.seg_head2(dec2_out), size=x.shape[2:], mode='bilinear', align_corners=False)
        out1 = F.interpolate(self.seg_head1(dec1_out), size=x.shape[2:], mode='bilinear', align_corners=False)

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
