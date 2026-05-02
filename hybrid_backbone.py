import torch
import torch.nn as nn
import timm


# ===================== SAFE TRANSFORMER FUSION =====================
class TransformerFusion(nn.Module):
    def __init__(self, dim=512, heads=4):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, f1, f2):

        B, C, H, W = f1.shape

        # flatten spatial → tokens
        f1 = f1.flatten(2).transpose(1, 2)  # B, N, C
        f2 = f2.flatten(2).transpose(1, 2)

        x = torch.cat([f1, f2], dim=1)

        attn_out, _ = self.attn(x, x, x)

        x = self.norm(attn_out + x)

        # global representation
        return x.mean(dim=1)


# ===================== HYBRID MODEL =====================
class HybridModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # ===== BACKBONES (UNCHANGED) =====
        self.convnext = timm.create_model(
            "convnext_tiny", pretrained=True, features_only=True
        )

        self.effnet = timm.create_model(
            "efficientnet_b3", pretrained=True, features_only=True
        )

        # ===== PROJECTION (SAFE DIM ALIGNMENT) =====
        self.conv_proj = nn.Conv2d(768, 512, 1)
        self.eff_proj = nn.Conv2d(384, 512, 1)

        # ===== NEW FUSION =====
        self.fusion = TransformerFusion(dim=512)

        # ===== CLASSIFIER (UNCHANGED STYLE) =====
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        # ===== FEATURE EXTRACTION =====
        f1 = self.convnext(x)[-1]   # (B, 768, H, W)
        f2 = self.effnet(x)[-1]     # (B, 384, H, W)

        # ===== ALIGN DIM =====
        f1 = self.conv_proj(f1)
        f2 = self.eff_proj(f2)

        # ===== TRANSFORMER FUSION =====
        fused = self.fusion(f1, f2)

        # ===== CLASSIFICATION =====
        out = self.classifier(fused)

        return out