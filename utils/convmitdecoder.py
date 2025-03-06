import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class MLP(nn.Module):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class ConvModule(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activate = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activate(self.bn(self.conv(x)))

class SegFormerHead(nn.Module):
    def __init__(self, dims: list, embed_dim: int = 256, num_classes: int = 19):
        super().__init__()
        self.linear_layers = nn.ModuleList([MLP(dim, embed_dim) for dim in dims])
        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        B, _, H, W = features[0].shape  # Get final upsampling target size (112,112)
        outs = []

        for i, feature in enumerate(features):
            x = self.linear_layers[i](feature).permute(0, 2, 1).reshape(B, -1, *feature.shape[-2:])
            if i > 0:  # Upsample all except the first feature map
                x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
            outs.append(x)

        seg = self.linear_fuse(torch.cat(outs, dim=1))
        seg = self.linear_pred(self.dropout(seg))
        return seg

class SegFormerClassifier(nn.Module):
    def __init__(self, embed_dims, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in embed_dims])  # LayerNorm per stage
        self.fc = nn.Linear(sum(embed_dims), num_classes)  # Fully Connected Layer

    def forward(self, features):
        x1, x2, x3, x4 = features  # Multi-scale features

        # Reshape to (B, N, C) before LayerNorm
        x1 = self.norms[0](x1.flatten(2).transpose(1, 2))  # (B, N, C)
        x2 = self.norms[1](x2.flatten(2).transpose(1, 2))  # (B, N, C)
        x3 = self.norms[2](x3.flatten(2).transpose(1, 2))  # (B, N, C)
        x4 = self.norms[3](x4.flatten(2).transpose(1, 2))  # (B, N, C)

        # Reshape back to (B, C, H, W) before pooling
        B, N1, C1 = x1.shape
        x1 = self.pool(x1.transpose(1, 2).reshape(B, C1, int(N1**0.5), int(N1**0.5))).flatten(1)

        B, N2, C2 = x2.shape
        x2 = self.pool(x2.transpose(1, 2).reshape(B, C2, int(N2**0.5), int(N2**0.5))).flatten(1)

        B, N3, C3 = x3.shape
        x3 = self.pool(x3.transpose(1, 2).reshape(B, C3, int(N3**0.5), int(N3**0.5))).flatten(1)

        B, N4, C4 = x4.shape
        x4 = self.pool(x4.transpose(1, 2).reshape(B, C4, int(N4**0.5), int(N4**0.5))).flatten(1)

        # Concatenate all scales
        x = torch.cat([x1, x2, x3, x4], dim=1)

        # Classification
        logits = self.fc(x)
        return logits  # Shape: [B, num_classes]


