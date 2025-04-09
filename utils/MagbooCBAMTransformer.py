import torch
import torch.nn as nn
import torch.nn.functional as F
from helpers import *
import einops

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        scale = self.sigmoid(avg_out + max_out)
        x = x * scale
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = self.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * spatial_attention

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class CNNTransformer(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        
        self.flatten = nn.Conv2d(128, 256, kernel_size=1)  # Projection before transformer
        self.transformer = TransformerBlock(dim=256, num_heads=4)
        
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        
        x = self.flatten(x).flatten(2).permute(0, 2, 1)  # Flatten for transformer
        x = self.transformer(x).mean(dim=1)  # Global pooling
        
        return torch.sigmoid(self.fc(x))

class CNNTransformerMultiBlock(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_transformers=4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1) 
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        
        self.flatten = nn.Conv2d(128, 256, kernel_size=1)  # Projection before transformer
        # self.transformer = TransformerBlock(dim=256, num_heads=4)
        self.transformers = nn.Sequential(*[TransformerBlock(dim=256, num_heads=4) for _ in range(num_transformers)])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.cbam3(x)
        
        x = self.flatten(x).flatten(2).permute(0, 2, 1)  # Flatten for transformer
        x = self.transformers(x).mean(dim=1)  # Global pooling
        
        return torch.sigmoid(self.fc(x))

class CNNTransformerMultiCNNAndBlock(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_transformers=4):
        super().__init__()

        self.stem = nn.Sequential(
            Conv2d_BN(in_channels, 8, kernel_size=3, stride=1, padding=1),  
            nn.ReLU(inplace=True),
            Conv2d_BN(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.stage1_conv = nn.Sequential(
            Conv2d_BN(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage1_down = Conv2d_BN(16, 32, kernel_size=3, stride=2, padding=1)  # Downsampling
        self.cbam1 = CBAM(32)
        self.stage1_res = Conv2d_BN(16, 32, kernel_size=1, stride=2)  # Residual projection

        self.stage2_conv = nn.Sequential(
            Conv2d_BN(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage2_down = Conv2d_BN(32, 64, kernel_size=3, stride=2, padding=1)
        self.cbam2 = CBAM(64)
        self.stage2_res = Conv2d_BN(32, 64, kernel_size=1, stride=2)

        self.stage3_conv = nn.Sequential(
            Conv2d_BN(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            Conv2d_BN(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3_down = Conv2d_BN(64, 128, kernel_size=3, stride=2, padding=1)
        self.cbam3 = CBAM(128)
        self.stage3_res = Conv2d_BN(64, 128, kernel_size=1, stride=2)
        
        self.flatten = nn.Conv2d(128, 256, kernel_size=1)
        self.transformers = nn.Sequential(*[TransformerBlock(dim=256, num_heads=4) for _ in range(num_transformers)])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        
        res = x
        x = self.stage1_conv(x)
        x = self.stage1_down(x)
        res = self.stage1_res(res)
        x = self.cbam1(x + res)  # Residual connection
        
        res = x
        x = self.stage2_conv(x)
        x = self.stage2_down(x)
        res = self.stage2_res(res)
        x = self.cbam2(x + res)
        
        res = x
        x = self.stage3_conv(x)
        x = self.stage3_down(x)
        res = self.stage3_res(res)
        x = self.cbam3(x + res)
        
        x = self.flatten(x).flatten(2).permute(0, 2, 1)
        x = self.transformers(x).mean(dim=1)
        return torch.sigmoid(self.fc(x))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

class DSCCBAMTransformer(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_transformers=4):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, 32, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 32),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            CBAM(32)
        )
        
        self.conv2 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 64),
            nn.ReLU(),
            DepthwiseSeparableConv(64, 64),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            CBAM(64)
        )
        
        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128),
            nn.ReLU(),
            DepthwiseSeparableConv(128, 128),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            CBAM(128)
        )
        
        self.flatten = nn.Conv2d(128, 256, kernel_size=1)  # Projection before transformer
        self.transformers = nn.Sequential(*[TransformerBlock(dim=256, num_heads=4) for _ in range(num_transformers)])
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.flatten(x).flatten(2).permute(0, 2, 1)  # Flatten for transformer
        x = self.transformers(x).mean(dim=1)  # Global pooling
        
        return torch.sigmoid(self.fc(x))
    
class CNNTransformerTransformer(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, num_transformers_3=2, num_transformers_4=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        
        self.flatten_3 = nn.Conv2d(64, 128, kernel_size=1)  # Projection before 3rd transformer
        self.transformers_3 = nn.Sequential(*[TransformerBlock(dim=128, num_heads=4) for _ in range(num_transformers_3)])
        
        self.transformers_4 = nn.Sequential(*[TransformerBlock(dim=128, num_heads=4) for _ in range(num_transformers_4)])
        
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.cbam1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.cbam2(x)
        
        x = self.flatten_3(x).flatten(2).permute(0, 2, 1)  # Flatten for transformer
        x = self.transformers_3(x)
        
        x = self.transformers_4(x)  # Directly pass to 4th transformer
        
        x = x.mean(dim=1)  # Global pooling
        
        return torch.sigmoid(self.fc(x))