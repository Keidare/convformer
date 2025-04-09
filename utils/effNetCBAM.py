import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# CBAM Module
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x) * x
        sa = self.spatial_attention(torch.cat([ca.mean(dim=1, keepdim=True), ca.max(dim=1, keepdim=True)[0]], dim=1)) * ca
        return sa

# Swish Activation
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Depthwise Separable Convolution
class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=kernel_size//2, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Swish()
        )

    def forward(self, x):
        return self.conv(x)

# MBConv Block with CBAM
class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion, stride, kernel_size=3, repeats=1):
        super(MBConv, self).__init__()
        layers = []
        for i in range(repeats):
            layers.append(self._make_mbconv(in_channels if i == 0 else out_channels, out_channels, expansion, stride if i == 0 else 1, kernel_size))
        self.block = nn.Sequential(*layers)

    def _make_mbconv(self, in_channels, out_channels, expansion, stride, kernel_size):
        mid_channels = in_channels * expansion
        return nn.Sequential(
            # 1x1 Expansion
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            Swish(),
            # Depthwise Convolution
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            Swish(),
            # CBAM Attention
            CBAM(mid_channels),
            # 1x1 Projection
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

# EfficientNet Architecture
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=1000, variant="b0"):
        super(EfficientNet_CBAM, self).__init__()

        # Define EfficientNet scaling factors
        params = {
            "b0": (1.0, 1.0, 224),
            "b1": (1.0, 1.1, 240)
        }
        width_mult, depth_mult, input_size = params[variant]

        # Base number of filters
        base_channels = int(32 * width_mult)
        last_channels = int(1280 * width_mult)

        # Stem Layer
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            Swish()
        )

        # EfficientNet Blocks (MBConv + CBAM)
        def round_repeats(repeats):
            return int(math.ceil(repeats * depth_mult))  # Apply depth scaling

        self.blocks = nn.Sequential(
            MBConv(base_channels, int(16 * width_mult), expansion=1, stride=1, repeats=round_repeats(1)),   # Stage 1
            MBConv(int(16 * width_mult), int(24 * width_mult), expansion=6, stride=2, repeats=round_repeats(2)),  # Stage 2
            MBConv(int(24 * width_mult), int(40 * width_mult), expansion=6, stride=2, repeats=round_repeats(2)),  # Stage 3
            MBConv(int(40 * width_mult), int(80 * width_mult), expansion=6, stride=2, repeats=round_repeats(3)),  # Stage 4
            MBConv(int(80 * width_mult), int(112 * width_mult), expansion=6, stride=1, repeats=round_repeats(3)), # Stage 5
            MBConv(int(112 * width_mult), int(192 * width_mult), expansion=6, stride=2, repeats=round_repeats(4)), # Stage 6
            MBConv(int(192 * width_mult), int(320 * width_mult), expansion=6, stride=1, repeats=round_repeats(1))  # Stage 7
        )

        # Final Layers
        self.head = nn.Sequential(
            nn.Conv2d(int(320 * width_mult), last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(last_channels, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
    
# Model Creation
def create_efficientnet_cbam(variant="b0", num_classes=2):
    return EfficientNet_CBAM(num_classes=num_classes, variant=variant)

