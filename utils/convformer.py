from helpers import *
import torch.nn
import torch.nn.functional as F
import torch
from convmitdecoder import SegFormerClassifier

class ConvFormer(nn.Module):
    def __init__(
            self,
            num_classes = 101,
            depths = [2,2,14,2],
            dims = [64, 128, 320, 512],
            layer_scale_init_value = 0,
            head_init_scale = 1.,
            drop_path_rate = 0.,
            downsample_kernels = [7,3,3,3],
            act_layer = nn.GELU
            ):
        super().__init__()
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        stem_kernel = downsample_kernels[0]
        stem = nn.Sequential(
            Conv2d_LN(3, dims[0] // 2, stem_kernel, 2, padding=stem_kernel // 2),
            act_layer(),
            EdgeResidual(dims[0] // 2, dims[0], stem_kernel, 2, exp_ratio=16, act_layer=act_layer)
        )
        self.downsamples.append(stem)
        for i in range(3):
            downsample_layer = Downsample(dims[i], dims[i+1], kernel=downsample_kernels[i+1])
            self.downsamples.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0,drop_path_rate, sum(depths))]
        cur = 0
        # First two stages with 2 ConvBlocks each
        for i in range(2):
            stage = nn.Sequential(
                *[ConvBlock(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(2)]
            )
            self.stages.append(stage)
            cur += 2

        # Third stage: 9 ConvBlocks, 3 AttnBlocks
        stage = nn.Sequential(
            *[ConvBlock(dim=dims[2], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(5)],
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 7]),
            *[ConvBlock(dim=dims[2], drop_path=dp_rates[cur + j + 8], layer_scale_init_value=layer_scale_init_value) for j in range(2)],
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 10]),
            *[ConvBlock(dim=dims[2], drop_path=dp_rates[cur + j + 11], layer_scale_init_value=layer_scale_init_value) for j in range(2)],
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 13]),
        )
        self.stages.append(stage)
        cur += 14

        # Last stage: 2 AttnBlocks
        stage = nn.Sequential(
            *[AttnBlock(dim=dims[3], sr_ratio=1, head=8, dpr=dp_rates[cur + j]) for j in range(2)]
        )

        self.stages.append(stage)
        cur += 2
        
        self.head = SegFormerClassifier(dims,num_classes)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final normalization layer
        # self.CLhead = nn.Linear(dims[-1], num_classes)
        # self.CLhead.weight.data.mul_(head_init_scale)
        # self.CLhead.bias.data.mul_(head_init_scale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = []  # Store multi-scale features

        for i in range(4):
            x = self.downsamples[i](x)  # Apply downsampling

            # Extract H, W if x is in (B, C, H, W) format
            if x.dim() == 4:
                B, C, H, W = x.shape
            else:
                H, W = None, None  # Default case for unexpected shape

            # Process each layer in the stage
            for j, layer in enumerate(self.stages[i]):
                if isinstance(layer, AttnBlock):
                    x = layer(x, H, W)  # Pass spatial dimensions explicitly
                else:
                    x = layer(x)  # Standard ConvBlock processing

            features.append(x)  # Store feature maps from each stage
        # x = self.norm(x.mean([-2, -1]))
        out = self.head(features)
        return out  # Return multi-scale feature maps

        # print(f"Final Output Shape: {out.shape}")


# model = ConvFormer(101)
# inp = torch.randn(1,3,64,64)
# x = model(inp).to('cuda')

# print(x.shape)

class ConvFormerBN(nn.Module):
    def __init__(
            self,
            num_classes = 101,
            depths = [2,2,14,2],
            dims = [64, 128, 320, 512],
            layer_scale_init_value = 0,
            head_init_scale = 1.,
            drop_path_rate = 0.,
            downsample_kernels = [7,3,3,3],
            act_layer = nn.GELU
            ):
        super().__init__()
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        stem_kernel = downsample_kernels[0]
        stem = nn.Sequential(
            Conv2d_BN(3, dims[0] // 2, stem_kernel, 2, padding=stem_kernel // 2),
            act_layer(),
            EdgeResidual(dims[0] // 2, dims[0], stem_kernel, 2, exp_ratio=8, act_layer=act_layer)
        )
        self.downsamples.append(stem)
        for i in range(3):
            downsample_layer = Downsample(dims[i], dims[i+1], kernel=downsample_kernels[i+1])
            self.downsamples.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0,drop_path_rate, sum(depths))]
        cur = 0
        # First two stages with 2 ConvBlocks each
        for i in range(2):
            stage = nn.Sequential(
                *[ConvBlockBN(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(2)]
            )
            self.stages.append(stage)
            cur += 2

        # Third stage: 9 ConvBlocks, 3 AttnBlocks
        stage = nn.Sequential(
            *[ConvBlockBN(dim=dims[2], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(5)],
            AttnBlockBN(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 7]),
            *[ConvBlockBN(dim=dims[2], drop_path=dp_rates[cur + j + 8], layer_scale_init_value=layer_scale_init_value) for j in range(2)],
            AttnBlockBN(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 10]),
            *[ConvBlockBN(dim=dims[2], drop_path=dp_rates[cur + j + 11], layer_scale_init_value=layer_scale_init_value) for j in range(2)],
            AttnBlockBN(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 13]),
        )
        self.stages.append(stage)
        cur += 13

        # Last stage: 2 AttnBlocks
        stage = nn.Sequential(
            *[AttnBlockBN(dim=dims[3], sr_ratio=1, head=8, dpr=dp_rates[cur + j]) for j in range(2)]
        )

        self.stages.append(stage)
        cur += 2
        
        self.head = SegFormerClassifier(dims,num_classes)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final normalization layer
        # self.CLhead = nn.Linear(dims[-1], num_classes)
        # self.CLhead.weight.data.mul_(head_init_scale)
        # self.CLhead.bias.data.mul_(head_init_scale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        features = []  # Store multi-scale features

        for i in range(4):
            x = self.downsamples[i](x)  # Apply downsampling

            # Extract H, W if x is in (B, C, H, W) format
            if x.dim() == 4:
                B, C, H, W = x.shape
            else:
                H, W = None, None  # Default case for unexpected shape

            # Process each layer in the stage
            for j, layer in enumerate(self.stages[i]):
                if isinstance(layer, AttnBlockBN):
                    x = layer(x, H, W)  # Pass spatial dimensions explicitly
                else:
                    x = layer(x)  # Standard ConvBlock processing

            features.append(x)  # Store feature maps from each stage
        # x = self.norm(x.mean([-2, -1]))
        out = self.head(features)
        return out  # Return multi-scale feature maps

        # print(f"Final Output Shape: {out.shape}")
