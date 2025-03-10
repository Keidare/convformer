import torchsummary
from new_helpers import *
import torch.nn
import torch.nn.functional as F
import torch

class ConvFormer(nn.Module):
    def __init__(
            self,
            num_classes = 101,
            depths = [2,2,19,2],
            dims = [96, 192, 384, 768],
            layer_scale_init_value = 0,
            head_init_scale = 1.,
            drop_path_rate = 0.,
            img_width = 224,
            img_height = 224
            ):
        super().__init__()
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0,drop_path_rate, sum(depths))]
        cur = 0
        # First two stages with 2 ConvBlocks each
        for i in range(2):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(2)]
            )
            self.stages.append(stage)
            cur += 2

        # Third stage: 9 ConvBlocks, 3 AttnBlocks, 1 ConvBlock
        stage = nn.Sequential(
            *[Block(dim=dims[2], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(9)],
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 10], img_height= img_height, img_width= img_width),
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 13], img_height= img_height, img_width= img_width),
            AttnBlock(dim=dims[2], sr_ratio=2, head=4, dpr=dp_rates[cur + 16], img_height= img_height, img_width= img_width),
            Block(dim=dims[2], drop_path=dp_rates[cur + 18], layer_scale_init_value=layer_scale_init_value)
        )
        self.stages.append(stage)
        cur += 19

        # Last stage: 2 AttnBlocks
        stage = nn.Sequential(
            *[AttnBlock(dim=dims[3], sr_ratio=1, head=8, dpr=dp_rates[cur + j], img_height= img_height, img_width= img_width) for j in range(2)]
        )

        self.stages.append(stage)
        cur += 2

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # Final normalization layer
        self.head = nn.Linear(dims[-1], num_classes)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # print(f"Input Shape: {x.shape}")  # Check input shape

        for i in range(4):
            x = self.downsample_layers[i](x)  # Apply downsampling
            # print(f"After Downsampling {i}: {x.shape}")

            # Extract H, W if x is in (B, C, H, W) format
            if x.dim() == 4:
                B, C, H, W = x.shape
            else:
                H, W = None, None  # Default case for unexpected shape

            # Process each layer in the stage
            for j, layer in enumerate(self.stages[i]):
                prev_shape = x.shape  # Save shape before processing
                if isinstance(layer, AttnBlock):
                    x = layer(x, H, W)  # Pass spatial dimensions explicitly
                else:
                    x = layer(x)  # Standard ConvBlock processing
                
                # print(f"Layer {j} ({layer.__class__.__name__}) - Shape: {prev_shape} -> {x.shape}")

        # print(f"Before Global Avg Pooling: {x.shape}")
        x = self.norm(x.mean([-2, -1]))  # Global average pooling (B, C, H, W) -> (B, C)
        # print(f"After Global Avg Pooling: {x.shape}")

        out = self.head(x)
        # print(f"Final Output Shape: {out.shape}")

        return out
    
