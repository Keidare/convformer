from helpers import *
import torch.nn
import torch.nn.functional as F
import torch

class ConvFormer(nn.Module):
    def __init__(
            self,
            num_classes = 101,
            depths = [2,2,19,2],
            dims = [64, 128, 320, 512],
            layer_scale_init_value = 0,
            head_init_scale = 1.,
            drop_path_rate = 0.,
            downsample_kernels = [3,3,3,3],
            act_layer = nn.GELU
            ):
        super().__init__()
        self.downsamples = nn.ModuleList()
        self.stages = nn.ModuleList()
        stem_kernel = downsample_kernels[0]
        stem = nn.Sequential(
            Conv2d_LN(3, dims[0] // 2, stem_kernel, 2, padding=stem_kernel // 2),
            act_layer(),
            EdgeResidual(dims[0] // 2, dims[0], stem_kernel, 2, exp_ratio=4, act_layer=act_layer)
        )
        self.downsamples.append(stem)
        for i in range(3):
            downsample_layer = Downsample(dims[i], dims[i+1])
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

        # Third stage: 9 ConvBlocks, 9 AttnBlocks, 1 ConvBlock
        stage = nn.Sequential(
            *[ConvBlock(dim=dims[2], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(9)],
            *[AttnBlock(dim=dims[2], sr_ratio=8, head=5, dpr=dp_rates[cur + 9 + j]) for j in range(3)],
            ConvBlock(dim=dims[2], drop_path=dp_rates[cur + 18], layer_scale_init_value=layer_scale_init_value)
        )
        self.stages.append(stage)
        cur += 19

        # Last stage: 2 AttnBlocks
        stage = nn.Sequential(
            *[AttnBlock(dim=dims[3], sr_ratio=4, head=8, dpr=dp_rates[cur + j], final=(j == 1)) for j in range(2)]
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
            x = self.downsamples[i](x)  # Apply downsampling
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

              

# Sample input tensor with batch size 1, 3 channels (RGB), height 224, width 224
x = torch.randn(1, 3, 1024, 512)

# Instantiate the model
model = ConvFormer()

# Forward pass
output = model(x)
print(output.shape)  # Expected output shape: (1, 101)
