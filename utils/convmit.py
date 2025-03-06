import torch
from torch import nn, Tensor
from torch.nn import functional as F
from drop import DropPath
from convmitdecoder import SegFormerClassifier
class ViTClassifierHead(nn.Module):
    def __init__(self, in_features, num_classes=101):
        super(ViTClassifierHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)

# replace with sepvit/ssa/add conv stem
def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv

class LinearSelfAttention(nn.Module):
    def __init__(self, embed_dim, attn_dropout=0):
        super().__init__()
        self.qkv_proj = conv_2d(embed_dim, 1+2*embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1)
        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio 
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            
        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        
    def forward(self, x: Tensor, H, W) -> Tensor:
        return self.fc2(F.gelu(self.dwconv(self.fc1(x), H, W)))


    
class Block(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*4))

    def forward(self, x: Tensor, H, W) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class ConvMiTBlock(nn.Module):
    def __init__(self, dim, H, W, attn_dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearSelfAttention(dim, attn_dropout)
        self.drop_path = DropPath(attn_dropout) if attn_dropout > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4))
        self.H, self.W = H, W  # Store spatial dimensions

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, N, C = x.shape
        assert N == H * W, f"Expected N = H * W, but got N={N}, H={H}, W={W}"
        
        # Normalize first, then reshape to 4D
        x = self.norm1(x)
        x = x.transpose(1, 2).reshape(B, C, H, W)  # Reshape for attention
        x = x + self.drop_path(self.attn(x))
        
        # Flatten back to 2D before passing to MLP
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


convmit_settings = {
    'B0': [[32, 64, 160, 256], [2, 2, 2, 2]],        
    'B1': [[64, 128, 320, 512], [2, 2, 2, 2]],
    'B2': [[64, 128, 320, 512], [3, 4, 6, 3]],
    'B3': [[64, 128, 320, 512], [3, 4, 18, 3]],
    'B4': [[64, 128, 320, 512], [3, 8, 27, 3]],
    'B5': [[64, 128, 320, 512], [3, 6, 40, 3]]
}

class ConvMiTPatchEmbed(nn.Module):
    def __init__(self, c1=3, c2=64, patch_size=4, stride=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=3, padding=1),  # Single downsampling conv
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),  # Feature extraction
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1),  # Additional feature extraction
        )
        self.norm = nn.LayerNorm(c2)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x = self.norm(x)
        return x, H, W


class ConvMiT(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes = 102):
        super().__init__()
        assert model_name in convmit_settings.keys(), f"ConvMiT model name should be in {list(convmit_settings.keys())}"
        embed_dims, depths = convmit_settings[model_name]
        drop_path_rate = 0.1
        self.channels = embed_dims

        # patch_embed
        self.patch_embed1 = ConvMiTPatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = ConvMiTPatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = ConvMiTPatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = ConvMiTPatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([ConvMiTBlock(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([ConvMiTBlock(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([ConvMiTBlock(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([ConvMiTBlock(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])


        # Classification head
        self.classifier = SegFormerClassifier(embed_dims, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Pass feature maps to classifier
        logits = self.classifier((x1, x2, x3, x4))

        return logits  # Shape: [B, num_classes]


class MiT(nn.Module):
    def __init__(self, model_name: str = 'B0', num_classes = 101):
        super().__init__()
        assert model_name in convmit_settings.keys(), f"MiT model name should be in {list(convmit_settings.keys())}"
        embed_dims, depths = convmit_settings[model_name]
        drop_path_rate = 0.1
        self.channels = embed_dims  

        # patch_embed
        self.patch_embed1 = ConvMiTPatchEmbed(3, embed_dims[0], 7, 4)
        self.patch_embed2 = ConvMiTPatchEmbed(embed_dims[0], embed_dims[1], 3, 2)
        self.patch_embed3 = ConvMiTPatchEmbed(embed_dims[1], embed_dims[2], 3, 2)
        self.patch_embed4 = ConvMiTPatchEmbed(embed_dims[2], embed_dims[3], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        cur = 0
        self.block1 = nn.ModuleList([Block(embed_dims[0], 1, 8, dpr[cur+i]) for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(embed_dims[1], 2, 4, dpr[cur+i]) for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(embed_dims[2], 5, 2, dpr[cur+i]) for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(embed_dims[3], 8, 1, dpr[cur+i]) for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])
        self.classifier = SegFormerClassifier(embed_dims, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x1 = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 2
        x, H, W = self.patch_embed2(x1)
        for blk in self.block2:
            x = blk(x, H, W)
        x2 = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 3
        x, H, W = self.patch_embed3(x2)
        for blk in self.block3:
            x = blk(x, H, W)
        x3 = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # stage 4
        x, H, W = self.patch_embed4(x3)
        for blk in self.block4:
            x = blk(x, H, W)
        x4 = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2)

        # Pass feature maps to classifier
        logits = self.classifier((x1, x2, x3, x4))

        return logits  # Shape: [B, num_classes]