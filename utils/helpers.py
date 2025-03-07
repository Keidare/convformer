import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import re

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        l = (a - mean) / std
        u = (b - mean) / std
        tensor.copy_(torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=l, b=u))
    return tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                dilation, 
                groups, 
                bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding, 
                dilation, 
                groups, 
                bias
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
import torch
import torch.nn as nn

class Conv2d_LN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, 
                 dilation=1, groups=1, ln_eps=1e-6):
        super().__init__()
        
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, 
            dilation, groups, bias=False
        ))

        # LayerNorm on channels
        self.add_module('ln', nn.LayerNorm(out_channels, eps=ln_eps))

    def forward(self, x):
        x = self.conv(x)  # Apply Conv2D
        x = x.permute(0, 2, 3, 1)  # Change to (B, H, W, C) for LayerNorm
        x = self.ln(x)  # Apply LayerNorm
        x = x.permute(0, 3, 1, 2)  # Change back to (B, C, H, W)
        return x



class Residual(torch.nn.Module):
    def __init__(self, m, drop_path=0., layer_scale_init_value=0, dim=None):
        super().__init__()
        self.m = m
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)), 
                                      requires_grad=True)
        else:
            self.gamma = None

    def forward(self, x):
        if self.gamma is not None:
            return x + self.gamma * self.drop_path(self.m(x))
        else:
            return x + self.drop_path(self.m(x))
        

class FFN2d(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 ratio=4,
                 act_layer=nn.GELU,
                 **kargs):
        super().__init__()
        mid_chs = ratio * dim
        self.channel_mixer = Residual(nn.Sequential(
            Conv2d_LN(dim, mid_chs),
            act_layer(),
            Conv2d_LN(mid_chs, dim)),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=dim,
        )

    def forward(self, x):
        if isinstance(x, tuple):
            return self.channel_mixer(x[0]), x[1]
        else:
            return self.channel_mixer(x)

class ConvBlock(nn.Module):
    def __init__(self,
                 dim,
                 out_dim=None,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 kernel=7,
                 stride=1,
                 ratio=4,
                 act_layer=nn.GELU,
                 reparameterize=False,
                 **kargs):
        super().__init__()
        mid_chs = ratio * dim
        if out_dim is None:
            out_dim = dim
        if reparameterize:
            assert stride == 1
            dw_conv = eval(reparameterize)(dim, kernel)
        else:
            dw_conv = Conv2d_LN(dim, dim, kernel, stride, padding=kernel // 2, groups=dim)  # depthwise conv
            
        self.token_channel_mixer = Residual(nn.Sequential(
            dw_conv,
            Conv2d_LN(dim, mid_chs),
            act_layer(),
            Conv2d_LN(mid_chs, out_dim)),
            drop_path=drop_path,
            layer_scale_init_value=layer_scale_init_value,
            dim=out_dim,
        )

    def forward(self, x):
        if isinstance(x, tuple):
            out = self.token_channel_mixer(x[0])
            # print(f"ConvBlock Output Shape: {out.shape}")  # Debugging Output
            return out, x[1]
        
        else:
            out = self.token_channel_mixer(x)
            # print(f"ConvBlock Output Shape: {out.shape}")  # Debugging Output
            return out
        
class RepCPE(nn.Module):
    def __init__(self, dim, kernel=7, **kargs):
        super().__init__()
        self.cpe = Residual(
            Conv2d_LN(dim, dim, kernel, 1, padding=kernel//2, groups=dim)  # depthwise conv with LN
        )

    def forward(self, x):
        # print(f"[RepCPE] Input Shape: {x[0].shape if isinstance(x, tuple) else x.shape}")  # Debug
        out = self.cpe(x[0]) if isinstance(x, tuple) else self.cpe(x)
        # print(f"[RepCPE] Output Shape: {out.shape}")  # Debug
        return (out, x[1]) if isinstance(x, tuple) else out

class Attention(nn.Module):
    def __init__(self, dim, head, sr_ratio):
        super().__init__()
        self.head = head
        self.sr_ratio = sr_ratio
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        # print(f"[Attention] Input Shape: {x.shape}")  # Debug

        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            # print(f"[Attention] Shape Before SR: {x.shape}")  # Debug
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            # print(f"[Attention] Shape After SR: {x.shape}")  # Debug

        k, v = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        # print(f"[Attention] Output Shape: {x.shape}")  # Debug
        return x

class MLP(nn.Module):
    def __init__(self, c1, c2, drop = 0.):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.fc2 = nn.Linear(c2, c1)
        self.drop = nn.Dropout(drop)

        
    def forward(self, x: Tensor, H, W) -> Tensor:
        # print(f"[MLP] Input Shape: {x.shape}")  # Debug
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = F.gelu(x)
        x = self.drop(x)
        out = self.fc2(x)
        out = self.drop(out)
        # print(f"[MLP] Output Shape: {out.shape}")  # Debug
        return out

    
    
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)

class AttnBlock(nn.Module):
    def __init__(self, dim, head, sr_ratio=1, dpr=0., drop = 0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head, sr_ratio)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * 4), drop=drop)

    def forward(self, x: Tensor, H, W) -> Tensor:
        # Apply RepCPE (Input: B, C, H, W)

        # Flatten to (B, N, C) for Attention, where N = H * W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # Reshape (B, C, H*W) -> (B, N, C)

        # print(f"[AttnBlock] Shape Before Attention: {x.shape}")  # Debug
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # print(f"[AttnBlock] Shape Before MLP: {x.shape}")  # Debug
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        # print(f"[AttnBlock] Shape After MLP: {x.shape}")  # Debug

        # Fix: Reshape back to (B, C, H, W) before returning
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(f"[AttnBlock] Reshaped Back to: {x.shape}")  # Debug

        return x



class EdgeResidual(nn.Module):
    """ FusedIB in MobileNetV4-like architectures, adapted to use LayerNorm instead of BatchNorm.
    """
    def __init__(self,
                 in_chs: int,
                 out_chs: int,
                 exp_kernel_size=3,
                 stride=1,
                 exp_ratio=1.0,
                 act_layer=nn.ReLU,
                 ):
        super(EdgeResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        
        self.conv_exp_ln1 = Conv2d_LN(in_chs, mid_chs, exp_kernel_size, stride, padding=exp_kernel_size//2)
        self.act = act_layer()
        self.conv_pwl_ln2 = Conv2d_LN(mid_chs, out_chs, 1)

    def forward(self, x):
        x = self.conv_exp_ln1(x)
        x = self.act(x)
        x = self.conv_pwl_ln2(x)
        return x


class Classfier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        x = self.classifier(x)
        return x
    

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)




class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.downsample = nn.Sequential(
            LayerNorm(in_dim, eps=1e-6, data_format="channels_first"),  # Channel-first LayerNorm
            nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)  # 2x2 strided convolution
        )

    def forward(self, x):
        return self.downsample(x)
    
