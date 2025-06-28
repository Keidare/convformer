from einops import rearrange

import torch
import torch.nn as nn


class Conv2x2(nn.Module):
    """
    2x2 Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Conv2x2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2,
            stride = stride, padding = 0, bias = True
        )

    def forward(self, x):
        result = self.conv(x)
        return result


class DWCONV(nn.Module):
    """
    Depthwise Convolution
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(DWCONV, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1, groups = in_channels, bias = True
        )

    def forward(self, x):
        result = self.depthwise(x)
        return result


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """
    def __init__(self, in_channels, out_channels):
        super(LPU, self).__init__()
        self.DWConv = DWCONV(in_channels, out_channels)

    def forward(self, x):
        result = self.DWConv(x) + x
        return result


class LMHSA(nn.Module):
    """
    Lightweight Multi-head-self-attention module.

    Inputs:
        Q: [N, C, H, W]
        K: [N, C, H / stride, W / stride]
        V: [N, C, H / stride, W / stride]
    Outputs:
        X: [N, C, H, W]
    """
    def __init__(self, input_size, channels, d_k, d_v, stride, heads, dropout):
        super(LMHSA, self).__init__()
        self.dwconv_k = DWCONV(channels, channels, stride = stride)
        self.dwconv_v = DWCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2
        self.B = nn.Parameter(torch.Tensor(1, self.heads, input_size ** 2, (input_size // stride) ** 2), requires_grad = True)


    def forward(self, x):
        b, c, h, w = x.shape

        # Reshape
        x_reshape = x.view(b, c, h * w).permute(0, 2, 1)
        # x_reshape = nn.LayerNorm(c).cuda()(x_reshape)
        x_reshape = torch.nn.functional.layer_norm(x_reshape, (b, h * w, c))

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, h * w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, h * w, d_k]

        k = self.dwconv_k(x)
        k_b, k_c, k_h, k_w = k.shape
        k = k.view(k_b, k_c, k_h * k_w).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_h * k_w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  # [b, heads, k_h * k_w, d_k]

        v = self.dwconv_v(x)
        v_b, v_c, v_h, v_w = v.shape
        v = v.view(v_b, v_c, v_h * v_w).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_h * v_w, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() # [b, heads, v_h * v_w, d_v]

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = attn + self.B
        attn = torch.softmax(attn, dim = -1) # [b, heads, h * w, k_h * k_w]

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, h * w, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, h, w)
        result = result + x
        return result


class IRFFN(nn.Module):
    """
    Inverted Residual Feed-forward Network
    """
    def __init__(self, in_channels, R):
        super(IRFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_channels, kernel_size = 1),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm2d(exp_channels),
            nn.GELU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result


class Patch_Aggregate(nn.Module):
    """
    Aggregate the patches into a single image.
    To produce the hierachical representation.

    Applied before each stage to reduce the size of intermediate features
    (2x downsampling of resolution), and project it to a larger dimension
    (2x enlargement of dimension).

    Input:
        - x: (B, In_C, H, W)
    Output:
        - x: (B, Out_C, H / 2, W / 2)
    """
    def __init__(self, in_channels, out_channels = None):
        super(Patch_Aggregate, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = Conv2x2(in_channels, out_channels, stride = 2)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LayerNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        _, c, h, w = x.size()
        # result = nn.LayerNorm((c, h, w)).cuda()(x)
        result = torch.nn.functional.layer_norm(x, (c, h, w))
        return result


class CMTStem(nn.Module):
    """
    Use CMTStem module to process input image and overcome the limitation of the
    non-overlapping patches.

    First past through the image with a 2x2 convolution to reduce the image size.
    Then past throught two 1x1 convolution for better local information.

    Input:
        - x: (B, 3, H, W)
    Output:
        - result: (B, 32, H / 2, W / 2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.gelu3(x)
        result = self.bn3(x)
        return result


class CMTBlock(nn.Module):
    def __init__(self, img_size, stride, d_k, d_v, num_heads, R = 3.6, in_channels = 46):
        super(CMTBlock, self).__init__()

        # Local Perception Unit
        self.lpu = LPU(in_channels, in_channels)

        # Lightweight MHSA
        self.lmhsa = LMHSA(img_size, in_channels, d_k, d_v, stride, num_heads, 0.0)

        # Inverted Residual FFN
        self.irffn = IRFFN(in_channels, R)

    def forward(self, x):
        x = self.lpu(x)
        x = self.lmhsa(x)
        x = self.irffn(x)
        return x

class CMT(nn.Module):
    def __init__(self,
        in_channels = 3,
        stem_channel = 32,
        cmt_channel = [46, 92, 184, 368],
        patch_channel = [46, 92, 184, 368],
        block_layer = [2, 2, 10, 2],
        R = 3.6,
        img_size = 224,
        num_class = 10
    ):
        super(CMT, self).__init__()

        # Image size for each stage
        size = [img_size // 4, img_size // 8, img_size // 16, img_size // 32]

        # Stem layer
        self.stem = CMTStem(in_channels, stem_channel)

        # Patch Aggregation Layer
        self.patch1 = Patch_Aggregate(stem_channel, patch_channel[0])
        self.patch2 = Patch_Aggregate(patch_channel[0], patch_channel[1])
        self.patch3 = Patch_Aggregate(patch_channel[1], patch_channel[2])
        self.patch4 = Patch_Aggregate(patch_channel[2], patch_channel[3])

        # CMT Block Layer
        stage1 = []
        for _ in range(block_layer[0]):
            cmt_layer = CMTBlock(
                    img_size = size[0],
                    stride = 8,
                    d_k = cmt_channel[0],
                    d_v = cmt_channel[0],
                    num_heads = 1,
                    R = R,
                    in_channels = patch_channel[0]
            )
            stage1.append(cmt_layer)
        self.stage1 = nn.Sequential(*stage1)

        stage2 = []
        for _ in range(block_layer[1]):
            cmt_layer = CMTBlock(
                    img_size = size[1],
                stride = 4,
                d_k = cmt_channel[1] // 2,
                d_v = cmt_channel[1] // 2,
                num_heads = 2,
                R = R,
                in_channels = patch_channel[1]
            )
            stage2.append(cmt_layer)
        self.stage2 = nn.Sequential(*stage2)

        stage3 = []
        for _ in range(block_layer[2]):
            cmt_layer = CMTBlock(
                img_size = size[2],
                stride = 2,
                d_k = cmt_channel[2] // 4,
                d_v = cmt_channel[2] // 4,
                num_heads = 4,
                R = R,
                in_channels = patch_channel[2]
            )
            stage3.append(cmt_layer)
        self.stage3 = nn.Sequential(*stage3)

        stage4 = []
        for _ in range(block_layer[3]):
            cmt_layer = CMTBlock(
                img_size = size[3],
                stride = 1,
                d_k = cmt_channel[3] // 8,
                d_v = cmt_channel[3] // 8,
                num_heads = 8,
                R = R,
                in_channels = patch_channel[3]
            )
            stage4.append(cmt_layer)
        self.stage4 = nn.Sequential(*stage4)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # FC
        self.fc = nn.Sequential(
            nn.Linear(cmt_channel[3], 1280),
            nn.ReLU(inplace = True),
        )

        # Final Classifier
        self.classifier = nn.Linear(1280, num_class)


    def forward(self, x):
        x = self.stem(x)

        x = self.patch1(x)
        x = self.stage1(x)

        x = self.patch2(x)
        x = self.stage2(x)

        x = self.patch3(x)
        x = self.stage3(x)

        x = self.patch4(x)
        x = self.stage4(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        logit = self.classifier(x)
        return logit


def CMT_Ti(img_size = 224, num_class = 1000):
    model = CMT(
        in_channels = 3,
        stem_channel = 16,
        cmt_channel = [46, 92, 184, 368],
        patch_channel = [46, 92, 184, 368],
        block_layer = [2, 2, 10, 2],
        R = 3.6,
        img_size = img_size,
        num_class = num_class
    )
    return model


def CMT_XS(img_size = 224, num_class = 1000):
    model = CMT(
        in_channels = 3,
        stem_channel = 16,
        cmt_channel = [52, 104, 208, 416],
        patch_channel = [52, 104, 208, 416],
        block_layer = [3, 3, 12, 3],
        R = 3.8,
        img_size = img_size,
        num_class = num_class
    )
    return model

def CMT_S(img_size = 224, num_class = 2):
    model = CMT(
        in_channels = 1,
        stem_channel = 32,
        cmt_channel = [64, 128, 256, 512],
        patch_channel = [64, 128, 256, 512],
        block_layer = [3, 3, 16, 3],
        R = 4,
        img_size = img_size,
        num_class = num_class
    )
    return model

def CMT_B(img_size = 224, num_class = 1000):
    model = CMT(
        in_channels = 3,
        stem_channel = 38,
        cmt_channel = [76, 152, 304, 608],
        patch_channel = [76, 152, 304, 608],
        block_layer = [4, 4, 20, 4],
        R = 4,
        img_size = img_size,
        num_class = num_class
    )
    return model


