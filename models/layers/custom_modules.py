# my_model/layers/custom_modules.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from .utils import dist2bbox, make_anchors



def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(BaseModule):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, init_cfg=None):
        super().__init__(init_cfg)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Conv):
    """Depth-wise convolution."""
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True, init_cfg=None): 
        super().__init__(c1, c2, k, s, p=p, g=math.gcd(c1, c2), d=d, act=act, init_cfg=init_cfg) 

class ContextPoolMixer(BaseModule):
    def __init__(self, c1, c2, k=5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))

class Bottleneck(BaseModule):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class DualConv(BaseModule):
    """Dual-branch convolution used inside HFE blocks."""

    def __init__(self, in_channels, out_channels, stride=1, g=4, init_cfg=None):
        super().__init__(init_cfg)
        self.gc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=g,
            bias=False)
        self.pwc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            bias=False)

    def forward(self, x):
        return self.gc(x) + self.pwc(x)


class EDLAN(BaseModule):
    """Efficient dual-layer aggregation block."""

    def __init__(self, channels, g=4, init_cfg=None):
        super().__init__(init_cfg)
        self.m = nn.Sequential(
            DualConv(channels, channels, 1, g=g),
            DualConv(channels, channels, 1, g=g))

    def forward(self, x):
        return self.m(x)


class HFEBlock(BaseModule):
    """Hierarchical feature enhancement block."""

    def __init__(self, c1, c2, n=1, g=4, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(EDLAN(self.c, g=g) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
        

class HGStem(BaseModule):
    def __init__(self, c1, c2, init_cfg=None):
        super().__init__(init_cfg)
        cm = c2
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class LightConv(BaseModule):
    def __init__(self, c1, c2, k=1, act=nn.ReLU(), init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))
        
class HGBlock(BaseModule):
    def __init__(self, c1, cm, c2, k=3, n=6, light=False, shortcut=False, act=nn.ReLU(), init_cfg=None):
        super().__init__(init_cfg)
        block = LightConv if light else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act) 
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y

class FGlo(BaseModule):
    def __init__(self, channel, reduction=16, init_cfg=None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        inter_channel = max(1, channel // reduction)
        self.fc = nn.Sequential(
                nn.Linear(channel, inter_channel), 
                nn.ReLU(inplace=True),
                nn.Linear(inter_channel, channel), 
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MRF(BaseModule):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True, init_cfg=None):
        super().__init__(init_cfg)
        n = int(nOut / 2)
        self.conv1x1 = Conv(nIn, n, 1, 1)
        self.F_loc = nn.Conv2d(n, n, 3, padding=1, groups=n)
        self.F_sur = nn.Conv2d(n, n, 3, padding=autopad(3, None, dilation_rate), dilation=dilation_rate, groups=n)
        self.bn_act = nn.Sequential(nn.BatchNorm2d(nOut), nn.SiLU())
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)
        joi_feat = torch.cat([loc, sur], 1)
        joi_feat = self.bn_act(joi_feat)
        output = self.F_glo(joi_feat)
        if self.add:
            output = input + output
        return output

class MRF_Block(BaseModule):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(MRF(c_, c_, add=shortcut) for _ in range(n)))
        
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class AG_Unit(BaseModule):
    def __init__(self, dim, reduction=16, init_cfg=None):
        super().__init__(init_cfg)
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
        self.bn = nn.BatchNorm2d(dim)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, 0))

    def forward(self, inputs):
        x, g = inputs
        x_ = x.clone()
        g_ = g.clone()
        
        multi = self.multi(x, g)
        multi = self.selection(multi)
        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)
        
        x_att = A.expand_as(x_) * x_
        g_att = B.expand_as(g_) * g_
        x_att = x_att + x_
        g_att = g_att + g_
        
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att
        
        interaction = x_att_2 * g_att_2
        projected = torch.sigmoid(self.bn(self.proj(interaction)))
        weighted = projected * x_
        y = self.conv_block(weighted)
        y = self.bn_2(y)
        return y

class MultiscaleFusion(BaseModule):
    def __init__(self, dim, init_cfg=None):
        super().__init__(init_cfg)
        self.local = ContextExtraction(dim)
        self.global_ = GlobalExtraction()
        self.bn = nn.BatchNorm2d(num_features=dim)

    def forward(self, x, g):
        x = self.local(x)
        g = self.global_(g)
        fuse = self.bn(x + g)
        return fuse

class ContextExtraction(BaseModule):
    def __init__(self, dim, reduction=None, init_cfg=None):
        super().__init__(init_cfg)
        self.reduction = 1 if reduction is None else 2
        self.dconv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // self.reduction, 1),
            nn.BatchNorm2d(dim // self.reduction)
        )
    def forward(self, x):
        x = self.dconv(x)
        x = self.proj(x)
        return x

class GlobalExtraction(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
        self.proj = nn.Sequential(nn.Conv2d(2, 1, 1, 1), nn.BatchNorm2d(1))

    def forward(self, x):
        x_avg = x.mean(1, keepdim=True)
        x_max = x.max(dim=1, keepdim=True)[0]
        cat = torch.cat((x_avg, x_max), dim=1)
        proj = self.proj(cat)
        return proj
        
class DCRM(BaseModule):
    def __init__(self, in_planes, ratio=4, flag=True, init_cfg=None):
        super().__init__(init_cfg)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.flag = flag
        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)

    def forward(self, x):
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out) * x if self.flag else self.sigmoid(out)

class DySample(BaseModule):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False, init_cfg=None):
        super().__init__(init_cfg)
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            out_channels = 2 * groups
            in_channels = in_channels // scale ** 2
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        if hasattr(self.offset, 'bias') and self.offset.bias is not None:
            nn.init.constant_(self.offset.bias, 0)
        
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            nn.init.constant_(self.scope.weight, 0.)
            if hasattr(self.scope, 'bias') and self.scope.bias is not None:
                nn.init.constant_(self.scope.bias, 0)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        grid_h, grid_w = torch.meshgrid(h, h, indexing='ij')
        return torch.stack([grid_w, grid_h]).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        
        num_points = offset.size(1) // 2
        offset = offset.view(B, num_points, 2, H, W).permute(0, 2, 1, 3, 4)
        
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        
        coords_h, coords_w = torch.meshgrid([coords_h, coords_w], indexing='ij')
        coords = torch.stack([coords_w, coords_h]).unsqueeze(0).unsqueeze(2).to(x.dtype)
        
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        
        # ======================== 关键修正: view -> reshape ========================
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).reshape(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        # ========================================================================
        
        x_reshaped = x.reshape(B * self.groups, -1, H, W)
        
        return F.grid_sample(x_reshaped, coords, mode='bilinear',
                             align_corners=False, padding_mode="border").reshape((B, -1, self.scale * H, self.scale * W))

    def forward(self, x):
        if self.style == 'lp':
            if hasattr(self, 'scope'):
                offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
            else:
                offset = self.offset(x) * 0.25 + self.init_pos
            return self.sample(x, offset)
        else: # 'pl' style
            x_ = F.pixel_shuffle(x, self.scale)
            if hasattr(self, 'scope'):
                offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
            else:
                offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
            return self.sample(x, offset)


class SplitAttentionMixer(BaseModule):
    def __init__(self, c1, c2, n=1, e=0.5, init_cfg=None):
        super().__init__(init_cfg)
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class PSABlock(BaseModule):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True, init_cfg=None):
        super().__init__(init_cfg)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
        
class Attention(BaseModule):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, init_cfg=None):
        super().__init__(init_cfg)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)
        
class Multiply(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(self, x):
        return x[0] * x[1]

class Add(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)
    
    def forward(self, x):
        return x[0] + x[1]
        
class DFL(BaseModule):
    def __init__(self, c1=16, init_cfg=None):
        super().__init__(init_cfg)
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
