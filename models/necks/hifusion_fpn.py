import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..layers.custom_modules import Conv, HFEBlock


@MODELS.register_module()
class HiFusionFPN(BaseModule):
    """Multi-scale feature fusion neck.

    Inputs are expected in the order returned by HFEBackbone:
    layer2(P2/4), layer4(P3/8), layer6(P4/16), layer10(P5/32).
    Outputs are returned as P3, P4, P5 to match EfficientDecoder strides
    [8, 16, 32].
    """

    def __init__(self,
                 in_channels,
                 out_channels=(64, 128, 256),
                 num_blocks=1,
                 g=4,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert len(in_channels) == 4
        p2_ch, p3_ch, p4_ch, p5_ch = in_channels
        p3_out, p4_out, p5_out = out_channels

        self.p4_to_p5 = Conv(p4_ch, p3_out, 3, 2)
        self.p5_fuse = HFEBlock(p3_out + p5_ch, p5_out, n=num_blocks, g=g)

        self.up_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_to_p4 = Conv(p3_ch, max(8, p3_out // 2), 3, 2)
        self.p4_fuse = HFEBlock(
            max(8, p3_out // 2) + p5_out + p4_ch,
            p4_out,
            n=num_blocks,
            g=g)

        self.up_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_to_p3 = Conv(p2_ch, max(8, p3_out // 4), 3, 2)
        self.p3_fuse = HFEBlock(
            max(8, p3_out // 4) + p4_out + p3_ch,
            p3_out,
            n=num_blocks,
            g=g)

    def forward(self, inputs):
        p2, p3, p4, p5 = inputs

        y11 = self.p4_to_p5(p4)
        y13 = self.p5_fuse(torch.cat([y11, p5], dim=1))

        y14 = self.up_p5(y13)
        y15 = self.p3_to_p4(p3)
        y17 = self.p4_fuse(torch.cat([y15, y14, p4], dim=1))

        y18 = self.up_p4(y17)
        y19 = self.p2_to_p3(p2)
        y21 = self.p3_fuse(torch.cat([y19, y18, p3], dim=1))

        return y21, y17, y13
