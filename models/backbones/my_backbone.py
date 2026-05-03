from mmengine.model import BaseModule

from mmdet.registry import MODELS
from ..layers.custom_modules import (ContextPoolMixer, Conv, HFEBlock,
                                     SplitAttentionMixer)


def make_divisible(x, divisor=8):
    return max(divisor, int(x + divisor / 2) // divisor * divisor)


def scaled_channels(channels, widen_factor=0.25, max_channels=1024):
    return make_divisible(min(channels, max_channels) * widen_factor, 8)


def scaled_depth(repeats, deepen_factor=0.5):
    return max(round(repeats * deepen_factor), 1) if repeats > 1 else repeats


@MODELS.register_module()
class HFEBackbone(BaseModule):
    """Backbone with hierarchical feature enhancement blocks."""

    def __init__(self,
                 in_channels=3,
                 deepen_factor=0.5,
                 widen_factor=0.25,
                 max_channels=1024,
                 out_indices=(2, 4, 6, 10),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.out_indices = out_indices
        c64 = scaled_channels(64, widen_factor, max_channels)
        c128 = scaled_channels(128, widen_factor, max_channels)
        c256 = scaled_channels(256, widen_factor, max_channels)
        c512 = scaled_channels(512, widen_factor, max_channels)
        c1024 = scaled_channels(1024, widen_factor, max_channels)
        n2 = scaled_depth(2, deepen_factor)

        self.layer_0 = Conv(in_channels, c64, 3, 2)
        self.layer_1 = Conv(c64, c128, 3, 2)
        self.layer_2 = HFEBlock(c128, c256, n=n2, g=4)
        self.layer_3 = Conv(c256, c256, 3, 2)
        self.layer_4 = HFEBlock(c256, c512, n=n2, g=4)
        self.layer_5 = Conv(c512, c512, 3, 2)
        self.layer_6 = HFEBlock(c512, c512, n=n2, g=4)
        self.layer_7 = Conv(c512, c1024, 3, 2)
        self.layer_8 = HFEBlock(c1024, c1024, n=n2, g=4)
        self.layer_9 = ContextPoolMixer(c1024, c1024, k=5)
        self.layer_10 = SplitAttentionMixer(c1024, c1024, n=n2)
        self.out_channels = (c256, c512, c512, c1024)

    def forward(self, x):
        outputs = {}
        for i in range(11):
            x = getattr(self, f'layer_{i}')(x)
            if i in self.out_indices:
                outputs[i] = x
        return tuple(outputs[i] for i in self.out_indices)
