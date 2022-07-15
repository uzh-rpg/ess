import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


class InterpolationLayer(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(InterpolationLayer, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode

    def forward(self, x):
        if self.scale_factor is not None:
            if self.mode == 'nearest' and self.scale_factor == 2:
                return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1),
                                                                                      2 * x.size(2), 2 * x.size(3))
            else:
                return self.interp(x, scale_factor=self.scale_factor, mode=self.mode)

        else:
            return self.interp(x, size=self.size, mode=self.mode)

