import torch.nn as nn
import torch.nn.functional as F
from models.utils.xcorr import xcorr_fast


class SiamFC(nn.Module):
    def __init__(self, out_scale=0.01):
        super(SiamFC, self).__init__()
        self.out_scale = out_scale

    def forward(self, z, x):
        return xcorr_fast(x, z) * self.out_scale

    # def _fast_xcorr(self, z, x):
    #     nz = z.size(0)
    #     nx, c, h, w = x.size()
    #     x = x.view(-1, nz * c, h, w)
    #     out = F.conv2d(x, z, groups=nz)
    #     out = out.view(nx, -2, out.size(-2), out.size(-1))
    #     return out
