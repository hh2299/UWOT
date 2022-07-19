import torch
import torch.nn as nn
from models.utils.xcorr import xcorr_fast


class RPN(nn.Module):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError


class UPChannelRPN(RPN):
    def __init__(self, anchor_num=5, feature_in=512):
        super(UPChannelRPN, self).__init__()

        cls_output = 2 * anchor_num
        loc_output = 4 * anchor_num

        self.template_cls_conv = nn.Conv2d(feature_in,
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in,
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in,
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in,
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc
