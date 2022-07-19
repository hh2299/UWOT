import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedLoss(nn.Module):
    def __init__(self, neg_weight=1.0):
        super(BalancedLoss, self).__init__()
        self.neg_weight = neg_weight

    def forward(self, input, target):
        pos_mark = (target == 1)
        neg_mark = (target == 0)
        pos_num = pos_mark.sum().float()
        neg_num = neg_mark.sum().float()
        weight = target.new_zeros(target.size())
        weight[pos_mark] = 1 / pos_num
        weight[neg_mark] = 1 / neg_num * self.neg_weight
        weight /= weight.sum()
        return F.binary_cross_entropy_with_logits(
            input, target, weight, reduction='sum'
        )


class SoftMaxLoss(nn.Module):

    def __init__(self):
        super(SoftMaxLoss, self).__init__()

    def forward(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
