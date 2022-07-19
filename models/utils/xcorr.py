import torch
import torch.nn.functional as F

def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    return po