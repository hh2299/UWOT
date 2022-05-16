import torch.nn as nn

from models.backbone import get_backbones


class ModelBuilder(nn.Module):
    def __init__(self, backbone_name):
        super(ModelBuilder, self).__init__()
        self.backbones = get_backbones(backbone_name)

