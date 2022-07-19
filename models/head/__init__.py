from models.head.siamfc_head import SiamFC
from models.head.rpn import UPChannelRPN

BACKBONES = {
    'SiamFC': SiamFC,
    'UPChannelRPN': UPChannelRPN
}


def get_backbones(name, **kwargs):
    return BACKBONES[name](**kwargs)