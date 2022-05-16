from models.backbone.alexnet import AlexNetV1, AlexNetV2, AlexNetV3

BACKBONES = {
    'AlexNetV1': AlexNetV1,
    'AlexNetV2': AlexNetV2,
    'AlexNetV3': AlexNetV3
}


def get_backbones(name, **kwargs):
    return BACKBONES[name](**kwargs)
