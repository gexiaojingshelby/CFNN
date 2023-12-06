# ------------------------------------------------------------------------
# Reference:
# https://github.com/facebookresearch/detr/blob/main/models/backbone.py
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from spatial_correlation_sampler import SpatialCorrelationSampler

from .position_encoding import build_position_encoding


class Backbone(nn.Module):
    def __init__(self, args):
        super(Backbone, self).__init__()

        backbone = getattr(torchvision.models, args.backbone)(
            replace_stride_with_dilation=[False, False, args.dilation], pretrained=True)

        self.num_frames = args.num_frame
        self.num_channels = 512 if args.backbone in ('resnet18', 'resnet34') else 2048

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1=x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x,x1

def build_backbone(args):
    backbone = Backbone(args)
    return backbone
