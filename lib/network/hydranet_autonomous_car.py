import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.network.common import *

class HydraNet(nn.Module):
    def __init__(self):        
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = 6
    
    def define_mobilenet(self):
        mobilenet_config = [[1, 16, 1, 1], # expansion rate, output channels, number of repeats, stride
                        [6, 24, 2, 2],
                        [6, 32, 3, 2],
                        [6, 64, 4, 2],
                        [6, 96, 3, 1],
                        [6, 160, 3, 2],
                        [6, 320, 1, 1],
                        ]
        self.in_channels = 32 # number of input channels
        self.num_layers = len(mobilenet_config)
        self.layer1 = convbnrelu(3, self.in_channels, kernel_size=3, stride=2)
        c_layer = 2
        for t,c,n,s in (mobilenet_config):
            layers = []
            for idx in range(n):
                layers.append(InvertedResidualBlock(self.in_channels, c, expansion_factor=t, stride=s if idx == 0 else 1))
                self.in_channels = c
            setattr(self, 'layer{}'.format(c_layer), nn.Sequential(*layers)) # setattr(object, name, value)
            c_layer += 1
        
    def _make_crp(self, in_planes, out_planes, stages, groups=False):
        layers = [CRPBlock(in_planes, out_planes,stages, groups=groups)]
        return nn.Sequential(*layers)
    
    def define_lightweight_refinenet(self):
        ## Light-Weight RefineNet ##
        self.conv8 = conv1x1(320, 256, bias=False)
        self.conv7 = conv1x1(160, 256, bias=False)
        self.conv6 = conv1x1(96, 256, bias=False)
        self.conv5 = conv1x1(64, 256, bias=False)
        self.conv4 = conv1x1(32, 256, bias=False)
        self.conv3 = conv1x1(24, 256, bias=False)
        self.crp4 = self._make_crp(256, 256, 4, groups=False)
        self.crp3 = self._make_crp(256, 256, 4, groups=False)
        self.crp2 = self._make_crp(256, 256, 4, groups=False)
        self.crp1 = self._make_crp(256, 256, 4, groups=True)

        self.conv_adapt4 = conv1x1(256, 256, bias=False)
        self.conv_adapt3 = conv1x1(256, 256, bias=False)
        self.conv_adapt2 = conv1x1(256, 256, bias=False)

        self.pre_depth = conv1x1(256, 256, groups=256, bias=False)
        self.depth = conv3x3(256, 1, bias=True)
        self.pre_segm = conv1x1(256, 256, groups=256, bias=False)
        self.segm = conv3x3(256, self.num_classes, bias=True)
        self.relu = nn.ReLU6(inplace=True)

        if self.num_tasks == 3:
            self.pre_normal = conv1x1(256, 256, groups=256, bias=False)
            self.normal = conv3x3(256, 3, bias=True)
    
    def forward(self, x):
        # MOBILENET V2
        x = self.layer1(x)
        x = self.layer2(x) # x / 2
        l3 = self.layer3(x) # 24, x / 4
        l4 = self.layer4(l3) # 32, x / 8
        l5 = self.layer5(l4) # 64, x / 16
        l6 = self.layer6(l5) # 96, x / 16
        l7 = self.layer7(l6) # 160, x / 32
        l8 = self.layer8(l7) # 320, x / 32

        # LIGHT-WEIGHT REFINENET
        l8 = self.conv8(l8)
        l7 = self.conv7(l7)
        l7 = self.relu(l8 + l7)
        l7 = self.crp4(l7)
        l7 = self.conv_adapt4(l7)
        l7 = nn.Upsample(size=l6.size()[2:], mode='bilinear', align_corners=False)(l7)

        l6 = self.conv6(l6)
        l5 = self.conv5(l5)
        l5 = self.relu(l5 + l6 + l7)
        l5 = self.crp3(l5)
        l5 = self.conv_adapt3(l5)
        l5 = nn.Upsample(size=l4.size()[2:], mode='bilinear', align_corners=False)(l5)

        l4 = self.conv4(l4)
        l4 = self.relu(l5 + l4)
        l4 = self.crp2(l4)
        l4 = self.conv_adapt2(l4)
        l4 = nn.Upsample(size=l3.size()[2:], mode='bilinear', align_corners=False)(l4)

        l3 = self.conv3(l3)
        l3 = self.relu(l3 + l4)
        l3 = self.crp1(l3)

        # HEADS
        out_segm = self.pre_segm(l3)
        out_segm = self.relu(out_segm)
        out_segm = self.segm(out_segm)

        out_d = self.pre_depth(l3)
        out_d = self.relu(out_d)
        out_d = self.depth(out_d)

        if self.num_tasks == 3:
            out_n = self.pre_normal(l3)
            out_n = self.relu(out_n)
            out_n = self.normal(out_n)
            return out_segm, out_d, out_n
        else:
            return out_segm, out_d
