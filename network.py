from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from common import *
from encoder import MobileNetV2Simple, MobileNetV2
from decoder import LightWeightRefineNetSimple, LightweightRefineNet

### There is no architecture difference between the simple and normal model implementations ###
### The simple implementation is easily readible ###
# class HydraNet(nn.Module):
#     def __init__(self):        
#         super().__init__() # Python 3
#         self.num_tasks = 2
#         self.num_classes = 6 
#         self.encoder = MobileNetv2Simple()
#         self.decoder = MTLWRefineNetSimple()

#     def forward(self, x):
#         encoder_outputs = self.encoder(x)
#         out_segm, out_depth = self.decoder(l3, l4, l5, l6, l7, l8)
#         return out_segm, out_depth

class HydraNet(nn.Module):
    def __init__(self):        
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = 6 
        self.encoder = MobileNetV2()
        self.decoder = MTLWRefineNet()

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs
        
        