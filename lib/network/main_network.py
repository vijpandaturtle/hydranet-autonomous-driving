from PIL import Image
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.network.common import *
from lib.network.encoder import MobileNetv2
from lib.network.decoder import MTLWRefineNet

class HydraNetNYUD(nn.Module):
    def __init__(self):        
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = 6 
        self.encoder = MobileNetv2()
        self.decoder = MTLWRefineNet(self.encoder._out_c, self.num_classes)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs
        
        