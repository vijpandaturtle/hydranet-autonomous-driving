from lib.network.common import *
from lib.network.encoder import MobileNetv2Simple, MobileNetv2
from lib.network.decoder import MTLWRefineNetSimple, MTLWRefineNet

class HydraNetNYUD(nn.Module):
    def __init__(self):        
        super().__init__() # Python 3
        self.num_tasks = 2
        self.num_classes = (40,1)
        self.encoder = MobileNetv2()
        self.decoder = MTLWRefineNet(self.encoder._out_c, self.num_classes)

    def forward(self, x):
        encoder_outputs = self.encoder(x)
        decoder_outputs = self.decoder(encoder_outputs)
        return decoder_outputs
        
        