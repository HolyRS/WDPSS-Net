import torch
import torch.nn as nn
from .encoder import Encoder
#from .decoder2 import TemporalVMambaDecoder
#from .final import UNetLikeDecoderWithAuxLoss
from .decoder import MultiScaleChannelFusion
from .final import UNetLikeDecoderWithAuxLoss
class TemporalFusionChange(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat1, feat2):
        diff = torch.abs(feat1 - feat2)
        x = torch.cat([feat1, feat2, diff], dim=1)   # [B, 3C, H, W]
        x = self.fuse(x)                             # [B, C, H, W]
        return x

class ChangeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.fuse_128 = TemporalFusionChange(64)
        self.fuse_64 = TemporalFusionChange(64)
        self.fuse_32 = TemporalFusionChange(128)
        self.fuse_16 = TemporalFusionChange(256)
        #self.decoder = TemporalVMambaDecoder()
        self.decoder = MultiScaleChannelFusion()
        self.final = UNetLikeDecoderWithAuxLoss()

    def forward(self, x1, x2):
        feat_128_1, feat_128_2, feat_64_1, feat_64_2, feat_32_1, feat_32_2, feat_16_1, feat_16_2 = self.encoder(x1, x2)
        feat_128 = self.fuse_128(feat_128_1, feat_128_2)
        feat_64 = self.fuse_64(feat_64_1, feat_64_2)
        feat_32 = self.fuse_32(feat_32_1, feat_32_2)
        feat_16 = self.fuse_16(feat_16_1, feat_16_2)
        out_128, out_64, out_32, out_16 = self.decoder(feat_128, feat_64, feat_32, feat_16)
        main_out, aux16, aux32, aux64 = self.final(out_128, out_64, out_32, out_16)
        #main_out = self.decoder(
            #feat_128_1, feat_128_2,
            #feat_64_1, feat_64_2,
            #feat_32_1, feat_32_2,
            #feat_16_1, feat_16_2
        #)

        #return main_out
        return main_out, aux16, aux32, aux64