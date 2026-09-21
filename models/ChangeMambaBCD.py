import torch
import torch.nn.functional as F
from .FM import FrequencyModule
from .SSM import SpatialModule1, SpatialModule2
import torch
import torch.nn as nn
from models.Mamba_backbone import Backbone_VSSM
from models.vmamba4 import LayerNorm2d, VSSBlock, Permute,VSSM
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from models.ChangeDecoder import ChangeDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
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
        #x = torch.abs(feat1 - feat2)
        return x

class ChangeMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(ChangeMambaBCD, self).__init__()
        self.encoder = Backbone_VSSM(out_indices=(0, 1, 2, 3), pretrained=pretrained, **kwargs)
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 

        norm_layer: nn.Module = _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        # -------------------------
        # 1. 两层层频率模块
        # -------------------------
        self.fm_128 = FrequencyModule(channels=128)  # 按你的真实通道改
        self.fm_64 = FrequencyModule(channels=256)


        # -------------------------
        # 2. 四层 ssm 模块
        # -------------------------
        self.ssm_128 = SpatialModule1(channels=128)
        self.ssm_64 = SpatialModule1(channels=256)
        self.ssm_32 = SpatialModule2(channels=512)
        self.ssm_16 = SpatialModule2(channels=1024)

        #self.decoder = ChangeDecoder(
            #encoder_dims=self.encoder.dims,
            #channel_first=self.encoder.channel_first,
            #norm_layer=norm_layer,
            #ssm_act_layer=ssm_act_layer,
            #mlp_act_layer=mlp_act_layer,
            #**clean_kwargs
        #)

        #self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)
        self.fuse_128 = TemporalFusionChange(128)
        self.fuse_64 = TemporalFusionChange(256)
        self.fuse_32 = TemporalFusionChange(512)
        self.fuse_16 = TemporalFusionChange(1024)
        # self.decoder = TemporalVMambaDecoder()
        self.decoder = MultiScaleChannelFusion()
        self.final = UNetLikeDecoderWithAuxLoss()

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def _process_scale1(self, feat1, feat2, fm_module, ssm_module):
        (
            HH1, HL1, LH1,
            HH2, HL2, LH2,
            #HH3, HL3, LH3,
            J1, J2#, Y
        ) = fm_module(feat1, feat2)

        out1, out2 = ssm_module(
            F1=feat1,
            F2=feat2,
            G1=J1,
            G2=J2,
            #G3=Y,
            HH1=HH1, HL1=HL1, LH1=LH1,
            HH2=HH2, HL2=HL2, LH2=LH2,
            #HH3=HH3, HL3=HL3, LH3=LH3
        )
        return out1, out2
    def _process_scale2(self, feat1, feat2, ssm_module):


        out1, out2 = ssm_module(
            F1=feat1,
            F2=feat2,
        )
        return out1, out2
    def forward(self, pre_data, post_data):
        # Encoder processing
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)
        #print(pre_features[2].shape)
        #print(post_features[2].shape)
        output_128_1, output_64_1, output_32_1, output_16_1 = pre_features
        output_128_2, output_64_2, output_32_2, output_16_2 = post_features

        feat_128_1, feat_128_2 = self._process_scale1(output_128_1, output_128_2, self.fm_128, self.ssm_128)
        feat_64_1, feat_64_2 = self._process_scale1(output_64_1, output_64_2, self.fm_64, self.ssm_64)
        feat_32_1, feat_32_2 = self._process_scale2(output_32_1, output_32_2,  self.ssm_32)
        feat_16_1, feat_16_2 = self._process_scale2(output_16_1, output_16_2,  self.ssm_16)
        
        feat_128 = self.fuse_128(feat_128_1, feat_128_2)
        feat_64 = self.fuse_64(feat_64_1, feat_64_2)
        feat_32 = self.fuse_32(feat_32_1, feat_32_2)
        feat_16 = self.fuse_16(feat_16_1, feat_16_2)

        #feat_128 = self.fuse_128(output_128_1, output_128_2)
        #feat_64 = self.fuse_64(output_64_1, output_64_2)
        #feat_32 = self.fuse_32(output_32_1, output_32_2)
        #feat_16 = self.fuse_16(output_16_1, output_16_2)

        out_128, out_64, out_32, out_16 = self.decoder(feat_128, feat_64, feat_32, feat_16)
        main_out, aux16, aux32, aux64 = self.final(out_128, out_64, out_32, out_16)
        #pre_features_new = [feat_128_1, feat_64_1, feat_32_1, feat_16_1]
        #post_features_new = [feat_128_2, feat_64_2, feat_32_2, feat_16_2]
        # Decoder processing - passing encoder outputs to the decoder
        #output = self.decoder(pre_features_new, post_features_new)

        #output = self.main_clf(output)
        #output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        #return output
        return main_out, aux16, aux32, aux64
