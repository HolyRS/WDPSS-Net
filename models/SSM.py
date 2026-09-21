import torch
import torch.nn as nn
from .vmamba import VSSBlock,LayerNorm2d
from .vmamba2 import VSSBlock as VSSBlock2
from .vmamba2 import LayerNorm2d as LayerNorm2d2
from .Frequencydirection import FrequencySubbandGate
class ConvReduce(nn.Module):
    """
    concat 后降维模块
    输入:  [B, 2C, H, W]
    输出:  [B, C, H, W]
    """
    def __init__(self, channels):
        super(ConvReduce, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SpatialModule1(nn.Module):
    """
    空间模块

    输入:
        F1, F2:   [B, C, H, W]
        G1, G2:   [B, C, H, W]
        G3:       [B, C, H, W]，若为 None，则内部用 G1-G2 计算
        HH1, HL1, LH1: [B, C, h, w]
        HH2, HL2, LH2: [B, C, h, w]

    输出:
        J1, J2, J3:   [B, C, H, W]
        fdf1, fdf2:   [B, C, 4, H, W]
    """
    def __init__(self, channels):
        super(SpatialModule1, self).__init__()
        self.channels = channels

        # 1) 频率方向门控模块（共享权重）
        self.freq_gate = FrequencySubbandGate()
        self.freq_gate2 = FrequencySubbandGate()

        # 2) H1 / H2 共用降维模块
        self.fuse_same = ConvReduce(channels)

        # 3) H3 差异分支单独降维
       # self.fuse_diff = ConvReduce(channels)

        # 4) VSSBlock
        # 建议 H1/H2 共用一个 VSSBlock，H3 单独一个
        self.vss_same = VSSBlock(hidden_dim=channels,channel_first=True,norm_layer=LayerNorm2d,ssm_ratio=1.0)
        #self.vss_diff = VSSBlock(hidden_dim=channels,channel_first=True,norm_layer=LayerNorm2d,ssm_ratio=1.0)

    def forward(
        self,
        F1, F2,
        G1, G2,#G3,
        HH1, HL1, LH1,
        HH2, HL2, LH2,
        #HH3, HL3, LH3

    ):
        """
        F1, F2, G1, G2: [B, C, H, W]
        HH1, HL1, LH1, HH2, HL2, LH2: [B, C, h, w]
        """

        # -------------------------
        # 1. 频率方向场门控
        # -------------------------
        H, W = F1.shape[-2], F1.shape[-1]

        fdf1 = self.freq_gate(HH1, HL1, LH1, target_size=(H, W))   # [B, C, 4, H, W]
        fdf2 = self.freq_gate(HH2, HL2, LH2, target_size=(H, W))   # [B, C, 4, H, W]
        #fdf3 = self.freq_gate2(HH3, HL3, LH3, target_size=(H, W))
        # -------------------------
        # 2. G3 若未给定，则内部计算
        # -------------------------
        #if G3 is None:
            #G3 = G1 - G2

        # -------------------------
        # 3. 构造 H1, H2, H3
        # -------------------------
        H1 = self.fuse_same(torch.cat([G1, F1], dim=1))           # [B, C, H, W]
        H2 = self.fuse_same(torch.cat([G2, F2], dim=1))           # [B, C, H, W]
        #H3 = self.fuse_diff(torch.cat([G3, F1 - F2], dim=1))      # [B, C, H, W]
        # -------------------------
        # 4. 送入 VSSBlock
        # -------------------------
        J1 = self.vss_same(input=H1,fdf=fdf1)    # [B, C, H, W]
        J2 = self.vss_same(H2,fdf2)    # [B, C, H, W]
        #J3 = self.vss_diff(H3,fdf3)    # [B, C, H, W]

        #Y = (J2 - J1) + (J2 - J1) * torch.sigmoid(J3)

        #return Y
        return J1,J2
class SpatialModule2(nn.Module):
    """
    空间模块

    输入:
        F1, F2:   [B, C, H, W]


    输出:
        Y  [B, C, H, W]

    """
    def __init__(self, channels):
        super(SpatialModule2, self).__init__()
        self.channels = channels
        # 建议 H1/H2 共用一个 VSSBlock，H3 单独一个
        self.vss_same = VSSBlock2(hidden_dim=channels,channel_first=True,norm_layer=LayerNorm2d2,ssm_ratio=1.0)
        #self.vss_diff = VSSBlock2(hidden_dim=channels,channel_first=True,norm_layer=LayerNorm2d2,ssm_ratio=1.0)

    def forward(
        self,
        F1, F2

    ):
        """
        F1, F2, G1, G2: [B, C, H, W]
        HH1, HL1, LH1, HH2, HL2, LH2: [B, C, h, w]
        """
        # -------------------------
        J1 = self.vss_same(F1)    # [B, C, H, W]
        J2 = self.vss_same(F2)    # [B, C, H, W]
        #J3 = self.vss_diff(F1-F2)    # [B, C, H, W]

        #Y = (J2 - J1) + (J2 - J1) * torch.sigmoid(J3)

        return J1,J2

if __name__ == "__main__":
    SM = SpatialModule1(channels=64).to('cuda:0')
    F1=torch.randn(2, 64, 128, 128).to('cuda:0')
    F2=torch.randn(2, 64, 128, 128).to('cuda:0')
    G1=torch.randn(2, 64, 128, 128).to('cuda:0')
    G2=torch.randn(2, 64, 128, 128).to('cuda:0')
    G3=torch.randn(2, 64, 128, 128).to('cuda:0')
    HH1=torch.randn(2, 64, 64, 64).to('cuda:0')
    HL1=torch.randn(2, 64, 64, 64).to('cuda:0')
    LH1=torch.randn(2, 64, 64, 64).to('cuda:0')
    HH2=torch.randn(2, 64, 64, 64).to('cuda:0')
    HL2=torch.randn(2, 64, 64, 64).to('cuda:0')
    LH2=torch.randn(2, 64, 64, 64).to('cuda:0')
    HH3=torch.randn(2, 64, 64, 64).to('cuda:0')
    HL3=torch.randn(2, 64, 64, 64).to('cuda:0')
    LH3=torch.randn(2, 64, 64, 64).to('cuda:0')
    Y= SM(F1=F1, F2=F2,G1=G1,G2=G2,G3=G3,HH1=HH1,HL1=HL1,LH1=LH1,HH2=HH2,HL2=HL2,LH2=LH2,HH3=HH3,HL3=HL3,LH3=LH3)
    print(Y.shape)
