import torch
import torch.nn as nn
from .DWT import HaarDWT2D
from .SR import SRModule
from .preidwtrefine import PreIDWTRefine
from .IDWT import HaarIDWT2D

class FrequencyModule(nn.Module):
    """
    总体频率模块

    流程：
        F1, F2
          -> DWT
          -> G1, G2
          -> G3 = G1 - G2
          -> SR(G1), SR(G2), SR(G3)
          -> H1, H2, H3
          -> I1 = H1 * H3, I2 = H2 * H3
          -> 3x3 refine
          -> IDWT
          -> J1, J2
          -> Y = J1 - J2

    输入:
        F1, F2: [B, C, H, W]

    输出:
        Y: [B, C, H, W]
    """
    def __init__(self, channels):
        super(FrequencyModule, self).__init__()

        self.channels = channels

        # 1. DWT
        self.dwt = HaarDWT2D()

        # 2. SR
        self.sr = SRModule(channels=channels)

        # 3. IDWT前的3x3整理模块
        self.pre_idwt_refine = PreIDWTRefine(channels=channels)

        # 4. IDWT
        self.idwt = HaarIDWT2D()

    def _pack_subbands(self, HH, HL, LH, LL):
        """
        将4个子带按固定顺序拼接成 [B, 4C, H, W]
        顺序固定为 [HH, HL, LH, LL]
        """
        return torch.cat([HH, HL, LH, LL], dim=1)

    def forward(self, F1, F2):
        """
        F1, F2: [B, C, H, W]
        """
        # -------------------------
        # Step 1: 双时相特征分别做 DWT
        # -------------------------
        HH1, HL1, LH1, LL1 = self.dwt(F1)
        HH2, HL2, LH2, LL2 = self.dwt(F2)

        # G1, G2: [B, 4C, H/2, W/2]
        #G1 = self._pack_subbands(HH1, HL1, LH1, LL1)
        #G2 = self._pack_subbands(HH2, HL2, LH2, LL2)

        # G3: 差异频率表示
        #G3 = G1 - G2
        #HH3,HL3,LH3,LL3= torch.chunk(G3, 4, dim=1)
        # -------------------------
        # Step 2: G1, G2, G3 均进入 SR 模块
        # -------------------------
        H1 = self.sr(HH1, HL1, LH1, LL1)   # [B, 4C, H/2, W/2]
        H2 = self.sr( HH2, HL2, LH2, LL2)   # [B, 4C, H/2, W/2]
        #H3 = self.sr( HH3,HL3,LH3,LL3)   # [B, 4C, H/2, W/2]

        # -------------------------
        # Step 3: H1,H2 与 H3 逐元素相乘
        # -------------------------
        I1 = H1 #* H3
        I2 = H2 #* H3

        # -------------------------
        # Step 4: I1, I2 分别经过 3x3 模块
        # -------------------------
        I1 = self.pre_idwt_refine(I1)
        I2 = self.pre_idwt_refine(I2)

        # -------------------------
        # Step 5: I1, I2 分别经过 IDWT
        # -------------------------
        J1 = self.idwt(I1)   # [B, C, H, W]
        J2 = self.idwt(I2)   # [B, C, H, W]

        # -------------------------
        # Step 6: 相减得到最终输出
        # -------------------------
        #Y = J1 - J2
        return HH1,HL1,LH1,HH2,HL2,LH2,J1,J2#,HH3,HL3,LH3,J1,J2,Y
if __name__ == "__main__":
    B, C, H, W = 2, 64, 128, 128
    F1 = torch.randn(B, C, H, W)
    F2 = torch.randn(B, C, H, W)

    model = FrequencyModule(channels=C)
    HH1,HL1,LH1,HH2,HL2,LH2,J1,J2,Y = model(F1, F2)

    print("F1 shape:", F1.shape)
    print("F2 shape:", F2.shape)
    print("HH1 shape :", HH1.shape)
    print("HL1 shape :", HL1.shape)
    print("LH1 shape :", LH1.shape)
    print("HH2 shape :", HH2.shape)
    print("HL2 shape :", HL2.shape)
    print("LH2 shape :", LH2.shape)
    print("J1 shape :", J1.shape)
    print("J2 shape :", J2.shape)
    print("Y shape :",   Y.shape)  