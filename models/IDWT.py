import torch
import torch.nn as nn


class HaarIDWT2D(nn.Module):
    """
    输入:
        x: [B, 4C, H, W]
        通道排列顺序固定为 [HH, HL, LH, LL]

    输出:
        out: [B, C, 2H, 2W]
    """
    def __init__(self):
        super(HaarIDWT2D, self).__init__()

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"输入张量维度必须是 [B, 4C, H, W]，但得到 {x.shape}")

        B, C4, H, W = x.shape
        if C4 % 4 != 0:
            raise ValueError(f"输入通道数必须能被4整除，但得到 {C4}")

        C = C4 // 4

        # 按顺序拆分: [HH, HL, LH, LL]
        HH, HL, LH, LL = torch.chunk(x, 4, dim=1)

        out = torch.zeros(
            (B, C, H * 2, W * 2),
            dtype=x.dtype,
            device=x.device
        )

        # 逆 Haar DWT
        out[:, :, 0::2, 0::2] = (LL + LH + HL + HH) * 0.5
        out[:, :, 0::2, 1::2] = (LL + LH - HL - HH) * 0.5
        out[:, :, 1::2, 0::2] = (LL - LH + HL - HH) * 0.5
        out[:, :, 1::2, 1::2] = (LL - LH - HL + HH) * 0.5

        return out


if __name__ == "__main__":
    x = torch.randn(2, 256, 64, 64)   # B=2, 4C=256, H=64, W=64
    idwt = HaarIDWT2D()
    y = idwt(x)
    print("input shape :", x.shape)
    print("output shape:", y.shape)    # [2, 64, 128, 128]