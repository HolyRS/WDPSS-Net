import torch
import torch.nn as nn


class PreIDWTRefine(nn.Module):
    """
    IDWT 前的轻量整理模块

    输入:
        x: [B, 4C, H, W]
           通道顺序固定为 [HH, HL, LH, LL]
    输出:
        out: [B, 4C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        in_channels = channels * 4

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = x + self.refine(x)   # 残差式整理
        return out