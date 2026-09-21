import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """
    PyTorch version of 2D Haar DWT
    Input:
        x: [B, C, H, W]
    Output:
        HH, HL, LH, LL
        each: [B, C, H/2, W/2]

    Notes:
        1. only supports even H and W
        2. no padding is used
        3. each channel is transformed independently
    """
    def __init__(self):
        super().__init__()

        # 四个 2x2 Haar 小波核
        # 这里我直接按你想要的输出顺序来放：HH, HL, LH, LL

        hh = torch.tensor([[1., -1.],
                           [-1., 1.]]) / 2.0

        hl = torch.tensor([[1., -1.],
                           [1., -1.]]) / 2.0

        lh = torch.tensor([[1.,  1.],
                           [-1., -1.]]) / 2.0

        ll = torch.tensor([[1.,  1.],
                           [1.,  1.]]) / 2.0

        # [4, 1, 2, 2]
        filters = torch.stack([hh, hl, lh, ll], dim=0).unsqueeze(1)

        # register_buffer: 不是可学习参数，但会跟随 .to(device)
        self.register_buffer("filters", filters)

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], but got {x.shape}")

        B, C, H, W = x.shape

        if H % 2 != 0 or W % 2 != 0:
            raise ValueError(f"H and W must be even, but got H={H}, W={W}")

        # 对每个通道独立做 4 个小波卷积核
        # self.filters: [4,1,2,2]
        # repeat 后: [4*C,1,2,2]
        weight = self.filters.repeat(C, 1, 1, 1)

        # groups=C 表示每个输入通道单独卷积，不发生通道混合
        y = F.conv2d(x, weight, stride=2, padding=0, groups=C)
        # y shape: [B, 4*C, H/2, W/2]

        # reshape 成 [B, C, 4, H/2, W/2]
        y = y.view(B, C, 4, H // 2, W // 2)

        # 按我们定义的顺序拆出来
        HH = y[:, :, 0, :, :]
        HL = y[:, :, 1, :, :]
        LH = y[:, :, 2, :, :]
        LL = y[:, :, 3, :, :]

        return HH, HL, LH, LL
if __name__ == "__main__":
    dwt = HaarDWT2D()

    x = torch.randn(2, 64, 128, 128)   # [B, C, H, W]
    HH, HL, LH, LL = dwt(x)

    print("HH shape:", HH.shape)
    print("HL shape:", HL.shape)
    print("LH shape:", LH.shape)
    print("LL shape:", LL.shape)