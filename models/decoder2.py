import torch
import torch.nn as nn
import torch.nn.functional as F
from .vmamba3 import *
# =========================
# 基础卷积块
# =========================
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================
# 单层解码块
# =========================
class DecodeStage(nn.Module):
    """
    功能：
    1. 当前层双时相特征 concat
    2. 如果有来自下一层的上采样特征，则一起融合
    3. 先用 1x1 / 3x3 卷积整合通道
    4. 送入 vmamba3
    """
    def __init__(self, cur_in_channels, up_in_channels, out_channels):
        """
        cur_in_channels: 当前层双时相 concat 后的通道数 = 2 * C_cur
        up_in_channels : 来自下一级上采样特征的通道数；若本层为最底层则传 0
        out_channels   : 本层输出通道数
        """
        super().__init__()

        total_in = cur_in_channels + up_in_channels

        self.fuse = nn.Sequential(
            ConvBNReLU(total_in, out_channels, k=1, s=1, p=0),
            ConvBNReLU(out_channels, out_channels, k=3, s=1, p=1),
        )

        self.vmamba3 = VSSBlock(hidden_dim=out_channels,channel_first=True,norm_layer=LayerNorm2d)

    def forward(self, feat_t1, feat_t2, up_feat=None):
        """
        feat_t1, feat_t2: 当前层双时相特征
        up_feat: 来自下一层的上采样特征，若无则为 None
        """
        x = torch.cat([feat_t1, feat_t2], dim=1)

        if up_feat is not None:
            if up_feat.shape[-2:] != x.shape[-2:]:
                up_feat = F.interpolate(
                    up_feat, size=x.shape[-2:], mode='bilinear', align_corners=False
                )
            x = torch.cat([x, up_feat], dim=1)

        x = self.fuse(x)
        x = self.vmamba3(x)
        return x


# =========================
# 整体 Decoder
# =========================
class TemporalVMambaDecoder(nn.Module):
    """
    输入:
        feat_128_1, feat_128_2 : [B, C128, 128, 128]
        feat_64_1,  feat_64_2  : [B, C64,   64,  64]
        feat_32_1,  feat_32_2  : [B, C32,   32,  32]
        feat_16_1,  feat_16_2  : [B, C16,   16,  16]

    解码流程:
        stage16: concat(feat_16_1, feat_16_2) -> vmamba3
        stage32: concat(feat_32_1, feat_32_2, up(stage16)) -> vmamba3
        stage64: concat(feat_64_1, feat_64_2, up(stage32)) -> vmamba3
        stage128: concat(feat_128_1, feat_128_2, up(stage64)) -> vmamba3
        head -> logits

    输出:
        logits: [B, num_classes, 128, 128]
    """
    def __init__(
        self,
        enc_channels=(64, 64, 128, 256),   # (C128, C64, C32, C16)
        dec_channels=(64, 64, 128, 256),   # (D128, D64, D32, D16)
        num_classes=2
    ):
        super().__init__()

        c128, c64, c32, c16 = enc_channels
        d128, d64, d32, d16 = dec_channels

        # 最底层：仅当前层双时相 concat
        self.stage16 = DecodeStage(
            cur_in_channels=2 * c16,
            up_in_channels=0,
            out_channels=d16
        )

        # 上一层开始：当前层双时相 concat + 上采样特征
        self.stage32 = DecodeStage(
            cur_in_channels=2 * c32,
            up_in_channels=d16,
            out_channels=d32
        )

        self.stage64 = DecodeStage(
            cur_in_channels=2 * c64,
            up_in_channels=d32,
            out_channels=d64
        )

        self.stage128 = DecodeStage(
            cur_in_channels=2 * c128,
            up_in_channels=d64,
            out_channels=d128
        )

        self.seg_head = nn.Sequential(
            ConvBNReLU(d128, d128, k=3, s=1, p=1),
            nn.Conv2d(d128, num_classes, kernel_size=1)
        )

    def forward(
        self,
        feat_128_1, feat_128_2,
        feat_64_1, feat_64_2,
        feat_32_1, feat_32_2,
        feat_16_1, feat_16_2
    ):
        # 16x16
        x16 = self.stage16(feat_16_1, feat_16_2, up_feat=None)

        # 32x32
        x32 = self.stage32(feat_32_1, feat_32_2, up_feat=x16)

        # 64x64
        x64 = self.stage64(feat_64_1, feat_64_2, up_feat=x32)

        # 128x128
        x128 = self.stage128(feat_128_1, feat_128_2, up_feat=x64)

        logits = self.seg_head(x128)
        logits=F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=False)
        return logits
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 假设 encoder 四层输出通道分别为:
    # feat_128: 64
    # feat_64 : 64
    # feat_32 : 128
    # feat_16 : 256
    model = TemporalVMambaDecoder(
        enc_channels=(64, 64, 128, 256),
        dec_channels=(64, 96, 128, 160),
        num_classes=2
    ).to(device)

    model.eval()



    # batch size
    B = 2

    # 构造假的 encoder 输出
    feat_128_1 = torch.randn(B, 64, 128, 128).to(device)
    feat_128_2 = torch.randn(B, 64, 128, 128).to(device)

    feat_64_1 = torch.randn(B, 64, 64, 64).to(device)
    feat_64_2 = torch.randn(B, 64, 64, 64).to(device)

    feat_32_1 = torch.randn(B, 128, 32, 32).to(device)
    feat_32_2 = torch.randn(B, 128, 32, 32).to(device)

    feat_16_1 = torch.randn(B, 256, 16, 16).to(device)
    feat_16_2 = torch.randn(B, 256, 16, 16).to(device)

    with torch.no_grad():
        out = model(
            feat_128_1, feat_128_2,
            feat_64_1, feat_64_2,
            feat_32_1, feat_32_2,
            feat_16_1, feat_16_2
        )

    print("feat_128_1 shape:", feat_128_1.shape)
    print("feat_128_2 shape:", feat_128_2.shape)
    print("feat_64_1  shape:", feat_64_1.shape)
    print("feat_64_2  shape:", feat_64_2.shape)
    print("feat_32_1  shape:", feat_32_1.shape)
    print("feat_32_2  shape:", feat_32_2.shape)
    print("feat_16_1  shape:", feat_16_1.shape)
    print("feat_16_2  shape:", feat_16_2.shape)
    print("output shape :", out.shape)

    # 简单断言
    assert out.shape == (B, 2, 256, 256), f"输出 shape 不对: got {out.shape}"
    print("Forward test passed!")


if __name__ == "__main__":
    main()