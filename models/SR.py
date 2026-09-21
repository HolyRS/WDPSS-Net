import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn


class DirectionalStripAttention(nn.Module):
    """
    方向条带卷积注意力
    输入:  [B, C, H, W]
    输出:  [B, C, H, W]

    思想:
    - 1x7 卷积提取横向条带上下文
    - 7x1 卷积提取纵向条带上下文
    - 融合后生成空间重标定图
    """
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"

        pad = kernel_size // 2

        self.conv_h = nn.Conv2d(
            channels, channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=channels,
            bias=False
        )

        self.conv_v = nn.Conv2d(
            channels, channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=channels,
            bias=False
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        h_feat = self.conv_h(x)
        v_feat = self.conv_v(x)

        attn = self.fuse(torch.cat([h_feat, v_feat], dim=1))
        out = x * attn
        return out

def _get_group_num(channels, max_groups=8):
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class DirectionalEnhance(nn.Module):
    """
    简化版方向增强卷积
    mode='h': 水平方向敏感
    mode='v': 垂直方向敏感
    """
    def __init__(self, channels, mode='h'):
        super().__init__()
        assert mode in ['h', 'v']

        if mode == 'h':
            ksize = (1, 3)
            padding = (0, 1)
        else:
            ksize = (3, 1)
            padding = (1, 0)

        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=ksize,
            padding=padding,
            groups=channels,
            bias=False
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class HighFreqPurifier(nn.Module):
    """
    高频净化分支
    输入: [B, 3C, H, W]
    输出: [B, 3C, H, W]
    """
    def __init__(self, channels_3c, expand_ratio=2):
        super().__init__()
        hidden = channels_3c * expand_ratio
        gn_groups = _get_group_num(channels_3c)

        self.dw = nn.Conv2d(
            channels_3c, channels_3c,
            kernel_size=3, padding=1,
            groups=channels_3c, bias=False
        )
        self.gn = nn.GroupNorm(gn_groups, channels_3c)

        self.pw1 = nn.Conv2d(channels_3c, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.pw2 = nn.Conv2d(hidden, channels_3c, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.dw(x)
        y = self.gn(y)
        y = self.pw1(y)
        y = self.act(y)
        y = self.pw2(y)
        return y


class SRModule(nn.Module):
    """
    输入:
        HH, HL, LH, LL, 每个都是 [B, C, H, W]
    输出:
        HH_ref, HL_ref, LH_ref, LL_ref, 每个都是 [B, C, H, W]
    """
    def __init__(self, channels,strip_kernel=7):
        super().__init__()
        self.channels = channels

        # LH / HL 方向增强
        self.heconv = DirectionalEnhance(channels, mode='h')  # 给 LH
        self.veconv = DirectionalEnhance(channels, mode='v')  # 给 HL

        # 高频净化器，输入 [LH_enh, HL_enh, HH]
        self.purifier = HighFreqPurifier(channels * 3, expand_ratio=2)

        # 重组后的融合映射
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels * 4),
            nn.ReLU(inplace=True)
        )
        # 用方向条带卷积替代原来的 7x7 卷积
        self.strip_attn = DirectionalStripAttention(channels * 4, kernel_size=strip_kernel)

    def forward(self, HH, HL, LH, LL):
        # 1. 方向增强
        LH_enh = self.heconv(LH)
        HL_enh = self.veconv(HL)

        # 2. 高频拼接
        high_cat = torch.cat([LH_enh, HL_enh, HH], dim=1)   # [B, 3C, H, W]

        # 3. 高频净化
        noise = self.purifier(high_cat)
        high_clean = torch.sign(high_cat) * torch.abs(high_cat - noise)

        # 拆回三个高频子带
        LH_clean, HL_clean, HH_clean = torch.chunk(high_clean, 3, dim=1)

        # 4. 与低频重组
        orig = torch.cat([LL, LH, HL, HH], dim=1)           # [B, 4C, H, W]
        refined = torch.cat([LL, LH_clean, HL_clean, HH_clean], dim=1)

        # 5. 残差融合
        fused = orig + self.fuse(refined)
        # 6) 方向条带卷积重标定
        fused = self.strip_attn(fused)
        # 7) 拆回四个子带
        LL_ref, LH_ref, HL_ref, HH_ref = torch.chunk(fused, 4, dim=1)

        # 按项目顺序先拼回去输出
        sr_out = torch.cat([HH_ref, HL_ref, LH_ref, LL_ref], dim=1)

        return sr_out

        # 按项目顺序返回
        return fused
if __name__ == '__main__':
    B, C, H, W = 2, 64, 64, 64
    HH = torch.randn(B, C, H, W)
    HL = torch.randn(B, C, H, W)
    LH = torch.randn(B, C, H, W)
    LL = torch.randn(B, C, H, W)

    sr = SRModule(channels=C)
    a = sr(HH, HL, LH, LL)

    print(a.shape)