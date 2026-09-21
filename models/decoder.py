import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


import torch
import torch.nn as nn

# 通道注意力模块（可直接用于特征增强）
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        hidden_planes = max(1, in_planes // ratio)  # 防止中间通道为0

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, kernel_size=1, bias=False)  # 降维
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_planes, in_planes, kernel_size=1, bias=False)  # 升维

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 平均池化分支
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 最大池化分支

        out = avg_out + max_out
        attn = self.sigmoid(out)   # [B, C, 1, 1]

        return x * attn            # 直接返回加权后的特征
class CrossScaleFusionBlock(nn.Module):
    """
    对某一目标层做跨尺度融合增强

    输入:
        feats: list of 4 feature maps
        target_idx: 当前目标层索引

    流程:
        1. 所有层 resize 到当前层大小
        2. 通道拼接
        3. 通道注意力
        4. 降维到当前层通道数
        5. 与原始当前层 concat
        6. 再降维回当前层通道数

    输出:
        与当前层 shape 相同的增强特征
    """
    def __init__(self, in_channels_list, target_idx):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.target_idx = target_idx
        self.target_channels = in_channels_list[target_idx]
        self.total_channels = sum(in_channels_list)

        # 拼接后做通道注意力
        self.channel_attn = ChannelAttention(self.total_channels)

        # 多尺度拼接后，先降到当前层通道数
        self.reduce1 = ConvBNReLU(self.total_channels, self.target_channels, kernel_size=1, padding=0)

        # 再与原始当前层特征 concat，继续融合降维
        self.reduce2 = nn.Sequential(
            ConvBNReLU(self.target_channels * 2, self.target_channels, kernel_size=3, padding=1),
            ConvBNReLU(self.target_channels, self.target_channels, kernel_size=3, padding=1)
        )

    def _resize_to_target(self, x, target_size):
        if x.shape[-2:] == target_size:
            return x
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

    def forward(self, feats):
        """
        feats: [f1, f2, f3, f4]
        """
        target_feat = feats[self.target_idx]
        target_size = target_feat.shape[-2:]

        resized_feats = [self._resize_to_target(f, target_size) for f in feats]

        # 4层特征全部拼接
        fused = torch.cat(resized_feats, dim=1)

        # 通道注意力
        fused = self.channel_attn(fused)

        # 降维到当前层通道数
        fused = self.reduce1(fused)

        # 与原始当前层特征再拼接
        out = torch.cat([target_feat, fused], dim=1)

        # 再降维
        delta = self.reduce2(out)
        x=target_feat+0.15*delta
        return x
class MultiScaleChannelFusion(nn.Module):
    """
    对 encoder 输出的 4 层特征图分别做跨尺度通道融合增强

    输入:
        feat_128, feat_64, feat_32, feat_16

    输出:
        out_128, out_64, out_32, out_16
    """
    def __init__(self, in_channels_list=(128,256, 512, 1024)):
        super().__init__()
        self.in_channels_list = list(in_channels_list)

        self.block1 = CrossScaleFusionBlock(self.in_channels_list, target_idx=0)
        self.block2 = CrossScaleFusionBlock(self.in_channels_list, target_idx=1)
        self.block3 = CrossScaleFusionBlock(self.in_channels_list, target_idx=2)
        self.block4 = CrossScaleFusionBlock(self.in_channels_list, target_idx=3)

    def forward(self, feat_128, feat_64, feat_32, feat_16):
        feats = [feat_128, feat_64, feat_32, feat_16]

        out_128 = self.block1(feats)
        out_64  = self.block2(feats)
        out_32  = self.block3(feats)
        out_16  = self.block4(feats)

        return out_128, out_64, out_32, out_16
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MultiScaleChannelFusion(
        in_channels_list=(64, 64, 128, 256),

    ).to(device)

    feat_128 = torch.randn(1, 64, 128, 128).to(device)
    feat_64  = torch.randn(1, 64, 64, 64).to(device)
    feat_32  = torch.randn(1, 128, 32, 32).to(device)
    feat_16  = torch.randn(1, 256, 16, 16).to(device)

    out_128, out_64, out_32, out_16 = model(feat_128, feat_64, feat_32, feat_16)

    print("Input:")
    print("feat_128:", feat_128.shape)
    print("feat_64 :", feat_64.shape)
    print("feat_32 :", feat_32.shape)
    print("feat_16 :", feat_16.shape)

    print("\nOutput:")
    print("out_128 :", out_128.shape)
    print("out_64  :", out_64.shape)
    print("out_32  :", out_32.shape)
    print("out_16  :", out_16.shape)
