import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet18Backbone, self).__init__()

        # 使用预训练的ResNet-18
        resnet18 = models.resnet18(pretrained=pretrained)

        # 获取ResNet18的卷积层和全连接层
        self.features = nn.Sequential(
            *list(resnet18.children())[:-2]  # 去掉ResNet-18最后的全连接层和平均池化层
        )

        # 保留ResNet-18的四个下采样输出
        self.layer1 = resnet18.layer1  # 输出: 64
        self.layer2 = resnet18.layer2  # 输出: 128
        self.layer3 = resnet18.layer3  # 输出: 256
        self.layer4 = resnet18.layer4  # 输出: 512

    def forward(self, x):
        # 经过第一层卷积和最大池化后，输出128x128大小
        x = self.features[0](x)  # Conv1 (7x7卷积，步长2)
        x = self.features[1](x)  # BN1
        x = self.features[2](x)  # ReLU
        output_128 = x  # 输出尺寸: [batch_size, 64, 128, 128]
        x = self.features[3](x)  # MaxPool


        # 第一层残差块后输出64x64大小
        x = self.layer1(x)  # 输出尺寸: [batch_size, 64, 64, 64]
        output_64 = x

        # 第二层残差块后输出32x32大小
        x = self.layer2(x)  # 输出尺寸: [batch_size, 128, 32, 32]
        output_32 = x

        # 第三层残差块后输出16x16大小
        x = self.layer3(x)  # 输出尺寸: [batch_size, 256, 16, 16]
        output_16 = x

        # 第四层残差块后输出8x8大小
        x = self.layer4(x)  # 输出尺寸: [batch_size, 512, 8, 8]
        # 返回四个特征图
        return output_128, output_64, output_32, output_16


# 变化检测网络（共享ResNet权重）
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

        # 创建一个ResNet-18 Backbone，共享权重
        self.resnet_backbone = ResNet18Backbone(pretrained=True)

    def forward(self, x1, x2):
        # 第一张输入图像经过ResNet-18 Backbone
        output_128_1, output_64_1, output_32_1, output_16_1 = self.resnet_backbone(x1)

        # 第二张输入图像也经过同一个ResNet-18 Backbone（共享权重）
        output_128_2, output_64_2, output_32_2, output_16_2 = self.resnet_backbone(x2)

        # 返回8个特征图
        return output_128_1, output_64_1, output_32_1, output_16_1, output_128_2, output_64_2, output_32_2, output_16_2


