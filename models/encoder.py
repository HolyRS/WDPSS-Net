import torch
import torch.nn as nn
#from models.Mamba_backbone import Backbone_VSSM
from .resnet import Backbone
from .FM import FrequencyModule
from .SSM import SpatialModule1, SpatialModule2
import torchvision.models as models

class Encoder(nn.Module):
    """
    双时相 Encoder 总模块

    流程：
    1. 双时相图像进入 Backbone
    2. 128 / 64 尺度先经过 FM，再进入高层 SSM
    3. 32 / 16 尺度直接进入低层 SSM
    4. 最终输出 4 个尺度特征图
    """
    def __init__(self):
        super().__init__()

        # Remove the explicitly passed args from kwargs to avoid "got multiple values" error
        #clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        #self.backbone = Backbone()
        resnet18 = models.resnet18(pretrained=True)
        # 获取ResNet18的卷积层和全连接层
        self.features = nn.Sequential(
            *list(resnet18.children())[:-2]  # 去掉ResNet-18最后的全连接层和平均池化层
        )

        # 保留ResNet-18的四个下采样输出
        self.layer1 = resnet18.layer1  # 输出: 64
        self.layer2 = resnet18.layer2  # 输出: 128
        self.layer3 = resnet18.layer3  # 输出: 256
        self.layer4 = resnet18.layer4  # 输出: 512
        # 这里的通道数按你的 backbone 实际输出改
        self.fm_128 = FrequencyModule(channels=64)
        self.fm_64  = FrequencyModule(channels=64)

        self.ssm_128 = SpatialModule1(channels=64)
        self.ssm_64  = SpatialModule1(channels=64)
        self.ssm_32  = SpatialModule2(channels=128)
        self.ssm_16  = SpatialModule2(channels=256)

    def forward(self, x1, x2):
        # Backbone 输出 8 个特征图
        #(
            #output_128_1, output_64_1, output_32_1, output_16_1,
            #output_128_2, output_64_2, output_32_2, output_16_2
        #) = self.backbone(x1, x2)
        # 经过第一层卷积和最大池化后，输出128x128大小
        x1 = self.features[0](x1)  # Conv1 (7x7卷积，步长2)
        x1 = self.features[1](x1)  # BN1
        x1 = self.features[2](x1)  # ReLU
        output_128_1 = x1  # 输出尺寸: [batch_size, 64, 128, 128]
        # 经过第一层卷积和最大池化后，输出128x128大小
        x2 = self.features[0](x2)  # Conv1 (7x7卷积，步长2)
        x2 = self.features[1](x2)  # BN1
        x2 = self.features[2](x2)  # ReLU
        output_128_2 = x2  # 输出尺寸: [batch_size, 64, 128, 128]
        # =========================
        # 128 尺度：先 FM，再 SSM
        # =========================
        (
            HH1_128, HL1_128, LH1_128,
            HH2_128, HL2_128, LH2_128,
            #HH3_128, HL3_128, LH3_128,
            J1_128, J2_128#, Y_128
        ) = self.fm_128(output_128_1, output_128_2)

        feat_128_1,feat_128_2 = self.ssm_128(
            F1=output_128_1,
            F2=output_128_2,
            G1=J1_128,
            G2=J2_128,
            #G3=Y_128,
            HH1=HH1_128, HL1=HL1_128, LH1=LH1_128,
            HH2=HH2_128, HL2=HL2_128, LH2=LH2_128,
            #HH3=HH3_128, HL3=HL3_128, LH3=LH3_128
        )
        output_64_1=self.features[3](feat_128_1)
        output_64_1=self.layer1(output_64_1)
        output_64_2 = self.features[3](feat_128_2)
        output_64_2 = self.layer1(output_64_2)
        # =========================
        # 64 尺度：先 FM，再 SSM
        # =========================
        (
            HH1_64, HL1_64, LH1_64,
            HH2_64, HL2_64, LH2_64,
            #HH3_64, HL3_64, LH3_64,
            J1_64, J2_64#, Y_64
        ) = self.fm_64(output_64_1, output_64_2)

        feat_64_1,feat_64_2 = self.ssm_64(
            F1=output_64_1,
            F2=output_64_2,
            G1=J1_64,
            G2=J2_64,
            #G3=Y_64,
            HH1=HH1_64, HL1=HL1_64, LH1=LH1_64,
            HH2=HH2_64, HL2=HL2_64, LH2=LH2_64,
            #HH3=HH3_64, HL3=HL3_64, LH3=LH3_64
        )

        output_32_1 = self.layer2(feat_64_1)

        output_32_2 = self.layer2(feat_64_2)
        # =========================
        # 32 尺度：直接 SSM
        # =========================
        feat_32_1,feat_32_2 = self.ssm_32(
            F1=output_32_1,
            F2=output_32_2
        )
        output_16_1 = self.layer3(feat_32_1)

        output_16_2 = self.layer3(feat_32_2)
        # =========================
        # 16 尺度：直接 SSM
        # =========================
        feat_16_1,feat_16_2 = self.ssm_16(
            F1=output_16_1,
            F2=output_16_2
        )


        return feat_128_1, feat_128_2,feat_64_1,feat_64_2 ,feat_32_1,feat_32_2, feat_16_1,feat_16_2
if __name__ == "__main__":
    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = Encoder().to(device)
    # 2. 构造双时相输入
    x1 = torch.randn(2, 3, 256, 256).to(device)
    x2 = torch.randn(2, 3, 256, 256).to(device)

    # 3. 前向传播
    feat_128_1, feat_128_2,feat_64_1,feat_64_2 ,feat_32_1,feat_32_2, feat_16_1,feat_16_2 = model(x1, x2)

    # 4. 打印结果
    print("feat_128 shape:", feat_128_1.shape)
    print("feat_64  shape:", feat_64_2.shape)
    print("feat_32  shape:", feat_32_1.shape)
    print("feat_16  shape:", feat_16_2.shape)