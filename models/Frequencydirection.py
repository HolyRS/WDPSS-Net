import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencySubbandGate(nn.Module):
    """
    基于频率子带的扫描方向偏好模块

    输入:
        HH, HL, LH: [B, C, h, w]

    输出:
        gate: [B, C, 4, H, W] 或 [B, C, 4, h, w]

    方向顺序固定为:
        [0°, 270°, 180°, 90°]

    对应映射:
        0°   -> 水平扫描偏好
        270° -> 垂直扫描偏好
        180° -> 水平扫描偏好
        90°  -> 垂直扫描偏好
    """
    def __init__(self, beta=1.0, eps=1e-6, smooth_kernel=3):
        super(FrequencySubbandGate, self).__init__()
        self.beta = beta
        self.eps = eps
        self.smooth_kernel = smooth_kernel

    def _smooth_energy(self, x):
        return F.avg_pool2d(
            x,
            kernel_size=self.smooth_kernel,
            stride=1,
            padding=self.smooth_kernel // 2
        )

    def forward(self, HH, HL, LH, target_size=None):
        """
        HH, HL, LH: [B, C, h, w]
        target_size: (H, W) or None

        return:
            gate: [B, C, 4, H, W] or [B, C, 4, h, w]
        """
        # 1. 局部能量
        Ex = self._smooth_energy(HL ** 2)   # x方向变化能量
        Ey = self._smooth_energy(LH ** 2)   # y方向变化能量
        Ed = self._smooth_energy(HH ** 2)   # 对角复杂细节能量

        # 2. 扫描方向偏好
        # Ey大 -> 更适合水平扫描
        # Ex大 -> 更适合垂直扫描
        Ph = Ey / (Ex + Ey + self.eps)   # 水平扫描偏好
        Pv = Ex / (Ex + Ey + self.eps)   # 垂直扫描偏好

        # 3. 方向置信度
        rho = torch.abs(Ex - Ey) / (Ex + Ey + self.eps)

        # 4. HH复杂细节抑制
        q = torch.exp(-self.beta * Ed / (Ex + Ey + self.eps))

        # 5. 最终置信度
        m = rho * q

        # 6. 平滑后的扫描偏好
        Ph_tilde = m * Ph + (1.0 - m) * 0.5
        Pv_tilde = m * Pv + (1.0 - m) * 0.5

        # 7. 按顺序 [0°, 270°, 180°, 90°] 组织输出
        gate = torch.stack(
            [Ph_tilde, Pv_tilde, Ph_tilde, Pv_tilde],
            dim=2
        )   # [B, C, 4, h, w]

        # 8. 上采样到目标尺寸
        if target_size is not None:
            B, C, _, h, w = gate.shape
            H, W = target_size
            gate = gate.contiguous().view(B, C * 4, h, w)
            gate = F.interpolate(gate, size=(H, W), mode='bilinear', align_corners=False)
            gate = gate.view(B, C, 4, H, W)

        return gate