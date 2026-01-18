from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import (
    DepthwiseSeparableConv2d,
    extract_topk_periods,
    reshape_1d_to_2d,
    reshape_2d_to_1d,
)


class MultiScaleGenerator(nn.Module):
    """多尺度序列生成：使用平均池化实现 stride=2 的下采样。"""

    def __init__(self, num_scales: int) -> None:
        super().__init__()
        self.num_scales = int(num_scales)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: [B, C, T]
        scales = [x]
        cur = x
        for _ in range(self.num_scales):
            # 平均池化下采样，保持变量一致
            cur = F.avg_pool1d(cur, kernel_size=2, stride=2, ceil_mode=False)
            scales.append(cur)
        return scales


class TPLCNet(nn.Module):
    """TPLCNet：多尺度 + FFT 周期识别 + 1D→2D + 深度可分离卷积 + 多周期/多尺度融合。"""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        num_scales: int = 2,
        top_k_periods: int = 3,
        hidden_dim: int = 64,
        dw_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.num_scales = int(num_scales)
        self.top_k_periods = int(top_k_periods)
        self.hidden_dim = int(hidden_dim)

        self.multi_scale = MultiScaleGenerator(num_scales=self.num_scales)
        self.conv2d = DepthwiseSeparableConv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=dw_kernel,
        )

        # 多尺度 head：每个尺度一个“时间映射线性层”，再共享一个“通道映射线性层”
        self.time_projs = nn.ModuleList()
        for m in range(self.num_scales + 1):
            l_m = self.seq_len // (2**m)
            l_m = max(1, l_m)
            self.time_projs.append(nn.Linear(l_m, self.pred_len))
        self.channel_proj = nn.Linear(self.hidden_dim, self.target_dim)

        # 多尺度融合权重：softmax 后再乘以 (num_scales+1)，使初始化等价于“直接相加”
        self.scale_logits = nn.Parameter(torch.zeros(self.num_scales + 1))        
        # Dropout 层用于正则化（默认 0.1）
        self.dropout = nn.Dropout(p=0.1)
        
        # 残差投影层（如果输入维度与输出维度不匹配）
        self.residual_proj = None
        if self.input_dim != self.target_dim:
            self.residual_proj = nn.Linear(self.input_dim, self.target_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向。

        参数
        - x: [B, seq_len, input_dim]
        返回
        - y_hat: [B, pred_len, target_dim]
        """

        if x.dim() != 3:
            raise ValueError("x 必须是 [B, T, C]")
        b, t, c = x.shape
        if t != self.seq_len:
            raise ValueError(f"seq_len 不匹配：输入 T={t}, 期望 {self.seq_len}")
        if c != self.input_dim:
            raise ValueError(f"input_dim 不匹配：输入 C={c}, 期望 {self.input_dim}")

        # 🔧 保存原始输入用于残差连接
        x_residual = x  # [B, seq_len, input_dim]
        
        x_ct = x.transpose(1, 2).contiguous()  # [B, C, T]
        scale_list = self.multi_scale(x_ct)

        # [M]，M=num_scales+1；初始化时每个尺度权重为 1（与旧版直接相加一致）
        m_scales = self.num_scales + 1
        scale_weights = torch.softmax(self.scale_logits, dim=0) * float(m_scales)

        y_sum = None
        for m, x_m in enumerate(scale_list):
            # x_m: [B, C, Lm]
            period_info = extract_topk_periods(x_m, top_k=self.top_k_periods)
            periods = period_info.periods  # [K]
            amps = period_info.amplitudes  # [K]
            weights = torch.softmax(amps, dim=0)  # [K]

            feats_1d = []
            for k in range(self.top_k_periods):
                p = int(periods[k].item())
                z2d, orig_len = reshape_1d_to_2d(x_m, period=p)  # [B,C,rows,p]
                y2d = self.conv2d(z2d)  # [B, hidden, rows, p]
                y1d = reshape_2d_to_1d(y2d, orig_len=orig_len)  # [B, hidden, Lm]
                feats_1d.append(y1d)

            # 多周期融合：x_m^l = sum_k softmax(A_k) * X_{m,k}
            fused = torch.zeros_like(feats_1d[0])
            for k in range(self.top_k_periods):
                fused = fused + weights[k] * feats_1d[k]
            
            # 🔧 新增：应用 Dropout
            fused = self.dropout(fused)

            # 预测头：时间维 Lm -> pred_len，再 hidden -> target_dim
            # fused: [B, hidden, Lm]
            l_m = fused.shape[-1]
            # time_proj_m 期望输入长度与构造时一致；若因池化导致 Lm 与整除不一致，做截断/补零
            expected_lm = self.time_projs[m].in_features
            if l_m > expected_lm:
                fused_use = fused[..., -expected_lm:]
            elif l_m < expected_lm:
                fused_use = F.pad(fused, (expected_lm - l_m, 0), value=0.0)
            else:
                fused_use = fused

            # [B, hidden, pred]
            pred_hidden = self.time_projs[m](fused_use)
            pred_hidden = pred_hidden.transpose(1, 2).contiguous()  # [B, pred, hidden]
            pred_hidden = self.dropout(pred_hidden)  # 🔧 新增：Dropout
            y_m = self.channel_proj(pred_hidden)  # [B, pred, target]
            w_m = scale_weights[m]
            y_sum = (w_m * y_m) if y_sum is None else (y_sum + w_m * y_m)

        assert y_sum is not None
        
        # 🔧 新增：残差连接（TimesNet 风格）
        # 取输入序列的后 pred_len 步作为残差基准
        if self.seq_len >= self.pred_len:
            x_res = x_residual[:, -self.pred_len:, :]  # [B, pred_len, input_dim]
        else:
            # 如果序列长度不足，用零填充
            x_res = F.pad(x_residual, (0, 0, self.pred_len - self.seq_len, 0))[:, :self.pred_len, :]
        
        # 如果维度不匹配，使用投影层
        if self.residual_proj is not None:
            x_res = self.residual_proj(x_res)  # [B, pred_len, target_dim]
        
        # 残差连接
        y_sum = y_sum + x_res
        
        return y_sum
