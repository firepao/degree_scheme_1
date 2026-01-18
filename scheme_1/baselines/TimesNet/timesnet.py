"""TimesNet 模型核心实现。

基于官方实现简化：专注于预测任务，适配温室数据。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DataEmbedding, Inception_Block_V1


def FFT_for_Period(x: torch.Tensor, k: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """FFT 提取周期信息。

    Args:
        x: [B, T, C] 输入序列
        k: 提取 top-k 个主频率

    Returns:
        period: [k] 周期列表
        period_weight: [B, k] 每个周期的权重（振幅）
    """
    # 修复：cuFFT 在半精度模式下要求序列长度为 2 的幂次
    # 临时转换为 float32 进行 FFT 计算
    x_dtype = x.dtype
    x_float = x.float() if x.dtype == torch.float16 else x
    
    # [B, T, C] -> FFT
    xf = torch.fft.rfft(x_float, dim=1)

    # 计算频率的平均振幅：在 batch 和 channel 维度上平均
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 去除直流分量

    # 找到振幅最大的 k 个频率
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()

    # 周期 = 序列长度 / 频率索引
    period = x.shape[1] // top_list

    # 返回周期和对应的权重（转回原始数据类型）
    period_weight = abs(xf).mean(-1)[:, top_list]
    if x_dtype == torch.float16:
        period_weight = period_weight.half()
    
    return period, period_weight


class TimesBlock(nn.Module):
    """TimesNet 的核心模块：TimesBlock。

    关键思想：
    1. 通过 FFT 找到序列的主要周期
    2. 将 1D 序列按周期重塑为 2D 表示
    3. 用 2D 卷积提取周期内和周期间的信息
    4. 将多个周期的结果加权融合
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        top_k: int,
        num_kernels: int
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k

        # 参数高效设计：两层 Inception 卷积块
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N] 其中 T = seq_len + pred_len

        Returns:
            out: [B, T, N] 经过周期建模后的输出
        """
        B, T, N = x.size()

        # 提取 top-k 个周期及其权重
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # Padding：确保序列长度可以被周期整除
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # Reshape 为 2D：[B, length // period, period, N]
            # -> [B, N, length // period, period] 以便 2D 卷积
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()

            # 2D 卷积提取周期内和周期间的信息
            out = self.conv(out)

            # Reshape 回 1D：[B, N, length // period, period] -> [B, length, N]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)

            # 截断掉 padding 的部分
            res.append(out[:, :(self.seq_len + self.pred_len), :])

        # 堆叠所有周期的结果：[B, T, N, k]
        res = torch.stack(res, dim=-1)

        # 自适应聚合：用 softmax 归一化的权重加权求和
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # 残差连接
        res = res + x
        return res


@dataclass
class TimesNetConfig:
    """TimesNet 模型配置。"""

    input_dim: int          # 输入特征维度
    target_dim: int         # 目标特征维度
    seq_len: int            # 输入序列长度
    pred_len: int           # 预测序列长度
    d_model: int = 64       # Embedding 维度
    d_ff: int = 128         # Feedforward 维度
    e_layers: int = 2       # TimesBlock 层数
    top_k: int = 3          # 提取 top-k 个周期
    num_kernels: int = 6    # Inception Block 中的卷积核数量
    dropout: float = 0.1    # Dropout 率


class TimesNetForecaster(nn.Module):
    """TimesNet 时间序列预测模型（简化版）。

    适配温室数据的预测任务：
    - 输入：[B, seq_len, input_dim]
    - 输出：[B, pred_len, target_dim]

    模型流程：
    1. Embedding：将输入序列嵌入到 d_model 维空间
    2. 时序扩展：通过线性层将 seq_len 扩展到 seq_len + pred_len
    3. TimesBlock：堆叠多层 TimesBlock 提取周期信息
    4. 投影：将 d_model 投影回 target_dim
    5. 归一化：使用输入的均值和方差进行反归一化
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        top_k: int = 3,
        num_kernels: int = 6,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.target_dim = target_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.e_layers = e_layers

        # Embedding 层：将输入特征嵌入到 d_model 维
        self.enc_embedding = DataEmbedding(
            c_in=input_dim,
            d_model=d_model,
            dropout=dropout
        )

        # 时序扩展：线性层将 seq_len -> seq_len + pred_len
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)

        # TimesBlock 堆叠
        self.model = nn.ModuleList([
            TimesBlock(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=d_model,
                d_ff=d_ff,
                top_k=top_k,
                num_kernels=num_kernels
            )
            for _ in range(e_layers)
        ])

        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)

        # 投影层：d_model -> target_dim
        self.projection = nn.Linear(d_model, target_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, input_dim]

        Returns:
            y_hat: [B, pred_len, target_dim]
        """
        # 1. 归一化（Non-stationary Transformer 风格）
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        # 2. Embedding：[B, seq_len, input_dim] -> [B, seq_len, d_model]
        enc_out = self.enc_embedding(x)

        # 3. 时序扩展：[B, seq_len, d_model] -> [B, seq_len + pred_len, d_model]
        # 通过线性层在时序维度上扩展
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. TimesBlock 堆叠
        for i in range(self.e_layers):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 5. 投影：[B, seq_len + pred_len, d_model] -> [B, seq_len + pred_len, target_dim]
        dec_out = self.projection(enc_out)

        # 6. 反归一化
        dec_out = dec_out * (stdev[:, 0, :self.target_dim].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :self.target_dim].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))

        # 7. 只返回预测部分：[B, pred_len, target_dim]
        return dec_out[:, -self.pred_len:, :]
