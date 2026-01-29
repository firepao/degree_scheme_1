"""TPLC 模型核心组件 - 消融实验用。

包含 TPLC 模型的所有核心组件的独立实现，用于消融实验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. FFT 周期提取模块
# ============================================================

@dataclass(frozen=True)
class PeriodInfo:
    """FFT 选取的周期信息。"""
    periods: torch.Tensor  # [K] int64
    amplitudes: torch.Tensor  # [K] float32


def extract_topk_periods(
    x: torch.Tensor,
    top_k: int,
    eps: float = 1e-6,
) -> PeriodInfo:
    """基于 FFT 从序列中提取 top_k 个显著周期。"""
    if x.dim() == 3:
        b, c, l = x.shape
        x_ln = F.layer_norm(x, normalized_shape=(l,))
        seq = x_ln.sum(dim=1)  # [B, L]
    elif x.dim() == 2:
        seq = x
        l = x.shape[-1]
    else:
        raise ValueError("x 维度必须是 [B,C,L] 或 [B,L]")

    fft = torch.fft.rfft(seq, dim=-1)
    mag = torch.abs(fft).mean(dim=0)
    if mag.numel() > 0:
        mag[0] = 0.0

    if mag.numel() <= 1:
        periods = torch.ones(top_k, dtype=torch.long, device=x.device)
        amps = torch.ones(top_k, dtype=torch.float32, device=x.device)
        return PeriodInfo(periods=periods, amplitudes=amps)

    k = min(top_k, mag.numel() - 1)
    top_vals, top_idx = torch.topk(mag[1:], k=k, largest=True)
    freq_idx = top_idx + 1

    periods = torch.clamp((l // freq_idx).to(torch.long), min=1)
    amps = top_vals.to(torch.float32).clamp(min=eps)

    uniq_periods: list[int] = []
    uniq_amps: list[float] = []
    for p, a in zip(periods.tolist(), amps.tolist()):
        if p not in uniq_periods:
            uniq_periods.append(p)
            uniq_amps.append(a)
        if len(uniq_periods) >= top_k:
            break

    while len(uniq_periods) < top_k:
        uniq_periods.append(1)
        uniq_amps.append(float(eps))

    return PeriodInfo(
        periods=torch.tensor(uniq_periods, dtype=torch.long, device=x.device),
        amplitudes=torch.tensor(uniq_amps, dtype=torch.float32, device=x.device),
    )


# ============================================================
# 2. 1D→2D 重塑模块
# ============================================================

def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """在时间维做 0-padding，使长度变为 multiple 的整数倍。"""
    length = x.shape[-1]
    pad_len = (multiple - (length % multiple)) % multiple
    if pad_len == 0:
        return x, length
    x_pad = F.pad(x, (0, pad_len), mode="constant", value=0.0)
    return x_pad, length


def reshape_1d_to_2d(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """将 1D 序列按周期重塑为 2D 图像。"""
    x_pad, orig_len = pad_to_multiple(x, period)
    b, c, l_pad = x_pad.shape
    rows = l_pad // period
    z = x_pad.view(b, c, rows, period)
    return z, orig_len


def reshape_2d_to_1d(z: torch.Tensor, orig_len: int) -> torch.Tensor:
    """将 2D 特征展平回 1D。"""
    b, c, rows, period = z.shape
    x = z.reshape(b, c, rows * period)
    return x[..., :orig_len]


# ============================================================
# 3. 深度可分离卷积模块
# ============================================================

class DepthwiseSeparableConv2d(nn.Module):
    """2D 深度可分离卷积：DepthwiseConv + PointwiseConv。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, padding=padding,
            groups=in_channels, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


# ============================================================
# 4. 标准 2D 卷积模块（消融对照）
# ============================================================

class StandardConv2d(nn.Module):
    """标准 2D 卷积（用于与深度可分离卷积对比）。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# ============================================================
# 5. 多尺度生成器
# ============================================================

class MultiScaleGenerator(nn.Module):
    """多尺度序列生成器。"""

    def __init__(self, num_scales: int) -> None:
        super().__init__()
        self.num_scales = num_scales

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        scales = [x]
        cur = x
        for _ in range(self.num_scales):
            cur = F.avg_pool1d(cur, kernel_size=2, stride=2, ceil_mode=False)
            scales.append(cur)
        return scales


# ============================================================
# 6. RevIN 归一化模块
# ============================================================

class RevIN(nn.Module):
    """Reversible Instance Normalization for Time Series."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = x.mean(dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(x.var(dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        x = x * self.stdev + self.mean
        return x


# ============================================================
# 7. Inception Block 模块
# ============================================================

class InceptionBlock(nn.Module):
    """Inception Block: 多尺度卷积核并行处理。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        kernels = []
        for i in range(num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2*i+1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for kernel in self.kernels:
            res_list.append(kernel(x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# ============================================================
# 8. 频域增强模块
# ============================================================

class FrequencyEnhancement(nn.Module):
    """频域增强：学习频域权重。"""

    def __init__(self, seq_len: int):
        super().__init__()
        self.freq_len = seq_len // 2 + 1
        self.freq_weight = nn.Parameter(torch.ones(self.freq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        x_fft = torch.fft.rfft(x, dim=-1)
        x_fft = x_fft * self.freq_weight.unsqueeze(0).unsqueeze(0)
        x_out = torch.fft.irfft(x_fft, n=x.size(-1), dim=-1)
        return x_out


# ============================================================
# 9. 季节-趋势分解模块
# ============================================================

class MovingAvgDecomp(nn.Module):
    """移动平均季节-趋势分解。"""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C] 或 [B, C, L]
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            # [B, L, C] -> [B, C, L]
            x = x.transpose(1, 2)
            transposed = True
        else:
            transposed = False
        
        # 前后填充
        front = x[:, :, :1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x_pad = torch.cat([front, x, end], dim=-1)
        
        trend = self.avg(x_pad)
        seasonal = x - trend
        
        if transposed:
            trend = trend.transpose(1, 2)
            seasonal = seasonal.transpose(1, 2)
        
        return seasonal, trend


class DFTDecomp(nn.Module):
    """DFT 季节-趋势分解。"""

    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, L, C]
        xf = torch.fft.rfft(x, dim=1)
        freq_list = abs(xf).mean(0).mean(-1)
        freq_list[0] = 0
        
        _, top_indices = torch.topk(freq_list, self.top_k)
        top_indices = top_indices.detach().cpu().numpy()
        
        mask = torch.zeros_like(xf)
        mask[:, top_indices, :] = 1
        
        x_season = torch.fft.irfft(xf * mask, n=x.size(1), dim=1)
        x_trend = x - x_season
        
        return x_season, x_trend


# ============================================================
# 10. 自适应注意力融合模块
# ============================================================

class AdaptiveScaleFusion(nn.Module):
    """自适应多尺度融合（带注意力）。"""

    def __init__(self, num_scales: int, hidden_dim: int):
        super().__init__()
        self.num_scales = num_scales
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_scales),
            nn.Softmax(dim=-1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, scale_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            scale_features: List of [B, hidden_dim, L_m] tensors
        Returns:
            weights: [B, num_scales]
            (用于后续加权融合)
        """
        # 对每个尺度做全局池化得到表示
        reps = []
        for feat in scale_features:
            rep = self.pool(feat).squeeze(-1)  # [B, hidden_dim]
            reps.append(rep)
        
        # 拼接所有尺度表示取平均
        combined = torch.stack(reps, dim=1).mean(dim=1)  # [B, hidden_dim]
        
        # 计算注意力权重
        weights = self.attention(combined)  # [B, num_scales]
        
        return weights


# ============================================================
# 11. 简单 MLP 预测头（消融对照）
# ============================================================

class MLPPredictor(nn.Module):
    """简单 MLP 预测头（不使用 FFT 和多尺度）。"""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.flatten_dim = seq_len * input_dim
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim]
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        out = self.net(x_flat)
        return out.reshape(b, self.pred_len, self.target_dim)


# ============================================================
# 12. 简单 1D 卷积预测（消融对照）
# ============================================================

class Conv1DPredictor(nn.Module):
    """1D 卷积预测（不使用 2D 重塑）。"""

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act = nn.GELU()
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.channel_proj = nn.Linear(hidden_dim, target_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, input_dim]
        x = x.transpose(1, 2)  # [B, input_dim, seq_len]
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        # [B, hidden_dim, seq_len]
        x = self.time_proj(x)  # [B, hidden_dim, pred_len]
        x = x.transpose(1, 2)  # [B, pred_len, hidden_dim]
        x = self.channel_proj(x)  # [B, pred_len, target_dim]
        return x
