from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """2D 深度可分离卷积：DepthwiseConv2d(groups=in_channels) + PointwiseConv2d(1x1)。"""

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
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class MultiScaleGenerator(nn.Module):
    """多尺度序列生成器：通过重复平均池化生成不同长度的序列。"""

    def __init__(self, num_scales: int) -> None:
        super().__init__()
        self.num_scales = num_scales
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        输入: [B, C, L]
        输出: [x_orig, x_scale1, ..., x_scaleN] 共 num_scales+1 个张量
        """
        out = [x]
        curr = x
        for _ in range(self.num_scales):
            curr = self.pool(curr)
            out.append(curr)
        return out


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
    """基于 FFT 从序列中提取 top_k 个显著周期。

    参考 scheme_1_algo.md：对归一化后的多变量序列求和得到 1D 序列，再 FFT，选振幅最大的前 K 个频率。

    参数
    - x: [B, C, L] 或 [B, L]，数值序列
    返回
    - periods: [K]，周期长度（整数）
    - amplitudes: [K]，对应振幅（用于 softmax 权重）
    """

    if x.dim() == 3:
        # 对每个变量在时间维做 LayerNorm，再沿变量维求和
        b, c, l = x.shape
        x_ln = F.layer_norm(x, normalized_shape=(l,))
        seq = x_ln.sum(dim=1)  # [B, L]
    elif x.dim() == 2:
        seq = x
        l = x.shape[-1]
    else:
        raise ValueError("x 维度必须是 [B,C,L] 或 [B,L]")

    # rFFT: [B, L//2 + 1]
    fft = torch.fft.rfft(seq, dim=-1)
    mag = torch.abs(fft)  # [B, F]
    mag = mag.mean(dim=0)  # [F]
    if mag.numel() > 0:
        mag[0] = 0.0  # 去掉直流分量

    # 若长度过短，直接返回周期=1
    if mag.numel() <= 1:
        periods = torch.ones(top_k, dtype=torch.long, device=x.device)
        amps = torch.ones(top_k, dtype=torch.float32, device=x.device)
        return PeriodInfo(periods=periods, amplitudes=amps)

    k = min(top_k, mag.numel() - 1)
    top_vals, top_idx = torch.topk(mag[1:], k=k, largest=True)
    freq_idx = top_idx + 1  # 补回偏移

    # period ≈ L / f
    periods = torch.clamp((l // freq_idx).to(torch.long), min=1)
    amps = top_vals.to(torch.float32).clamp(min=eps)

    # 去重：保持顺序
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


def pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, int]:
    """在时间维（最后一维）做 0-padding，使长度变为 multiple 的整数倍。"""

    length = x.shape[-1]
    if multiple <= 0:
        raise ValueError("multiple 必须为正")
    pad_len = (multiple - (length % multiple)) % multiple
    if pad_len == 0:
        return x, length
    pad = (0, pad_len)
    x_pad = F.pad(x, pad, mode="constant", value=0.0)
    return x_pad, length


def reshape_1d_to_2d(x: torch.Tensor, period: int) -> Tuple[torch.Tensor, int]:
    """将 1D 序列按周期重塑为 2D 图像。

    输入 x: [B, C, L]
    输出 z: [B, C, rows, period]，其中 rows = ceil(L/period)
    同时返回原始长度（便于反向裁剪）。
    """

    x_pad, orig_len = pad_to_multiple(x, period)
    b, c, l_pad = x_pad.shape
    rows = l_pad // period
    z = x_pad.view(b, c, rows, period)
    return z, orig_len


def reshape_2d_to_1d(z: torch.Tensor, orig_len: int) -> torch.Tensor:
    """将 2D 特征展平回 1D，并裁剪回 orig_len。"""

    b, c, rows, period = z.shape
    x = z.reshape(b, c, rows * period)
    return x[..., :orig_len]
