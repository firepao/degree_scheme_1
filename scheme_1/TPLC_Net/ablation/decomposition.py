"""季节-趋势分解模块。

参考：
- TimeMixer (ICLR 2024) - DFT_series_decomp
- Autoformer (NeurIPS 2021) - series_decomp (Moving Average)

功能：
将时序数据分解为季节性（周期性）成分和趋势性成分，
分别处理后再融合，提升模型对不同模式的建模能力。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFT_series_decomp(nn.Module):
    """基于 DFT（离散傅里叶变换）的序列分解。
    
    保留振幅最大的 top_k 个频率成分作为季节性，其余作为趋势性。
    
    原理：
    1. 对输入序列做 FFT
    2. 保留振幅最大的 top_k 个频率
    3. IFFT 得到季节性成分
    4. 原序列 - 季节性 = 趋势性
    
    Args:
        top_k: 保留的主频率数量
    
    输入: [B, T, C]
    输出: (x_season, x_trend)，各为 [B, T, C]
    """
    
    def __init__(self, top_k: int = 5) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。"""
        # x: [B, T, C]
        
        # FFT：[B, T//2+1, C]（复数）
        xf = torch.fft.rfft(x, dim=1)
        
        # 计算振幅
        freq = torch.abs(xf)
        
        # 去除直流分量（频率 0）
        freq[:, 0, :] = 0
        
        # 找到振幅最大的 top_k 个频率
        top_k_freq, _ = torch.topk(freq, k=min(self.top_k, freq.shape[1]), dim=1)
        
        # 过滤掉低振幅频率：振幅 <= top_k 最小值的设为 0
        threshold = top_k_freq.min(dim=1, keepdim=True)[0]
        xf_filtered = xf.clone()
        xf_filtered[freq <= threshold] = 0
        
        # IFFT 得到季节性成分
        x_season = torch.fft.irfft(xf_filtered, n=x.shape[1], dim=1)
        
        # 趋势性 = 原序列 - 季节性
        x_trend = x - x_season
        
        return x_season, x_trend


class series_decomp(nn.Module):
    """基于移动平均的序列分解（Autoformer 风格）。
    
    使用平均池化提取趋势性，原序列 - 趋势 = 季节性。
    
    Args:
        kernel_size: 移动平均窗口大小（奇数）
    
    输入: [B, T, C]
    输出: (x_season, x_trend)，各为 [B, T, C]
    """
    
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        # 确保是奇数
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1
        
        self.avg_pool = nn.AvgPool1d(
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播。"""
        # x: [B, T, C]
        
        # 转换为 [B, C, T] 以便 AvgPool1d
        x_t = x.permute(0, 2, 1)
        
        # 填充以保持长度
        padding = self.kernel_size // 2
        x_padded = F.pad(x_t, (padding, padding), mode='reflect')
        
        # 移动平均得到趋势
        x_trend = self.avg_pool(x_padded)
        
        # 转回 [B, T, C]
        x_trend = x_trend.permute(0, 2, 1)
        
        # 季节性 = 原序列 - 趋势
        x_season = x - x_trend
        
        return x_season, x_trend


class MultiScaleDecomp(nn.Module):
    """多尺度分解模块。
    
    使用多个不同窗口大小的移动平均，提取不同粒度的趋势。
    
    Args:
        kernel_sizes: 多个窗口大小列表
    
    输入: [B, T, C]
    输出: (x_season, [x_trend_1, x_trend_2, ...])
    """
    
    def __init__(self, kernel_sizes: list[int] = [13, 25, 49]) -> None:
        super().__init__()
        self.decomps = nn.ModuleList([
            series_decomp(k) for k in kernel_sizes
        ])
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """前向传播。"""
        trends = []
        season_sum = torch.zeros_like(x)
        
        for decomp in self.decomps:
            season, trend = decomp(x)
            season_sum = season_sum + season
            trends.append(trend)
        
        # 平均季节性
        x_season = season_sum / len(self.decomps)
        
        return x_season, trends


class DecompLayer(nn.Module):
    """分解层：封装分解 + 可选的线性投影。
    
    用于在模型中间层使用分解。
    
    Args:
        decomp_type: 分解类型 'dft' | 'moving_avg'
        top_k: DFT 分解的 top_k
        kernel_size: 移动平均的窗口大小
        d_model: 特征维度（用于投影）
    """
    
    def __init__(
        self,
        decomp_type: str = 'dft',
        top_k: int = 5,
        kernel_size: int = 25,
        d_model: int | None = None,
    ) -> None:
        super().__init__()
        
        if decomp_type == 'dft':
            self.decomp = DFT_series_decomp(top_k=top_k)
        elif decomp_type == 'moving_avg':
            self.decomp = series_decomp(kernel_size=kernel_size)
        else:
            raise ValueError(f"未知的分解类型: {decomp_type}")
        
        # 可选的投影层
        self.season_proj = None
        self.trend_proj = None
        if d_model is not None:
            self.season_proj = nn.Linear(d_model, d_model)
            self.trend_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        return_separate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """前向传播。
        
        Args:
            x: [B, T, C] 输入
            return_separate: 是否分别返回季节和趋势
        
        Returns:
            如果 return_separate=False，返回 season + trend
            如果 return_separate=True，返回 (season, trend)
        """
        season, trend = self.decomp(x)
        
        if self.season_proj is not None:
            season = self.season_proj(season)
        if self.trend_proj is not None:
            trend = self.trend_proj(trend)
        
        if return_separate:
            return season, trend
        else:
            return season + trend
