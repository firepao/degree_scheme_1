"""多层堆叠 TPLC Block 模块。

参考：
- TimesNet (ICLR 2023) - 多层 TimesBlock + LayerNorm
- Transformer - 堆叠 Encoder Layer

功能：
将 TPLC 的核心处理逻辑封装为 Block，支持多层堆叠以增强特征提取能力。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入 TPLC 的基础组件
import sys
from pathlib import Path

# 添加父目录到路径以支持导入
_current_dir = Path(__file__).parent
_tplc_algo_path = _current_dir.parent / 'tplc_algo'
if str(_tplc_algo_path) not in sys.path:
    sys.path.insert(0, str(_tplc_algo_path))

try:
    from tplc_algo.models.layers import (
        DepthwiseSeparableConv2d,
        extract_topk_periods,
        reshape_1d_to_2d,
        reshape_2d_to_1d,
    )
except ImportError:
    # 备用：使用相对导入
    from ..tplc_algo.models.layers import (
        DepthwiseSeparableConv2d,
        extract_topk_periods,
        reshape_1d_to_2d,
        reshape_2d_to_1d,
    )


class TPLCBlock(nn.Module):
    """TPLC 单层处理块。
    
    封装了 TPLC 的核心逻辑：FFT 周期提取 -> 1D→2D -> 卷积 -> 2D→1D -> 周期融合
    
    Args:
        d_model: 特征维度
        seq_len: 序列长度（保留参数，兼容性）
        pred_len: 预测长度（保留参数，兼容性）
        top_k: 提取的主周期数量
        num_scales: 尺度数量（保留参数，兼容性）
        use_inception: 是否使用 Inception Block（替代 DepthwiseSep）
        dw_kernel: 深度可分离卷积核大小
        dropout: Dropout 概率
    
    输入: [B, T, C]
    输出: [B, T, C]（残差连接后）
    """
    
    def __init__(
        self,
        d_model: int,
        seq_len: int = 96,
        pred_len: int = 24,
        top_k: int = 3,
        num_scales: int = 2,
        use_inception: bool = False,
        dw_kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.top_k_periods = top_k
        self.use_inception = use_inception
        
        # 选择卷积模块
        if use_inception:
            from inception import Inception_Block_V1
            self.conv2d = nn.Sequential(
                Inception_Block_V1(d_model, d_model, num_kernels=6),
                nn.GELU(),
                Inception_Block_V1(d_model, d_model, num_kernels=6),
            )
        else:
            # 深度可分离卷积
            self.conv2d = DepthwiseSeparableConv2d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=dw_kernel,
            )
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [B, T, C] 输入
        
        Returns:
            [B, T, C] 输出（含残差）
        """
        B, T, C = x.shape
        
        # 保存残差
        residual = x
        
        # 转置为 [B, C, T]
        x_ct = x.transpose(1, 2).contiguous()
        
        # FFT 提取周期
        period_info = extract_topk_periods(x_ct, top_k=self.top_k_periods)
        periods = period_info.periods
        amps = period_info.amplitudes
        weights = torch.softmax(amps, dim=0)
        
        feats_1d = []
        for k in range(self.top_k_periods):
            p = int(periods[k].item())
            
            # 1D -> 2D
            z2d, orig_len = reshape_1d_to_2d(x_ct, period=p)
            
            # 2D 卷积
            y2d = self.conv2d(z2d)
            
            # 2D -> 1D
            y1d = reshape_2d_to_1d(y2d, orig_len=orig_len)
            feats_1d.append(y1d)
        
        # 多周期融合
        fused = torch.zeros_like(feats_1d[0])
        for k in range(self.top_k_periods):
            fused = fused + weights[k] * feats_1d[k]
        
        # Dropout
        fused = self.dropout(fused)
        
        # 转回 [B, T, C]
        out = fused.transpose(1, 2).contiguous()
        
        # 残差连接
        out = out + residual
        
        return out


class StackedTPLC(nn.Module):
    """堆叠多层 TPLC Block。
    
    类似 Transformer Encoder 的堆叠方式：
    for block in blocks:
        x = layer_norm(block(x))
    
    Args:
        d_model: 特征维度
        e_layers: 堆叠层数
        top_k_periods: 提取的主周期数量
        dw_kernel: 卷积核大小
        dropout: Dropout 概率
    
    输入: [B, T, C]
    输出: [B, T, C]
    """
    
    def __init__(
        self,
        d_model: int,
        e_layers: int = 2,
        top_k_periods: int = 3,
        dw_kernel: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.e_layers = e_layers
        
        # 堆叠 TPLC Block
        self.blocks = nn.ModuleList([
            TPLCBlock(
                d_model=d_model,
                top_k_periods=top_k_periods,
                dw_kernel=dw_kernel,
                dropout=dropout,
            )
            for _ in range(e_layers)
        ])
        
        # LayerNorm
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        for block in self.blocks:
            x = self.layer_norm(block(x))
        return x


class MultiScaleStackedTPLC(nn.Module):
    """多尺度 + 多层堆叠 TPLC。
    
    结合原始 TPLC 的多尺度设计和 TimesNet 的多层堆叠。
    
    Args:
        input_dim: 输入维度
        d_model: 隐藏维度
        num_scales: 尺度数量
        e_layers: 每个尺度的堆叠层数
        top_k_periods: 周期数量
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_scales: int = 2,
        e_layers: int = 2,
        top_k_periods: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_scales = num_scales
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 多尺度处理
        self.scale_blocks = nn.ModuleList([
            StackedTPLC(
                d_model=d_model,
                e_layers=e_layers,
                top_k_periods=top_k_periods,
                dropout=dropout,
            )
            for _ in range(num_scales + 1)
        ])
        
        # 尺度融合权重
        self.scale_logits = nn.Parameter(torch.zeros(num_scales + 1))
        
        # 下采样池化
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """前向传播。
        
        Args:
            x: [B, T, input_dim]
        
        Returns:
            多尺度输出列表 [(B, T_0, d_model), (B, T_1, d_model), ...]
        """
        # 输入投影
        x = self.input_proj(x)  # [B, T, d_model]
        
        # 生成多尺度序列
        scales = [x]
        curr = x.transpose(1, 2)  # [B, d_model, T]
        for _ in range(self.num_scales):
            curr = self.pool(curr)
            scales.append(curr.transpose(1, 2))
        
        # 对每个尺度应用堆叠 TPLC
        outputs = []
        for i, (scale_x, block) in enumerate(zip(scales, self.scale_blocks)):
            out = block(scale_x)
            outputs.append(out)
        
        return outputs


class TPLCEncoder(nn.Module):
    """TPLC 编码器：输入嵌入 + 多层堆叠。
    
    完整的编码器，包含：
    1. 输入投影（可选 DataEmbedding）
    2. 多层 TPLC Block + LayerNorm
    3. 输出投影
    
    Args:
        input_dim: 输入特征维度
        d_model: 隐藏维度
        e_layers: 层数
        top_k_periods: 周期数
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        e_layers: int = 2,
        top_k_periods: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 堆叠 TPLC
        self.encoder = StackedTPLC(
            d_model=d_model,
            e_layers=e_layers,
            top_k_periods=top_k_periods,
            dropout=dropout,
        )
        
        # 最终 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [B, T, input_dim]
        
        Returns:
            [B, T, d_model]
        """
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        return x
