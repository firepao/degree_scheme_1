"""Inception Block 多尺度卷积模块。

参考：
- TimesNet (ICLR 2023) - Inception_Block_V1
- GoogLeNet (CVPR 2015) - 原始 Inception

功能：
使用多个不同大小的卷积核并行处理，捕捉不同尺度的局部模式，
最后将结果融合（平均或拼接）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_Block_V1(nn.Module):
    """Inception Block V1：多尺度卷积并行 + 平均融合。
    
    使用 kernel_size = 1, 3, 5, 7, ... (2*i+1) 的多个卷积核并行处理，
    然后在最后一个维度上求平均。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_kernels: 并行卷积核数量（默认 6，即 kernel 1,3,5,7,9,11）
        
    输入: [B, C_in, H, W]
    输出: [B, C_out, H, W]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # 构建多尺度卷积核：kernel_size = 1, 3, 5, 7, ...
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * i + 1,  # 1, 3, 5, 7, ...
                padding=i,              # same padding
                bias=False,
            )
            for i in range(num_kernels)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        # 并行计算所有卷积
        res_list = [kernel(x) for kernel in self.kernels]
        
        # 堆叠后在最后一维求平均
        # [B, C_out, H, W, num_kernels] -> mean -> [B, C_out, H, W]
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        
        return res


class Inception_Block_V2(nn.Module):
    """Inception Block V2：多尺度卷积 + BatchNorm + 激活。
    
    相比 V1 增加了 BatchNorm 和激活函数，更稳定。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_kernels: 并行卷积核数量
        activation: 激活函数类型
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        activation: str = 'gelu',
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        
        # 卷积层
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * i + 1,
                padding=i,
                bias=False,
            )
            for i in range(num_kernels)
        ])
        
        # BatchNorm
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 激活函数
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        res_list = [kernel(x) for kernel in self.kernels]
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        res = self.bn(res)
        res = self.act(res)
        return res


class InceptionConv2d(nn.Module):
    """用于替换 DepthwiseSeparableConv2d 的 Inception 卷积模块。
    
    这是为 TPLC 模型设计的即插即用替换模块。
    
    结构：Inception_Block_V1 -> GELU -> Inception_Block_V1
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_kernels: 并行卷积核数量
        hidden_factor: 中间层通道扩展因子
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        hidden_factor: float = 2.0,
    ) -> None:
        super().__init__()
        hidden_channels = int(in_channels * hidden_factor)
        
        self.conv = nn.Sequential(
            Inception_Block_V1(in_channels, hidden_channels, num_kernels),
            nn.GELU(),
            Inception_Block_V1(hidden_channels, out_channels, num_kernels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AdaptiveInceptionBlock(nn.Module):
    """自适应 Inception Block：根据输入动态选择卷积核。
    
    使用注意力机制对不同尺度的卷积结果进行加权融合，
    而非简单平均。
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        num_kernels: 并行卷积核数量
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
    ) -> None:
        super().__init__()
        self.num_kernels = num_kernels
        
        # 多尺度卷积
        self.kernels = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2 * i + 1,
                padding=i,
                bias=False,
            )
            for i in range(num_kernels)
        ])
        
        # 注意力权重生成器
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, num_kernels),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        B = x.shape[0]
        
        # 计算注意力权重 [B, num_kernels]
        attn_weights = self.attention(x)
        
        # 并行卷积
        res_list = [kernel(x) for kernel in self.kernels]
        
        # 堆叠 [B, C_out, H, W, num_kernels]
        res = torch.stack(res_list, dim=-1)
        
        # 加权融合
        attn_weights = attn_weights.view(B, 1, 1, 1, self.num_kernels)
        res = (res * attn_weights).sum(dim=-1)
        
        return res
