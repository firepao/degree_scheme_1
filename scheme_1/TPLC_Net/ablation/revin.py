"""RevIN (Reversible Instance Normalization) 模块。

参考：
- Non-stationary Transformers (NeurIPS 2022)
- TimesNet (ICLR 2023)

功能：
1. 前向归一化：减均值、除标准差，消除分布偏移
2. 反向归一化：恢复原始分布，保持预测值物理意义
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """可逆实例归一化层。
    
    在时间维度上进行实例归一化，支持可学习的 affine 参数。
    
    Args:
        num_features: 特征维度（通道数）
        eps: 数值稳定性常数
        affine: 是否使用可学习的 scale 和 bias
        subtract_last: 是否用最后一个时间步替代均值（某些场景下更稳定）
    
    用法：
        revin = RevIN(num_features=10)
        
        # 归一化
        x_norm = revin(x, mode='norm')
        
        # ... 模型处理 ...
        
        # 反归一化（注意：target_dim 可能与 num_features 不同）
        y = revin(y_pred, mode='denorm', target_dim=target_dim)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        
        # 用于存储归一化统计量（在 denorm 时使用）
        self.mean: torch.Tensor | None = None
        self.stdev: torch.Tensor | None = None
    
    def forward(
        self,
        x: torch.Tensor,
        mode: str = 'norm',
        target_dim: int | None = None,
    ) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量 [B, T, C]
            mode: 'norm' 归一化 | 'denorm' 反归一化
            target_dim: 反归一化时的目标维度（用于 C_out != C_in 的情况）
        
        Returns:
            处理后的张量
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x, target_dim)
        else:
            raise ValueError(f"mode 必须是 'norm' 或 'denorm'，收到 '{mode}'")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化：减均值、除标准差。"""
        # x: [B, T, C]
        if self.subtract_last:
            # 用最后一个时间步作为参考点
            self.mean = x[:, -1:, :].detach()
        else:
            # 用时间维度的均值
            self.mean = x.mean(dim=1, keepdim=True).detach()
        
        x = x - self.mean
        
        # 计算标准差
        self.stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        
        x = x / self.stdev
        
        # 可学习的 affine 变换
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        
        return x
    
    def _denormalize(
        self,
        x: torch.Tensor,
        target_dim: int | None = None,
    ) -> torch.Tensor:
        """反归一化：恢复原始分布。"""
        if self.mean is None or self.stdev is None:
            raise RuntimeError("必须先调用 normalize 才能调用 denormalize")
        
        # 确定要使用的维度
        if target_dim is None:
            target_dim = x.shape[-1]
        
        # 可学习的 affine 逆变换
        if self.affine:
            x = (x - self.affine_bias[:target_dim]) / (self.affine_weight[:target_dim] + self.eps)
        
        # 获取对应维度的统计量
        # mean/stdev: [B, 1, C_in]
        # 如果 target_dim < C_in，只取前 target_dim 个通道
        mean = self.mean[:, :, :target_dim]
        stdev = self.stdev[:, :, :target_dim]
        
        # 广播到输出的时间维度
        # x: [B, pred_len, target_dim]
        x = x * stdev + mean
        
        return x


class RevINWrapper(nn.Module):
    """RevIN 包装器，方便在模型中使用。
    
    将 RevIN 的归一化和反归一化逻辑封装，简化模型代码。
    
    用法：
        class MyModel(nn.Module):
            def __init__(self, ...):
                self.revin = RevINWrapper(input_dim, target_dim)
            
            def forward(self, x):
                x = self.revin.normalize(x)
                # ... 模型逻辑 ...
                y = self.revin.denormalize(y)
                return y
    """
    
    def __init__(
        self,
        input_dim: int,
        target_dim: int | None = None,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim if target_dim is not None else input_dim
        self.revin = RevIN(num_features=input_dim, eps=eps, affine=affine)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """归一化输入。"""
        return self.revin(x, mode='norm')
    
    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """反归一化输出。"""
        return self.revin(y, mode='denorm', target_dim=self.target_dim)


# ========== 简化版 RevIN（无可学习参数，适合快速实验） ==========

def revin_norm(x: torch.Tensor, eps: float = 1e-5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """函数式 RevIN 归一化。
    
    Args:
        x: [B, T, C] 输入
        eps: 数值稳定性
    
    Returns:
        x_norm: 归一化后的张量
        mean: [B, 1, C] 均值
        stdev: [B, 1, C] 标准差
    """
    mean = x.mean(dim=1, keepdim=True).detach()
    x = x - mean
    stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + eps).detach()
    x_norm = x / stdev
    return x_norm, mean, stdev


def revin_denorm(
    y: torch.Tensor,
    mean: torch.Tensor,
    stdev: torch.Tensor,
    target_dim: int | None = None,
) -> torch.Tensor:
    """函数式 RevIN 反归一化。
    
    Args:
        y: [B, pred_len, target_dim] 模型输出
        mean: [B, 1, C] 归一化时的均值
        stdev: [B, 1, C] 归一化时的标准差
        target_dim: 目标维度
    
    Returns:
        反归一化后的张量
    """
    if target_dim is None:
        target_dim = y.shape[-1]
    
    y = y * stdev[:, :, :target_dim] + mean[:, :, :target_dim]
    return y
