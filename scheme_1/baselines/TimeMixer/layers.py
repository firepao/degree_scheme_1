"""TimeMixer 所需的辅助层。

包含：
- DataEmbedding_wo_pos: 数据嵌入（无位置编码）
- Normalize: 归一化层（RevIN 风格）
- series_decomp: 序列分解（移动平均）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Embedding 层 ==========

class PositionalEmbedding(nn.Module):
    """位置编码。"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        # 计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * 
                    -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """Token 嵌入（1D 卷积）。"""
    
    def __init__(self, c_in: int, d_model: int):
        super().__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            bias=False
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu'
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T] -> Conv -> [B, d_model, T] -> [B, T, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class TemporalEmbedding(nn.Module):
    """时间特征嵌入（固定或可学习）。"""
    
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        super().__init__()
        
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = nn.Embedding  # 简化版：使用可学习嵌入
        
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x


class DataEmbedding_wo_pos(nn.Module):
    """数据嵌入（不含位置编码）。
    
    TimeMixer 使用该嵌入方式：值嵌入 + 时间嵌入。
    """
    
    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed_type: str = 'fixed',
        freq: str = 'h',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(
            d_model=d_model,
            embed_type=embed_type,
            freq=freq
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] 输入序列
            x_mark: [B, T, 4-5] 时间特征（可选）
        
        Returns:
            [B, T, d_model] 嵌入后的表示
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


# ========== 归一化层 ==========

class Normalize(nn.Module):
    """归一化层（RevIN 风格）。
    
    支持：
    - norm: 标准化（均值为 0，方差为 1）
    - denorm: 反标准化
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        subtract_last: bool = False,
        non_norm: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.non_norm:
            return x
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# ========== 序列分解层 ==========

class moving_avg(nn.Module):
    """移动平均（用于提取趋势）。"""
    
    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 在两端填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """序列分解：季节性 + 趋势性。"""
    
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] 输入序列
        
        Returns:
            season: [B, T, C] 季节性成分
            trend: [B, T, C] 趋势性成分
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean
