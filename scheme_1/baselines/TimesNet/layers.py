"""TimesNet 所需的层实现。"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """位置编码。"""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        # 计算位置编码
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """特征值 Embedding：1D 卷积投影。"""

    def __init__(self, c_in: int, d_model: int) -> None:
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
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C] -> [B, C, T] -> Conv1d -> [B, d_model, T] -> [B, T, d_model]
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """数据 Embedding：特征值 + 位置编码。

    对于温室数据，我们不使用时间标记（x_mark），只用值和位置编码。
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    """Inception 卷积块：多尺度 2D 卷积。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=2 * i + 1,
                padding=i
            ))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
