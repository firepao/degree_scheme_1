from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class LSTMConfig:
    """LSTM 基线配置。"""

    input_dim: int
    target_dim: int
    seq_len: int
    pred_len: int
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


class LSTMForecaster(nn.Module):
    """时间序列预测 LSTM 基线。

    输入：x -> [B, seq_len, input_dim]
    输出：y_hat -> [B, pred_len, target_dim]

    说明：
    - 使用 LSTM 编码历史序列；
    - 取最后时刻隐藏状态，通过线性层一次性预测未来 pred_len 步。
    """

    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = int(input_dim)
        self.target_dim = int(target_dim)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

        lstm_dropout = self.dropout if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        proj_in = self.hidden_dim * (2 if self.bidirectional else 1)
        self.proj = nn.Sequential(
            nn.LayerNorm(proj_in),
            nn.Linear(proj_in, self.pred_len * self.target_dim),
        )

    @classmethod
    def from_config(cls, cfg: LSTMConfig) -> "LSTMForecaster":
        return cls(
            input_dim=cfg.input_dim,
            target_dim=cfg.target_dim,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            bidirectional=cfg.bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x 维度应为 [B,T,C]，但得到：{tuple(x.shape)}")

        out, _ = self.lstm(x)  # [B, T, H]
        h_last = out[:, -1, :]  # [B, H]
        y = self.proj(h_last)  # [B, pred_len*target_dim]
        y = y.view(x.size(0), self.pred_len, self.target_dim)
        return y
