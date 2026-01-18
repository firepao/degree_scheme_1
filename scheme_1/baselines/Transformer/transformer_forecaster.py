from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    task_name: str = "long_term_forecast"
    seq_len: int = 288
    pred_len: int = 72
    enc_in: int = 8
    dec_in: int = 1
    c_out: int = 1
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    e_layers: int = 2
    d_layers: int = 1
    factor: int = 3
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"


class TransformerForecaster(nn.Module):
    """封装 Time-Series-Library 的 Vanilla Transformer 预测模型。

    输入：x -> [B, seq_len, input_dim]
    输出：y_hat -> [B, pred_len, target_dim]
    """

    def __init__(self, *, input_dim: int, target_dim: int, seq_len: int, pred_len: int,
                 d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 e_layers: int = 2, d_layers: int = 1, factor: int = 3,
                 dropout: float = 0.1, activation: str = "gelu",
                 embed: str = "timeF", freq: str = "h",
                 tsl_root: Path | None = None) -> None:
        super().__init__()

        # 注入 TSL 路径
        if tsl_root is not None and str(tsl_root) not in sys.path:
            sys.path.insert(0, str(tsl_root))
        else:
            candidates = [
                Path(__file__).parents[3] / "Time_Series_Library" / "Time-Series-Library",
                Path("d:/Time_Series_Library/Time-Series-Library"),
                Path("c:/Users/32698/OneDrive/文档/学位会代码/Time-Series-Library"),
            ]
            for p in candidates:
                if p.exists() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
                    break

        try:
            from models.Transformer import Model as TSLTransformer
        except Exception as e:
            raise ImportError(f"无法导入 Time-Series-Library 的 Transformer 模型，请确认路径与依赖：{e}")

        class Cfg:  # 动态配置对象
            pass

        cfg = Cfg()
        cfg.task_name = "long_term_forecast"
        cfg.seq_len = int(seq_len)
        cfg.pred_len = int(pred_len)
        cfg.enc_in = int(input_dim)
        cfg.dec_in = int(target_dim)
        cfg.c_out = int(target_dim)
        cfg.d_model = int(d_model)
        cfg.n_heads = int(n_heads)
        cfg.d_ff = int(d_ff)
        cfg.e_layers = int(e_layers)
        cfg.d_layers = int(d_layers)
        cfg.factor = int(factor)
        cfg.dropout = float(dropout)
        cfg.activation = str(activation)
        cfg.embed = str(embed)
        cfg.freq = str(freq)

        self.model = TSLTransformer(cfg)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_dim = int(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        # 不使用时间 markup，传 None
        x_mark_enc = None
        # 解码端输入：简单使用最后一个时间步的目标维度（零向量），长度 pred_len
        x_dec = torch.zeros((B, self.pred_len, self.target_dim), device=x.device, dtype=x.dtype)
        x_mark_dec = None
        y = self.model(x, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, target_dim]
        return y
