from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import torch
import torch.nn as nn


@dataclass
class PatchTSTConfig:
    """PatchTST 配置（与 Time-Series-Library 模型保持一致字段）。

    仅保留预测所需字段，其他按默认值设定。
    """

    task_name: str = "long_term_forecast"
    seq_len: int = 288
    pred_len: int = 72
    enc_in: int = 8
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    e_layers: int = 2
    factor: int = 3
    dropout: float = 0.1
    activation: str = "gelu"

    # patch 配置
    patch_len: int = 16
    stride: int = 8


class PatchTSTForecaster(nn.Module):
    """封装 Time-Series-Library 的 PatchTST 预测模型。

    输入：x -> [B, seq_len, input_dim]
    输出：y_hat -> [B, pred_len, target_dim]

    注意：TSL 原始 PatchTST 的输出维度是 enc_in（每个变量都预测）。
    如需仅预测部分目标维度（target_dim < enc_in），此处按前 target_dim 截取。
    """

    def __init__(self, *, input_dim: int, target_dim: int, seq_len: int, pred_len: int,
                 d_model: int = 64, n_heads: int = 4, d_ff: int = 128,
                 e_layers: int = 2, factor: int = 3, dropout: float = 0.1,
                 activation: str = "gelu", patch_len: int = 16, stride: int = 8,
                 tsl_root: Path | None = None) -> None:
        super().__init__()

        # 注入 TSL 路径（优先使用传入路径，其次尝试工作区常见位置）
        if tsl_root is not None and str(tsl_root) not in sys.path:
            sys.path.insert(0, str(tsl_root))
        else:
            # 兼容：尝试项目根之外的常见位置
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
            from models.PatchTST import Model as TSLPatchTST
        except Exception as e:
            raise ImportError(f"无法导入 Time-Series-Library 的 PatchTST 模型，请确认路径与依赖：{e}")

        # 构造与 TSL 兼容的配置对象（使用简单的动态对象）
        class Cfg:
            pass

        cfg = Cfg()
        cfg.task_name = "long_term_forecast"
        cfg.seq_len = int(seq_len)
        cfg.pred_len = int(pred_len)
        cfg.enc_in = int(input_dim)
        cfg.d_model = int(d_model)
        cfg.n_heads = int(n_heads)
        cfg.d_ff = int(d_ff)
        cfg.e_layers = int(e_layers)
        cfg.factor = int(factor)
        cfg.dropout = float(dropout)
        cfg.activation = str(activation)

        self.model = TSLPatchTST(cfg, patch_len=patch_len, stride=stride)
        self.target_dim = int(target_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TSL PatchTST forward 需要 x_mark/x_dec/x_mark_dec，但预测路径可传 None
        y = self.model(x, None, None, None)  # [B, pred_len, enc_in]
        if y.size(-1) == self.target_dim:
            return y
        # 截取前 target_dim 作为预测输出
        return y[..., : self.target_dim]
