"""Transformer 基线适配。

封装 Time-Series-Library 中的 Vanilla Transformer 模型，使其满足统一预测接口。
"""

from .transformer_forecaster import TransformerForecaster, TransformerConfig

__all__ = ["TransformerForecaster", "TransformerConfig"]
