"""LSTM 基线模型。

用于与 TPLCNet 做对比实验。
"""

from .lstm import LSTMForecaster

__all__ = [
    "LSTMForecaster",
]
