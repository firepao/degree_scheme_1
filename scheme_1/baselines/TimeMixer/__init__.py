"""TimeMixer 基线模型。

基于 Time-Series-Library 的 TimeMixer 实现。
"""

from .timemixer import TimeMixerForecaster, TimeMixerConfig

__all__ = ['TimeMixerForecaster', 'TimeMixerConfig']
