"""TimesNet 时间序列预测模型。

基于论文：TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
项目地址：https://github.com/thuml/Time-Series-Library
"""

from .timesnet import TimesNetForecaster, TimesNetConfig

__all__ = ['TimesNetForecaster', 'TimesNetConfig']
