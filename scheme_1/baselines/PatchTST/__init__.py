"""PatchTST 基线适配。

封装 Time-Series-Library 中的 PatchTST 模型，使其满足本方案统一的数据管线与预测接口：
- 输入：x -> [B, seq_len, input_dim]
- 输出：y_hat -> [B, pred_len, target_dim]

依赖：Time-Series-Library。
如果库不在本工作区，请在运行脚本中加入路径注入或将其放到本仓库子目录。
"""

from .patchtst_forecaster import PatchTSTForecaster, PatchTSTConfig

__all__ = ["PatchTSTForecaster", "PatchTSTConfig"]
