"""TPLC_Net：基于时序周期与轻量卷积网络的温室环境变量预测模型。

该包实现了方案文档 scheme_1_algo.md 中描述的核心模块：
- 多尺度序列生成（平均池化下采样）
- FFT 周期识别
- 1D→2D 重塑
- 2D 深度可分离卷积
- 多周期加权融合（Softmax(振幅)）
- 多尺度预测头融合
"""

from .config import TPLCConfig
from .models.tplc_model import TPLCNet
from .pipeline import PreparedData, make_loaders, prepare_greenhouse_datasets
from .exp_utils import create_run_dir

__all__ = [
	"TPLCConfig",
	"TPLCNet",
	"PreparedData",
	"prepare_greenhouse_datasets",
	"make_loaders",
	"create_run_dir",
]
