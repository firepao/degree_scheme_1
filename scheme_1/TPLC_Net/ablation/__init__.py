"""TPLC 消融实验模块。

本包包含 TPLC 模型的各个可插拔改进模块，用于消融实验。

=== 新版消融模块（推荐）===
- components: 核心组件（FFT 周期提取、1D→2D 重塑、深度可分离卷积等）
- ablation_model: 消融模型类（支持对各组件独立开关）
- run_ablation_new: 新版消融实验运行脚本

=== 旧版消融模块 ===
- revin: RevIN 可逆实例归一化
- inception: Inception Block 多尺度卷积
- decomposition: 季节-趋势分解
- stacked_block: 多层堆叠 TPLC Block
"""

# 新版消融模块
from .components import (
    extract_topk_periods,
    PeriodInfo,
    reshape_1d_to_2d,
    reshape_2d_to_1d,
    DepthwiseSeparableConv2d,
    StandardConv2d,
    MultiScaleGenerator,
    RevIN,
    InceptionBlock,
    FrequencyEnhancement,
    MovingAvgDecomp,
    DFTDecomp,
    AdaptiveScaleFusion,
    MLPPredictor,
    Conv1DPredictor,
)

from .ablation_model import (
    TPLCNet_Ablation,
    AblationConfig,
    create_ablation_model,
)

# 旧版消融模块（保持兼容）
from .revin import RevIN as RevIN_Legacy
from .inception import Inception_Block_V1
from .decomposition import DFT_series_decomp, series_decomp
from .stacked_block import TPLCBlock

__all__ = [
    # 新版核心组件
    "extract_topk_periods",
    "PeriodInfo",
    "reshape_1d_to_2d",
    "reshape_2d_to_1d",
    "DepthwiseSeparableConv2d",
    "StandardConv2d",
    "MultiScaleGenerator",
    "RevIN",
    "InceptionBlock",
    "FrequencyEnhancement",
    "MovingAvgDecomp",
    "DFTDecomp",
    "AdaptiveScaleFusion",
    "MLPPredictor",
    "Conv1DPredictor",
    # 新版消融模型
    "TPLCNet_Ablation",
    "AblationConfig",
    "create_ablation_model",
    # 旧版模块
    "RevIN_Legacy",
    "Inception_Block_V1",
    "DFT_series_decomp",
    "series_decomp",
    "TPLCBlock",
]
