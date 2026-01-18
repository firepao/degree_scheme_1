from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class TPLCConfig:
    """TPLC 模型与训练默认配置。

    注意：该配置偏向“能跑通”的最小默认值；你可以在 Notebook 中按数据/任务调整。
    """

    # 数据
    dataset_root: Path = Path("../datasets/自主温室挑战赛")
    team: str = "AICU"  # 例如：AICU / Reference / IUACAAS ...
    base_table: str = "GreenhouseClimate.csv"  # 以高频表为时间轴
    extra_tables: Sequence[str] = (
        "CropParameters.csv",
        "GrodanSens.csv",
        "LabAnalysis.csv",
        "Production.csv",
        "Resources.csv",
        "TomQuality.csv",
    )

    # 预测任务
    # 修改为只预测 温度、湿度、CO2
    target_cols: Sequence[str] | None = ("Tair", "Rhair", "CO2air")

    # 特征选择: 显式定义输入特征，包含 自回归项 + 控制量 + 外部扰动
    # 如果为 None，则默认使用所有非 setpoint/vip 列
    feature_cols: Sequence[str] | None = (
        # 目标变量自身 (Auto-regressive)
        "Tair", "Rhair", "CO2air",
        # 主要控制变量 (Actuators)
        "VentLee", "Ventwind",   # 通风
        "PipeGrow", "PipeLow",   # 加热
        "co2_dos",               # CO2 施肥
        "AssimLight",            # 补光
        "EnScr", "BlackScr",     # 幕布 (遮阳/保温)
        # 主要外部扰动 (Disturbance)
        "Tot_PAR",               # 光照辐射 (GreenhouseClimate 可能不含 Tout/Rhout，Par 是最强扰动)
    )

    # 序列切片
    seq_len: int = 288  # 约等于 1 天（5min 频率：288 点/天）
    pred_len: int = 72  # 约等于 6 小时（5min 频率：72 点）
    stride: int = 1

    # 模型结构
    num_scales: int = 2  # M：额外尺度数（总尺度数= M+1）
    top_k_periods: int = 3  # K：每个尺度选取的周期数量
    hidden_dim: int = 64  # 深度可分离卷积输出通道数（C_out）
    dw_kernel: int = 3
    pw_kernel: int = 1

    # 训练
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 10
    seed: int = 42

    def team_dir(self) -> Path:
        return Path(self.dataset_root) / self.team
