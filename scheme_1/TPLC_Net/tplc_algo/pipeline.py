from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .data.greenhouse_dataset import GreenhouseMergedDataset, WindowSpec, load_team_merged_dataframe
from .utils.scaler import NumpyStandardScaler


@dataclass
class PreparedData:
    """数据准备结果，用于训练/评估。"""

    feature_cols: list[str]
    target_cols: list[str]
    train_ds: GreenhouseMergedDataset
    val_ds: GreenhouseMergedDataset
    test_ds: GreenhouseMergedDataset
    all_scaler: NumpyStandardScaler
    target_scaler: NumpyStandardScaler


def prepare_greenhouse_datasets(
    dataset_root: Path,
    team: str,
    seq_len: int,
    pred_len: int,
    stride: int = 1,
    base_table: str = "GreenhouseClimate.csv",
    extra_tables: Sequence[str] = (
        "CropParameters.csv",
        "GrodanSens.csv",
        "LabAnalysis.csv",
        "Production.csv",
        "Resources.csv",
        "TomQuality.csv",
    ),
    feature_cols: Sequence[str] | None = None,
    target_cols: Sequence[str] | None = None,
    selected_features: Sequence[str] | None = None,
    split: Tuple[float, float, float] = (0.7, 0.1, 0.2),
    *,
    missing_rate_threshold: float = 0.7,
    drop_constant: bool = True,
    protect_target_cols: bool = True,
) -> PreparedData:
    """读取/对齐数据并生成 train/val/test 的滑窗数据集。

    说明：
    - 默认输入：所有数值列（除 time）
    - 默认输出：优先预测 ['Tair','Rhair','CO2air']，否则取前 3 列
    - selected_features: 如果指定，只保留这些特征（目标变量会自动添加）
    - 标准化：用 feature_cols 的 scaler 统一标准化；target_scaler 从 all_scaler 中切片得到。
    """

    dataset_root = Path(dataset_root)
    team_dir = dataset_root / team

    # 先读取“未填充”的数据，用于统计缺失率做特征筛选；随后再按兼容策略统一填充。
    merged_df_raw = load_team_merged_dataframe(
        team_dir,
        base_table=base_table,
        extra_tables=extra_tables,
        drop_unnamed=True,
        fill_strategy="none",
    )
    all_cols = [c for c in merged_df_raw.columns if c != "time"]

    feature_cols_list = list(feature_cols) if feature_cols is not None else all_cols

    if target_cols is None:
        preferred = ["Tair", "Rhair", "CO2air"]
        target_cols_list = [c for c in preferred if c in merged_df_raw.columns]
        if len(target_cols_list) == 0:
            target_cols_list = all_cols[:3]
    else:
        target_cols_list = list(target_cols)

    # 目标列必须存在；并确保目标列包含在输入特征列中（便于 target_scaler 从 all_scaler 切片）。
    missing_targets = [c for c in target_cols_list if c not in merged_df_raw.columns]
    if missing_targets:
        raise ValueError(f"target_cols 中存在数据里没有的列: {missing_targets}")

    for c in target_cols_list:
        if c not in feature_cols_list:
            feature_cols_list.append(c)

    if not (0.0 <= missing_rate_threshold <= 1.0):
        raise ValueError("missing_rate_threshold 必须在 [0,1] 范围内")

    n = len(merged_df_raw)
    s0, s1, s2 = split
    if abs((s0 + s1 + s2) - 1.0) > 1e-6:
        raise ValueError("split 三段比例之和必须为 1")

    n_train = int(n * s0)
    n_val = int(n * s1)

    train_df_raw = merged_df_raw.iloc[:n_train].reset_index(drop=True)

    # ========= 特征筛选（仅基于 train 统计，避免未来信息进入筛选规则） =========
    # 1) 缺失率过滤：缺失率 > 阈值的列丢弃（默认阈值更保守：0.7）
    missing_rate = train_df_raw[feature_cols_list].isna().mean(axis=0)
    drop_by_missing = set(missing_rate[missing_rate > missing_rate_threshold].index.tolist())
    if protect_target_cols:
        drop_by_missing -= set(target_cols_list)

    kept_feature_cols = [c for c in feature_cols_list if c not in drop_by_missing]
    if len(kept_feature_cols) == 0:
        raise RuntimeError("特征筛选后 feature_cols 为空：请放宽 missing_rate_threshold 或检查数据")

    # 2) 兼容旧行为：对“保留列”在全表范围做 ffill+bfill+0 的缺失填充
    merged_df = merged_df_raw[["time"] + kept_feature_cols].copy()
    merged_df = merged_df.sort_values(by=["time"]).reset_index(drop=True)
    merged_df = merged_df.ffill().bfill().fillna(0.0)

    # 3) 常数列过滤：在填充后的 train 上判断（避免 NaN 影响），但仍仅用 train 统计
    if drop_constant:
        train_df_tmp = merged_df.iloc[:n_train].reset_index(drop=True)
        nunique = train_df_tmp[kept_feature_cols].nunique(axis=0, dropna=False)
        drop_by_const = set(nunique[nunique <= 1].index.tolist())
        if protect_target_cols:
            drop_by_const -= set(target_cols_list)
        kept_feature_cols = [c for c in kept_feature_cols if c not in drop_by_const]
        if len(kept_feature_cols) == 0:
            raise RuntimeError("常数列过滤后 feature_cols 为空：请关闭 drop_constant 或检查数据")

    # 最终切分（基于已填充+已对齐+已筛选的数据）
    train_df = merged_df.iloc[:n_train].reset_index(drop=True)
    val_df = merged_df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = merged_df.iloc[n_train + n_val :].reset_index(drop=True)

    # 训练/评估都使用同一套 feature_cols（筛选后的）
    feature_cols_list = kept_feature_cols

    # 如果指定了 selected_features，则进一步筛选
    if selected_features is not None:
        selected_features_list = list(selected_features)
        # 确保目标变量包含在内
        for tc in target_cols_list:
            if tc not in selected_features_list:
                selected_features_list.append(tc)
        # 只保留存在且在 selected_features 中的特征
        feature_cols_list = [c for c in feature_cols_list if c in selected_features_list]
        if len(feature_cols_list) == 0:
            raise RuntimeError("selected_features 筛选后 feature_cols 为空：请检查特征名称")
        print(f"使用 selected_features: {len(feature_cols_list)} 个特征")

    all_scaler = NumpyStandardScaler().fit(train_df[feature_cols_list].to_numpy(np.float32))
    if all_scaler.mean_ is None or all_scaler.std_ is None:
        raise RuntimeError("标准化器拟合失败：mean_/std_ 为空")

    target_indices = [feature_cols_list.index(c) for c in target_cols_list]
    target_scaler = NumpyStandardScaler(
        mean_=all_scaler.mean_[target_indices],
        std_=all_scaler.std_[target_indices],
    )

    def apply(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[feature_cols_list] = all_scaler.transform(out[feature_cols_list].to_numpy(np.float32))
        return out

    train_df_s = apply(train_df)
    val_df_s = apply(val_df)
    test_df_s = apply(test_df)

    window = WindowSpec(seq_len=seq_len, pred_len=pred_len, stride=stride)
    train_ds = GreenhouseMergedDataset(train_df_s, window, feature_cols=feature_cols_list, target_cols=target_cols_list)
    val_ds = GreenhouseMergedDataset(val_df_s, window, feature_cols=feature_cols_list, target_cols=target_cols_list)
    test_ds = GreenhouseMergedDataset(test_df_s, window, feature_cols=feature_cols_list, target_cols=target_cols_list)

    return PreparedData(
        feature_cols=feature_cols_list,
        target_cols=target_cols_list,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        all_scaler=all_scaler,
        target_scaler=target_scaler,
    )


def make_loaders(
    prepared: PreparedData,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建 DataLoader（三份）。"""

    train_loader = DataLoader(
        prepared.train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        prepared.val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        prepared.test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader
