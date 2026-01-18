from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def _find_time_col(columns: Iterable[str]) -> str:
    candidates: list[str] = []
    for col in columns:
        c = str(col).strip().lower()
        c = c.replace(" ", "")
        if c in {"%time", "time", "%time,"} or c.startswith("%time") or c == "%time":
            candidates.append(col)
        elif c in {"%time", "%time", "%time,"}:
            candidates.append(col)
        elif c.startswith("%time") or c.endswith("time") or "time" in c:
            candidates.append(col)
    if not candidates:
        raise ValueError("未找到时间列（包含 time/%time 的列名）。")
    # 优先选择看起来最标准的 %time / %Time
    candidates_sorted = sorted(candidates, key=lambda x: (str(x).strip().lower().replace(" ", "") != "%time", len(str(x))))
    return candidates_sorted[0]


def _read_table_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    time_col = _find_time_col(df.columns)
    df = df.rename(columns={time_col: "time"})
    # time 既可能是整数天，也可能是带小数的高频天（Excel serial day）
    # merge_asof 要求左右两侧 merge key dtype 完全一致；这里强制统一为 float64
    df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
    df = df.dropna(subset=["time"]).sort_values("time")
    # 去除重复时间点（取最后一条）
    df = df.drop_duplicates(subset=["time"], keep="last")
    return df


def load_team_merged_dataframe(
    team_dir: Path,
    base_table: str = "GreenhouseClimate.csv",
    extra_tables: Sequence[str] = (
        "CropParameters.csv",
        "GrodanSens.csv",
        "LabAnalysis.csv",
        "Production.csv",
        "Resources.csv",
        "TomQuality.csv",
    ),
    *,
    drop_unnamed: bool = True,
    fill_strategy: str = "ffill_bfill_0",
) -> pd.DataFrame:
    """读取并对齐某个队伍目录下的多表数据。

        以 base_table 的高频 time 作为主轴，对其他表做 merge_asof 对齐（向后填充）。

        参数
        - drop_unnamed: 是否删除形如 "Unnamed: 0" 的无意义列。
        - fill_strategy:
            - "ffill_bfill_0"（默认，兼容旧行为）：全表按 time 排序后 ffill + bfill，剩余 NaN 置 0。
            - "none"：不做缺失填充（保留 NaN），供上层 pipeline 进行统计/筛选。
    """

    team_dir = Path(team_dir)
    base_path = team_dir / base_table
    if not base_path.exists():
        raise FileNotFoundError(f"未找到 base_table: {base_path}")

    base_df = _read_table_csv(base_path)
    merged = base_df

    # 保险：统一主表 time dtype，避免后续 merge_asof 报 dtype 不一致
    merged["time"] = pd.to_numeric(merged["time"], errors="coerce").astype("float64")

    for table in extra_tables:
        table_path = team_dir / table
        if not table_path.exists():
            continue
        df = _read_table_csv(table_path)

        # 保险：merge_asof 要求 merge key dtype 完全一致
        df["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
        # 避免列名冲突：同名列加前缀
        overlap = set(merged.columns).intersection(df.columns) - {"time"}
        if overlap:
            df = df.rename(columns={c: f"{table_path.stem}__{c}" for c in overlap})

        merged["time"] = pd.to_numeric(merged["time"], errors="coerce").astype("float64")
        merged = pd.merge_asof(
            merged.sort_values("time"),
            df.sort_values("time"),
            on="time",
            direction="backward",
            allow_exact_matches=True,
        )

    if drop_unnamed:
        merged = merged.drop(columns=[c for c in merged.columns if str(c).strip().lower().startswith("unnamed")], errors="ignore")

    # 保留数值列（time + 数值特征）
    numeric_cols = ["time"]
    for c in merged.columns:
        if c == "time":
            continue
        if pd.api.types.is_numeric_dtype(merged[c]):
            numeric_cols.append(c)
        else:
            # 尝试转为数值
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
            numeric_cols.append(c)
    merged = merged[numeric_cols]

    # 缺失值处理（默认保持兼容旧行为）
    merged = merged.sort_values("time").reset_index(drop=True)
    if fill_strategy not in {"ffill_bfill_0", "none"}:
        raise ValueError("fill_strategy 仅支持 'ffill_bfill_0' 或 'none'")
    if fill_strategy == "ffill_bfill_0":
        merged = merged.ffill().bfill().fillna(0.0)
    return merged


@dataclass
class WindowSpec:
    seq_len: int
    pred_len: int
    stride: int = 1


class GreenhouseMergedDataset(Dataset):
    """温室多表对齐后的滑窗数据集。

    - x: [seq_len, input_dim]
    - y: [pred_len, target_dim]
    """

    def __init__(
        self,
        merged_df: pd.DataFrame,
        window: WindowSpec,
        feature_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        return_time: bool = False,
    ) -> None:
        super().__init__()
        if "time" not in merged_df.columns:
            raise ValueError("merged_df 必须包含 'time' 列")

        self.merged_df = merged_df.copy()
        self.window = window
        self.return_time = return_time

        all_cols = [c for c in merged_df.columns if c != "time"]
        self.feature_cols = list(feature_cols) if feature_cols is not None else all_cols
        self.target_cols = list(target_cols) if target_cols is not None else all_cols

        self._x = merged_df[self.feature_cols].to_numpy(dtype=np.float32)
        self._y = merged_df[self.target_cols].to_numpy(dtype=np.float32)
        self._t = merged_df["time"].to_numpy(dtype=np.float64)

        total_len = len(merged_df)
        self._max_start = total_len - (window.seq_len + window.pred_len)
        if self._max_start < 0:
            raise ValueError(
                f"数据长度不足：len={total_len}, seq_len={window.seq_len}, pred_len={window.pred_len}"
            )

    def __len__(self) -> int:
        return (self._max_start // self.window.stride) + 1

    def __getitem__(self, idx: int):
        start = idx * self.window.stride
        s0 = start
        s1 = start + self.window.seq_len
        s2 = s1 + self.window.pred_len
        x = self._x[s0:s1]
        y = self._y[s1:s2]
        x_t = torch.from_numpy(x)
        y_t = torch.from_numpy(y)

        if not self.return_time:
            return x_t, y_t

        t_in = torch.from_numpy(self._t[s0:s1])
        t_out = torch.from_numpy(self._t[s1:s2])
        return x_t, y_t, t_in, t_out
