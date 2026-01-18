from __future__ import annotations

import json
import platform
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from .paths import RESULTS_DIR


def now_timestamp() -> str:
    """生成时间戳字符串：YYYYMMDD_HHMMSS。"""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def create_run_dir(exp_name: str, base_dir: Path | None = None) -> Path:
    """创建带时间戳的实验目录。

    目录结构示例：
    results/<exp_name>_<timestamp>/
        checkpoints/
        figures/
        artifacts/
    """

    base_dir = Path(base_dir) if base_dir is not None else RESULTS_DIR
    run_dir = base_dir / f"{exp_name}_{now_timestamp()}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return run_dir


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return obj


def save_json(path: Path, data: Any, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=_to_jsonable)


def save_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def collect_env_info() -> Dict[str, Any]:
    """收集运行环境信息，便于复现。"""

    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["cuda_device"] = None
        info["cuda_version"] = torch.version.cuda
    return info


def save_history_csv(run_dir: Path, history: Mapping[str, list[float]]) -> Path:
    """把 Trainer.fit 返回的 history 保存为 CSV。"""

    run_dir = Path(run_dir)
    keys = list(history.keys())
    length = max((len(history[k]) for k in keys), default=0)
    csv_path = run_dir / "artifacts" / "history.csv"
    lines = [",".join(["epoch"] + keys)]
    for i in range(length):
        row = [str(i + 1)]
        for k in keys:
            v = history.get(k, [])
            row.append("" if i >= len(v) else str(v[i]))
        lines.append(",".join(row))
    save_text(csv_path, "\n".join(lines) + "\n")
    return csv_path


def save_metrics_json(run_dir: Path, metrics: Mapping[str, Any]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "artifacts" / "metrics.json"
    save_json(path, dict(metrics))
    return path


def save_config_json(run_dir: Path, config: Mapping[str, Any]) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "artifacts" / "config.json"
    save_json(path, dict(config))
    return path


def save_env_json(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    path = run_dir / "artifacts" / "env.json"
    save_json(path, collect_env_info())
    return path


def save_figure(fig, path: Path) -> None:
    """保存 matplotlib figure。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
