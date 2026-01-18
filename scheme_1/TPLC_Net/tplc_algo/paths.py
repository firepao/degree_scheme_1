from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """返回 TPLC_Net 目录（本包所在目录的上一级）。"""

    return Path(__file__).resolve().parents[1]


RESULTS_DIR = project_root() / "results"
TUNER_LOGS_DIR = project_root() / "tuner_logs"
