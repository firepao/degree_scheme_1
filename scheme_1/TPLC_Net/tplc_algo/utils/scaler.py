from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyStandardScaler:
    """最小版标准化器：x'=(x-mean)/std。"""

    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_ = np.where(self.std_ < 1e-6, 1.0, self.std_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("请先 fit 再 transform")
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("请先 fit 再 inverse_transform")
        return x * self.std_ + self.mean_
