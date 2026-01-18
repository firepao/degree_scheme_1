from __future__ import annotations

import torch


@torch.no_grad()
def mae(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(y_true - y_pred))


@torch.no_grad()
def rmse(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))
