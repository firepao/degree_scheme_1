from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import mae, rmse


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: Path | None = None
    show_progress: bool = True
    progress_granularity: str = "batch"  # "batch" | "epoch" | "none"

    # 训练稳定性增强（默认不启用，避免影响现有实验）
    grad_clip_max_norm: float = 0.0
    use_amp: bool = False

    # 学习率调度：none | plateau | cosine
    lr_scheduler: str = "none"
    min_lr: float = 0.0
    plateau_factor: float = 0.5
    plateau_patience: int = 3

    # 早停（基于 val loss），val_loader=None 时不生效
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0


def _maybe_tqdm():
    try:
        # 说明：不要用 tqdm.auto（在未安装 ipywidgets 时会报 IProgress 警告）；
        # 这里用最稳的纯文本进度条，Notebook/终端都可用。
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _tqdm_kwargs() -> Dict[str, Any]:
    # Windows 终端常见编码/字体问题：用纯 ASCII 进度条更稳定
    return {"ascii": os.name == "nt"}


class Trainer:
    """最小可用训练器（MSE 损失 + MAE/RMSE 评估）。"""

    def __init__(self, model: nn.Module, cfg: TrainConfig) -> None:
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # AMP 仅在 CUDA 上启用更合理
        self._use_amp = bool(cfg.use_amp and str(cfg.device).startswith("cuda") and torch.cuda.is_available())
        # torch.cuda.amp.GradScaler 已弃用：统一使用 torch.amp.* 新接口
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)

        self._scheduler = None
        if cfg.lr_scheduler == "plateau":
            self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optim,
                mode="min",
                factor=float(cfg.plateau_factor),
                patience=int(cfg.plateau_patience),
                min_lr=float(cfg.min_lr),
            )
        elif cfg.lr_scheduler == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim,
                T_max=int(cfg.epochs),
                eta_min=float(cfg.min_lr),
            )

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, list[float]]:
        history: Dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_mae": [], "val_rmse": []}

        best_val = None
        best_epoch = None
        bad_epochs = 0
        tqdm_fn = _maybe_tqdm() if self.cfg.show_progress else None
        epoch_pbar: Any | None = None

        epoch_iter: Iterable[int] = range(1, self.cfg.epochs + 1)
        if tqdm_fn is not None and self.cfg.progress_granularity != "none":
            epoch_pbar = tqdm_fn(
                epoch_iter,
                total=self.cfg.epochs,
                desc="train (epoch)",
                unit="epoch",
                **_tqdm_kwargs(),
            )
            epoch_iter = epoch_pbar

        for epoch in epoch_iter:
            self.model.train()
            total_loss = 0.0
            n = 0

            batch_pbar: Any | None = None
            batch_iter: Iterable = train_loader
            if tqdm_fn is not None and self.cfg.progress_granularity == "batch" and hasattr(train_loader, "__len__"):
                batch_pbar = tqdm_fn(
                    train_loader,
                    total=len(train_loader),
                    desc=f"epoch {epoch}/{self.cfg.epochs}",
                    unit="batch",
                    leave=False,
                    **_tqdm_kwargs(),
                )
                batch_iter = batch_pbar

            for x, y in batch_iter:
                x = x.to(self.cfg.device)
                y = y.to(self.cfg.device)
                self.optim.zero_grad(set_to_none=True)

                if self._use_amp:
                    with torch.amp.autocast("cuda"):
                        y_hat = self.model(x)
                        loss = self.loss_fn(y_hat, y)
                    self._scaler.scale(loss).backward()
                    if float(self.cfg.grad_clip_max_norm) > 0:
                        self._scaler.unscale_(self.optim)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(self.cfg.grad_clip_max_norm),
                        )
                    self._scaler.step(self.optim)
                    self._scaler.update()
                else:
                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)
                    loss.backward()
                    if float(self.cfg.grad_clip_max_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=float(self.cfg.grad_clip_max_norm),
                        )
                    self.optim.step()
                total_loss += float(loss.item()) * x.size(0)
                n += x.size(0)

                if batch_pbar is not None:
                    avg_loss = total_loss / max(1, n)
                    try:
                        batch_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
                    except Exception:
                        pass

            train_loss = total_loss / max(1, n)
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_mae"].append(val_metrics["mae"])
                history["val_rmse"].append(val_metrics["rmse"])

                # scheduler
                if self._scheduler is not None:
                    if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self._scheduler.step(val_metrics["loss"])
                    else:
                        self._scheduler.step()

                if epoch_pbar is not None:
                    try:
                        epoch_pbar.set_postfix({
                            "train": f"{train_loss:.4f}",
                            "val": f"{val_metrics['loss']:.4f}",
                        })
                    except Exception:
                        pass

                if best_val is None or val_metrics["loss"] < best_val:
                    improved = True
                else:
                    improved = (val_metrics["loss"] < (best_val - float(self.cfg.early_stop_min_delta))) if best_val is not None else True

                if best_val is None or improved:
                    best_val = val_metrics["loss"]
                    best_epoch = int(epoch)
                    bad_epochs = 0
                    if self.cfg.ckpt_path is not None:
                        self.save(self.cfg.ckpt_path)
                else:
                    bad_epochs += 1

                # early stopping
                if int(self.cfg.early_stop_patience) > 0 and bad_epochs >= int(self.cfg.early_stop_patience):
                    if epoch_pbar is not None:
                        try:
                            epoch_pbar.set_postfix({
                                "train": f"{train_loss:.4f}",
                                "val": f"{val_metrics['loss']:.4f}",
                                "stop": f"best@{best_epoch}",
                            })
                        except Exception:
                            pass
                    break
            else:
                if epoch_pbar is not None:
                    try:
                        epoch_pbar.set_postfix({"train": f"{train_loss:.4f}"})
                    except Exception:
                        pass

            # cosine scheduler（无 val 也可以走）
            if val_loader is None and self._scheduler is not None and not isinstance(
                self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self._scheduler.step()

        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        total_rmse = 0.0
        n = 0

        tqdm_fn = _maybe_tqdm() if self.cfg.show_progress else None
        eval_pbar: Any | None = None
        it: Iterable = loader
        if tqdm_fn is not None and self.cfg.progress_granularity == "batch" and hasattr(loader, "__len__"):
            eval_pbar = tqdm_fn(
                loader,
                total=len(loader),
                desc="eval",
                unit="batch",
                leave=False,
                **_tqdm_kwargs(),
            )
            it = eval_pbar

        for x, y in it:
            x = x.to(self.cfg.device)
            y = y.to(self.cfg.device)
            if self._use_amp:
                with torch.amp.autocast("cuda"):
                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)
            else:
                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)
            total_loss += float(loss.item()) * x.size(0)
            total_mae += float(mae(y, y_hat).item()) * x.size(0)
            total_rmse += float(rmse(y, y_hat).item()) * x.size(0)
            n += x.size(0)

        n = max(1, n)
        return {
            "loss": total_loss / n,
            "mae": total_mae / n,
            "rmse": total_rmse / n,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)

    def load(self, path: Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.cfg.device)
        self.model.load_state_dict(ckpt["state_dict"])
