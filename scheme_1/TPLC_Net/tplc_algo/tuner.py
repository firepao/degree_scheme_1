from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .exp_utils import now_timestamp, save_json
from .paths import TUNER_LOGS_DIR
from .train.trainer import Trainer, TrainConfig
from .utils.seed import seed_everything


def _maybe_tqdm():
    try:
        from tqdm.auto import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


def _tqdm_kwargs() -> Dict[str, Any]:
    # Windows 终端常见编码/字体问题：用纯 ASCII 进度条更稳定
    return {"ascii": os.name == "nt"}


@dataclass
class TrialResult:
    trial_id: int
    params: Dict[str, Any]
    val_loss: float


def _sample_value(rng: random.Random, spec: Any) -> Any:
    """从 spec 里采样一个值。

    支持：
    - list/tuple 的离散候选：从中随机选
    - (low, high, "int") 或 (low, high, "float")：连续/整数范围
    """

    if isinstance(spec, (list, tuple)) and len(spec) >= 1:
        # 连续范围约定：三元组 (low, high, type)
        if len(spec) == 3 and isinstance(spec[2], str) and spec[2] in {"int", "float"}:
            low, high, kind = spec
            if kind == "int":
                return rng.randint(int(low), int(high))
            return rng.uniform(float(low), float(high))
        return rng.choice(list(spec))
    return spec


def random_search(
    build_model_fn,
    train_loader: DataLoader,
    val_loader: DataLoader,
    search_space: Mapping[str, Any],
    max_trials: int,
    base_seed: int = 42,
    exp_name: str = "tplc_tuning",
    device: str | None = None,
    fixed_train_cfg: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[TrialResult], Path]:
    """最小可用随机搜索调参。

    - build_model_fn(params) -> torch.nn.Module
    - 用 val_loss 作为目标（越小越好）
    - 日志写入：tuner_logs/<exp_name>_<timestamp>/trials.jsonl + best.json
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fixed_train_cfg = dict(fixed_train_cfg or {})

    run_dir = TUNER_LOGS_DIR / f"{exp_name}_{now_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    trials_path = run_dir / "trials.jsonl"

    rng = random.Random(base_seed)
    results: List[TrialResult] = []
    best_params: Dict[str, Any] | None = None
    best_loss: float | None = None

    # 写入 meta
    save_json(run_dir / "meta.json", {
        "exp_name": exp_name,
        "max_trials": max_trials,
        "base_seed": base_seed,
        "device": device,
        "search_space": search_space,
        "fixed_train_cfg": fixed_train_cfg,
    })

    with trials_path.open("w", encoding="utf-8") as f:
        tqdm = _maybe_tqdm()
        trial_iter = range(max_trials)
        if tqdm is not None:
            trial_iter = tqdm(
                trial_iter,
                total=max_trials,
                desc="tuning (trial)",
                unit="trial",
                **_tqdm_kwargs(),
            )

        for trial_id in trial_iter:
            params = {k: _sample_value(rng, v) for k, v in search_space.items()}

            # 每个 trial 设定不同种子，保证可复现
            seed = int(base_seed + trial_id)
            seed_everything(seed)

            model = build_model_fn(params)
            train_cfg = TrainConfig(
                epochs=int(fixed_train_cfg.get("epochs", 5)),
                lr=float(params.get("lr", fixed_train_cfg.get("lr", 1e-3))),
                weight_decay=float(params.get("weight_decay", fixed_train_cfg.get("weight_decay", 0.0))),
                device=device,
                ckpt_path=None,
                show_progress=True,
                progress_granularity="epoch",
            )
            trainer = Trainer(model=model, cfg=train_cfg)
            trainer.fit(train_loader, val_loader=val_loader)
            val_metrics = trainer.evaluate(val_loader)
            val_loss = float(val_metrics["loss"])

            rec = {
                "trial_id": trial_id,
                "seed": seed,
                "params": params,
                "val": val_metrics,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            results.append(TrialResult(trial_id=trial_id, params=params, val_loss=val_loss))
            if best_loss is None or val_loss < best_loss:
                best_loss = val_loss
                best_params = params

            if tqdm is not None:
                try:
                    trial_iter.set_postfix({"best": f"{(best_loss if best_loss is not None else float('nan')):.4f}", "last": f"{val_loss:.4f}"})
                except Exception:
                    pass

    if best_params is None:
        best_params = {}
    save_json(run_dir / "best.json", {"best_params": best_params, "best_val_loss": best_loss})
    return best_params, results, run_dir


def tune_tplcnet_random_search(
    *,
    input_dim: int,
    target_dim: int,
    seq_len: int,
    pred_len: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_trials: int = 10,
    exp_name: str = "tplcnet_tuning",
    device: str | None = None,
    base_seed: int = 42,
    fixed_train_cfg: Mapping[str, Any] | None = None,
    search_space: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[TrialResult], Path]:
    """TPLCNet 的最小可用自动调参（随机搜索）。

    默认搜索空间（可通过 search_space 覆盖/扩展）：
    - hidden_dim: [32, 64, 96, 128]
    - num_scales: [1, 2, 3]
    - top_k_periods: [2, 3, 4, 5]
    - lr: (1e-4, 3e-3, 'float')
    - weight_decay: (0.0, 1e-2, 'float')
    """

    from .models import TPLCNet

    default_space: Dict[str, Any] = {
        "hidden_dim": [32, 64, 96, 128],
        "num_scales": [1, 2, 3],
        "top_k_periods": [2, 3, 4, 5],
        "lr": (1e-4, 3e-3, "float"),
        "weight_decay": (0.0, 1e-2, "float"),
    }

    merged_space = dict(default_space)
    if search_space is not None:
        merged_space.update(dict(search_space))

    def build_model_fn(params: Mapping[str, Any]):
        return TPLCNet(
            input_dim=int(input_dim),
            target_dim=int(target_dim),
            seq_len=int(seq_len),
            pred_len=int(pred_len),
            num_scales=int(params.get("num_scales", 2)),
            top_k_periods=int(params.get("top_k_periods", 3)),
            hidden_dim=int(params.get("hidden_dim", 64)),
        )

    return random_search(
        build_model_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        search_space=merged_space,
        max_trials=int(max_trials),
        base_seed=int(base_seed),
        exp_name=str(exp_name),
        device=device,
        fixed_train_cfg=fixed_train_cfg,
    )
