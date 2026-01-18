from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def _add_repo_paths(scheme_root: Path) -> None:
    """把本方案需要的代码目录加到 sys.path，便于直接运行脚本。"""

    tplc_net_root = scheme_root / "TPLC_Net"
    baselines_root = scheme_root / "baselines"

    sys.path.insert(0, str(tplc_net_root))
    sys.path.insert(0, str(baselines_root))


def _prepare_data(
    team_dir: Path,
    seq_len: int,
    pred_len: int,
    stride: int,
    batch_size: int,
    selected_features: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[str],
    list[str],
    Any,
]:
    from tplc_algo.pipeline import make_loaders, prepare_greenhouse_datasets

    prepared = prepare_greenhouse_datasets(
        dataset_root=team_dir.parent,
        team=team_dir.name,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        selected_features=selected_features,
        target_cols=target_cols,
        # 用户已确认：保持兼容缺失填充；缺失率阈值更保守（0.7）
        missing_rate_threshold=0.7,
        drop_constant=True,
        protect_target_cols=True,
    )

    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=batch_size)
    return (
        train_loader,
        val_loader,
        test_loader,
        prepared.feature_cols,
        prepared.target_cols,
        prepared.target_scaler,
    )


@torch.no_grad()
def _inverse_metrics(
    *,
    model,
    loader: DataLoader,
    device: str,
    target_scaler,
) -> Dict[str, float]:
    """在原始量纲（反标准化后）计算 MAE/RMSE。"""

    model.eval()
    y_true_list = []
    y_pred_list = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        y_true_list.append(y.detach().cpu().numpy())
        y_pred_list.append(y_hat.detach().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)  # [N, pred, D]
    y_pred = np.concatenate(y_pred_list, axis=0)  # [N, pred, D]

    # inverse_transform 期望 [*, D]
    y_true_2d = y_true.reshape(-1, y_true.shape[-1])
    y_pred_2d = y_pred.reshape(-1, y_pred.shape[-1])
    y_true_raw = target_scaler.inverse_transform(y_true_2d).reshape(y_true.shape)
    y_pred_raw = target_scaler.inverse_transform(y_pred_2d).reshape(y_pred.shape)

    mae_raw = float(np.mean(np.abs(y_true_raw - y_pred_raw)))
    rmse_raw = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))
    return {"mae_raw": mae_raw, "rmse_raw": rmse_raw}


def _ensure_run_subdirs(model_dir: Path) -> None:
    (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (model_dir / "figures").mkdir(parents=True, exist_ok=True)
    (model_dir / "artifacts").mkdir(parents=True, exist_ok=True)


def _train_eval_one(
    *,
    model_name: str,
    model,
    run_dir: Path,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_scaler,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> Dict[str, Any]:
    from tplc_algo.exp_utils import save_config_json, save_env_json, save_history_csv, save_metrics_json, save_text
    from tplc_algo.train import TrainConfig, Trainer

    model_dir = run_dir / model_name
    _ensure_run_subdirs(model_dir)

    ckpt_path = model_dir / "checkpoints" / "best.pt"

    trainer = Trainer(
        model=model,
        cfg=TrainConfig(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            ckpt_path=ckpt_path,
            show_progress=True,
            progress_granularity="batch",

            # 稳定性增强（CUDA 上收益明显）
            use_amp=(device == "cuda"),
            grad_clip_max_norm=1.0,
            lr_scheduler="plateau",
            plateau_patience=3,
            plateau_factor=0.5,
            early_stop_patience=6,
        ),
    )

    history = trainer.fit(train_loader, val_loader=val_loader)
    metrics = trainer.evaluate(test_loader)

    # 额外：反标准化后的 MAE/RMSE（更直观）
    if target_scaler is not None:
        metrics.update(_inverse_metrics(model=model, loader=test_loader, device=device, target_scaler=target_scaler))

    save_config_json(
        model_dir,
        {
            "model_name": model_name,
            "train": {"epochs": epochs, "lr": lr, "weight_decay": weight_decay, "device": device},
        },
    )
    save_env_json(model_dir)
    save_history_csv(model_dir, history)
    save_metrics_json(model_dir, metrics)
    save_text(model_dir / "artifacts" / "checkpoint_path.txt", str(ckpt_path))

    return {"history": history, "metrics": metrics, "model_dir": str(model_dir)}


def main() -> None:
    scheme_root = Path(__file__).resolve().parent
    _add_repo_paths(scheme_root)

    parser = argparse.ArgumentParser(description="TPLCNet vs LSTM 对比实验（温室数据）")
    parser.add_argument("--team", type=str, default="AICU", help="队伍目录名，如 AICU/Reference/IUACAAS")
    parser.add_argument("--seq-len", type=int, default=288)
    parser.add_argument("--pred-len", type=int, default=72)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=64, help="TPLCNet hidden_dim")
    parser.add_argument("--top-k-periods", type=int, default=3)
    parser.add_argument("--num-scales", type=int, default=2)
    parser.add_argument("--tune-tplc", action="store_true", help="是否对 TPLCNet 进行随机搜索自动调参")
    parser.add_argument("--tune-trials", type=int, default=10, help="TPLCNet 调参 trial 数")
    parser.add_argument("--tune-epochs", type=int, default=8, help="每个 trial 训练轮数（建议 5-10；越大越准，越慢）")
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    args = parser.parse_args()

    dataset_root = (scheme_root / "datasets" / "自主温室挑战赛").resolve()
    team_dir = dataset_root / args.team

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from tplc_algo.exp_utils import create_run_dir, save_json

    # 单次对比实验总目录（时间戳）
    run_dir = create_run_dir(exp_name=f"compare_tplc_vs_lstm_{args.team}")

    # 读取 Config 中的特征/目标配置 (如果没有则为 None)
    from tplc_algo.config import TPLCConfig
    selected_features = list(TPLCConfig.feature_cols) if TPLCConfig.feature_cols else None
    target_cols_cfg = list(TPLCConfig.target_cols) if TPLCConfig.target_cols else None

    # 如果 config 指定了目标，打印一下提示
    if target_cols_cfg:
        print(f"Config: 仅预测以下目标变量: {target_cols_cfg}")
    if selected_features:
        print(f"Config: 仅使用以下输入特征: {selected_features}")

    train_loader, val_loader, test_loader, feature_cols, target_cols, target_scaler = _prepare_data(
        team_dir=team_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride,
        batch_size=args.batch_size,
        selected_features=selected_features,
        target_cols=target_cols_cfg,
    )

    from tplc_algo.models import TPLCNet
    from LSTM import LSTMForecaster

    # ===== 可选：TPLCNet 自动调参（使用 val_loss 选择 best） =====
    tplc_lr = float(args.lr)
    tplc_weight_decay = float(args.weight_decay)
    tplc_hidden_dim = int(args.hidden_dim)
    tplc_top_k_periods = int(args.top_k_periods)
    tplc_num_scales = int(args.num_scales)
    tplc_tuner_dir = None

    if args.tune_tplc:
        from tplc_algo.tuner import tune_tplcnet_random_search

        best_params, _trial_results, tuner_dir = tune_tplcnet_random_search(
            input_dim=len(feature_cols),
            target_dim=len(target_cols),
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            train_loader=train_loader,
            val_loader=val_loader,
            max_trials=args.tune_trials,
            exp_name=f"tplcnet_tuning_{args.team}",
            device=device,
            fixed_train_cfg={"epochs": args.tune_epochs},
        )
        tplc_tuner_dir = str(tuner_dir)
        tplc_lr = float(best_params.get("lr", tplc_lr))
        tplc_weight_decay = float(best_params.get("weight_decay", tplc_weight_decay))
        tplc_hidden_dim = int(best_params.get("hidden_dim", tplc_hidden_dim))
        tplc_top_k_periods = int(best_params.get("top_k_periods", tplc_top_k_periods))
        tplc_num_scales = int(best_params.get("num_scales", tplc_num_scales))

        print("\n===== TPLCNet 自动调参完成 =====")
        print("tuner_dir =", tuner_dir)
        print(
            "best params ->",
            {
                "hidden_dim": tplc_hidden_dim,
                "num_scales": tplc_num_scales,
                "top_k_periods": tplc_top_k_periods,
                "lr": tplc_lr,
                "weight_decay": tplc_weight_decay,
            },
        )

    tplc_model = TPLCNet(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_scales=tplc_num_scales,
        top_k_periods=tplc_top_k_periods,
        hidden_dim=tplc_hidden_dim,
    )

    lstm_model = LSTMForecaster(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        hidden_dim=args.lstm_hidden,
        num_layers=args.lstm_layers,
        dropout=0.1,
        bidirectional=False,
    )

    print("device =", device)
    print("team_dir =", team_dir)
    print("run_dir =", run_dir)
    print("input_dim =", len(feature_cols), "target_dim =", len(target_cols), target_cols)

    summary: Dict[str, Any] = {
        "team": args.team,
        "dataset_root": str(dataset_root),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": device,
        "tplc_tuning": {
            "enabled": bool(args.tune_tplc),
            "tuner_dir": tplc_tuner_dir,
            "trials": int(args.tune_trials),
            "trial_epochs": int(args.tune_epochs),
        },
        "models": {},
    }

    print("\n===== 训练并评估：TPLCNet =====")
    summary["models"]["tplcnet"] = _train_eval_one(
        model_name="tplcnet",
        model=tplc_model,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device=device,
        epochs=args.epochs,
        lr=tplc_lr,
        weight_decay=tplc_weight_decay,
    )

    print("\n===== 训练并评估：LSTM =====")
    summary["models"]["lstm"] = _train_eval_one(
        model_name="lstm",
        model=lstm_model,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        target_scaler=target_scaler,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ===== 汇总表（方便汇报/Excel） =====
    rows = []
    for name in ["tplcnet", "lstm"]:
        rec = summary["models"].get(name, {})
        metrics = rec.get("metrics", {})
        rows.append(
            {
                "model": name,
                "loss": float(metrics.get("loss", float("nan"))),
                "mae": float(metrics.get("mae", float("nan"))),
                "rmse": float(metrics.get("rmse", float("nan"))),
                "mae_raw": float(metrics.get("mae_raw", float("nan"))),
                "rmse_raw": float(metrics.get("rmse_raw", float("nan"))),
                "model_dir": str(rec.get("model_dir", "")),
            }
        )

    df_summary = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    summary_csv = Path(run_dir) / "artifacts" / "summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\n===== 汇总（test 指标）=====")
    try:
        print(df_summary.to_string(index=False))
    except Exception:
        print(df_summary)

    save_json(Path(run_dir) / "artifacts" / "summary.json", summary)
    print("\n完成：summary 写入", Path(run_dir) / "artifacts" / "summary.json")
    print("完成：summary.csv 写入", summary_csv)


if __name__ == "__main__":
    main()
