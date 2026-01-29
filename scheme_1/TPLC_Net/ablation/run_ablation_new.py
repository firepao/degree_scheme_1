"""TPLC 模型核心组件消融实验主脚本。

本脚本针对 TPLC 模型的核心创新组件进行消融实验：
1. FFT 周期提取 (use_fft): 动态发现数据周期 vs 固定周期
2. 1D→2D 重塑 (use_reshape_2d): 将时序转为 2D 图像处理 vs 纯 1D
3. 深度可分离卷积 (use_depthwise): 高效参数化 vs 标准卷积
4. 多尺度处理 (use_multi_scale): 多粒度特征 vs 单尺度
5. 多周期融合 (use_multi_period): 多周期组合 vs 单周期
6. 振幅加权融合 (use_amp_weight): 使用 FFT 振幅作为周期权重 vs 非振幅权重
7. 多尺度融合 (use_scale_weight): 等权求和 vs 可学习权重
8. 残差连接 (use_residual): 梯度流优化 vs 无残差
9. RevIN 归一化 (use_revin): 分布对齐 vs 不使用

使用方式:
    # 运行单个消融
    python run_ablation_new.py --config full           # 完整模型（基准）
    python run_ablation_new.py --config no_fft         # 不使用 FFT
    python run_ablation_new.py --config no_reshape_2d  # 不使用 1D→2D 重塑
    python run_ablation_new.py --config no_depthwise   # 不使用深度可分离卷积
    python run_ablation_new.py --config no_multi_scale # 不使用多尺度
    python run_ablation_new.py --config no_multi_period# 不使用多周期
    python run_ablation_new.py --config no_amp_weight  # 不使用 FFT 振幅权重
    python run_ablation_new.py --config no_residual    # 不使用残差
    python run_ablation_new.py --config no_revin       # 不使用 RevIN
    python run_ablation_new.py --config baseline_mlp   # MLP 基线
    
    # 运行全部消融实验
    python run_ablation_new.py --config all --epochs 20
    
    # 自定义参数
    python run_ablation_new.py --config full --team letsgrow --epochs 50 --seeds 42 123 456
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 添加项目路径
ABLATION_DIR = Path(__file__).parent.resolve()
TPLC_NET_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(TPLC_NET_DIR))

# 导入 tplc_algo 模块
from tplc_algo.pipeline import make_loaders, prepare_greenhouse_datasets
from tplc_algo.train import Trainer, TrainConfig
from tplc_algo.utils import seed_everything
from tplc_algo.config import TPLCConfig

# 导入消融模型
from ablation_model import TPLCNet_Ablation, AblationConfig, create_ablation_model


# ============================================================
# 数据集配置
# ============================================================

DATASETS = {
    "single_file": r"D:\学位会\数据集\温室环境数据(4万条)",  # 文件夹路径，不是文件
    "AICU": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛",
    "Reference": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛", 
    "IUACAAS": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛",
    "Automatoes": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛",
    "Digilog": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛",
    "TheAutomators": r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛",
}


# ============================================================
# 消融实验配置
# ============================================================

# 所有消融实验名称
ALL_ABLATION_NAMES = [
    "full",           # 完整模型（作为基准）
    "no_fft",         # 不使用 FFT 周期提取
    "no_reshape_2d",  # 不使用 1D→2D 重塑（纯 1D 卷积）
    "no_depthwise",   # 不使用深度可分离卷积（使用标准卷积）
    "no_multi_scale", # 不使用多尺度处理
    "no_multi_period",# 不使用多周期融合
    "no_amp_weight",  # 不使用 FFT 振幅权重
    "no_residual",    # 不使用残差连接
    "no_revin",       # 不使用 RevIN 归一化
    "baseline_mlp",   # MLP 基线（所有创新都关闭）
]


@dataclass
class ExperimentConfig:
    """实验总体配置。"""
    # 数据集
    team: str = "single_file"
    
    # 序列参数
    seq_len: int = 96
    pred_len: int = 24
    
    # 训练参数
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 6
    
    # 模型参数
    hidden_dim: int = 32
    num_scales: int = 2
    top_k_periods: int = 3
    dropout: float = 0.1
    
    # 随机种子（支持多个）
    seeds: List[int] = None
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42]


# ============================================================
# 训练与评估
# ============================================================

def compute_metrics(y_pred: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
    """计算评估指标。"""
    mae = np.mean(np.abs(y_pred - y_true))
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)
    
    # MAPE（避免除零）
    mask = np.abs(y_true) > 1e-8
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100
    else:
        mape = 0.0
    
    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """在测试集上评估模型。"""
    model.eval()
    preds_list = []
    trues_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)      # 输入数据
            y_true = batch[1].to(device) # 目标数据
            
            y_pred = model(x)
            
            preds_list.append(y_pred.cpu().numpy())
            trues_list.append(y_true.cpu().numpy())
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    metrics = compute_metrics(preds, trues)
    return metrics, preds, trues


def run_single_experiment(
    ablation_name: str,
    exp_config: ExperimentConfig,
    seed: int,
) -> Dict[str, Any]:
    """运行单次消融实验。"""
    
    print(f"\n{'='*60}")
    print(f"消融实验: {ablation_name} | 种子: {seed}")
    print(f"{'='*60}")
    
    # 设置随机种子
    seed_everything(seed)
    
    # 准备数据
    dataset_path = DATASETS.get(exp_config.team, exp_config.team)
    print(f"数据集: {dataset_path}")
    
    # 使用 TPLCConfig 的特征选择
    selected_features = list(TPLCConfig.feature_cols) if TPLCConfig.feature_cols else None
    if selected_features:
        print(f"使用指定特征: {len(selected_features)} 个 - {selected_features}")
    
    if exp_config.team == "single_file":
        prepared_data = prepare_greenhouse_datasets(
            dataset_root=Path(dataset_path),
            team="",
            seq_len=exp_config.seq_len,
            pred_len=exp_config.pred_len,
            selected_features=selected_features,
            base_table="GreenhouseClimate1.csv",
            extra_tables=(),
        )
    else:
        prepared_data = prepare_greenhouse_datasets(
            dataset_root=Path(dataset_path),
            team=exp_config.team,
            seq_len=exp_config.seq_len,
            pred_len=exp_config.pred_len,
            selected_features=selected_features,
        )
    
    train_ds = prepared_data.train_ds
    val_ds = prepared_data.val_ds
    test_ds = prepared_data.test_ds
    
    train_loader, val_loader, test_loader = make_loaders(
        prepared_data,
        batch_size=exp_config.batch_size,
    )
    
    # 获取数据维度
    sample = next(iter(train_loader))
    input_dim = sample[0].shape[-1]  # x 的最后一个维度
    target_dim = sample[1].shape[-1] # y 的最后一个维度
    
    print(f"输入维度: {input_dim}, 目标维度: {target_dim}")
    print(f"序列长度: {exp_config.seq_len}, 预测长度: {exp_config.pred_len}")
    
    # 创建消融配置
    ablation_cfg = AblationConfig(
        hidden_dim=exp_config.hidden_dim,
        num_scales=exp_config.num_scales,
        top_k_periods=exp_config.top_k_periods,
        dropout=exp_config.dropout,
    )
    
    # 根据消融名称修改配置
    config_methods = {
        "full": lambda c: c,
        "no_fft": lambda c: setattr(c, 'use_fft', False) or c,
        "no_reshape_2d": lambda c: setattr(c, 'use_reshape_2d', False) or c,
        "no_depthwise": lambda c: setattr(c, 'use_depthwise', False) or c,
        "no_multi_scale": lambda c: (setattr(c, 'use_multi_scale', False), setattr(c, 'num_scales', 0)) or c,
        "no_multi_period": lambda c: (setattr(c, 'use_multi_period', False), setattr(c, 'top_k_periods', 1)) or c,
        "no_amp_weight": lambda c: setattr(c, 'use_amp_weight', False) or c,
        "no_residual": lambda c: setattr(c, 'use_residual', False) or c,
        "no_revin": lambda c: setattr(c, 'use_revin', False) or c,
        "baseline_mlp": lambda c: (
            setattr(c, 'use_fft', False),
            setattr(c, 'use_reshape_2d', False),
            setattr(c, 'use_depthwise', False),
            setattr(c, 'use_multi_scale', False),
            setattr(c, 'use_multi_period', False),
            setattr(c, 'use_amp_weight', False),
            setattr(c, 'use_residual', False),
            setattr(c, 'use_revin', False),
        ) or c,
    }
    
    if ablation_name in config_methods:
        config_methods[ablation_name](ablation_cfg)
    else:
        raise ValueError(f"未知的消融名称: {ablation_name}")
    
    print(f"消融配置: {ablation_cfg}")
    
    # 创建模型
    model = TPLCNet_Ablation(
        input_dim=input_dim,
        target_dim=target_dim,
        seq_len=exp_config.seq_len,
        pred_len=exp_config.pred_len,
        config=ablation_cfg,
    )
    model = model.to(exp_config.device)
    
    # 统计参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params:,}")
    
    # 训练配置
    train_config = TrainConfig(
        epochs=exp_config.epochs,
        lr=exp_config.lr,
        weight_decay=exp_config.weight_decay,
        early_stop_patience=exp_config.early_stop_patience,
        grad_clip_max_norm=1.0,
        use_amp=True,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        cfg=train_config,
    )
    
    # 训练
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader)
    train_time = time.time() - start_time
    
    print(f"训练完成，耗时: {train_time:.1f}s")
    
    # 测试评估
    test_metrics, preds, trues = evaluate_model(
        model, test_loader, exp_config.device
    )
    
    print(f"测试指标: MAE={test_metrics['MAE']:.4f}, "
          f"MSE={test_metrics['MSE']:.4f}, "
          f"RMSE={test_metrics['RMSE']:.4f}")
    
    return {
        "ablation": ablation_name,
        "seed": seed,
        "num_params": num_params,
        "train_time": train_time,
        "history": history,
        **test_metrics,
    }


def run_all_experiments(
    ablation_names: List[str],
    exp_config: ExperimentConfig,
) -> pd.DataFrame:
    """运行所有消融实验。"""
    
    all_results = []
    
    for ablation_name in ablation_names:
        for seed in exp_config.seeds:
            try:
                result = run_single_experiment(ablation_name, exp_config, seed)
                all_results.append(result)
            except Exception as e:
                print(f"实验失败 [{ablation_name}, seed={seed}]: {e}")
                all_results.append({
                    "ablation": ablation_name,
                    "seed": seed,
                    "error": str(e),
                })
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_results)
    
    return df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """汇总多次实验结果。"""
    
    # 过滤掉出错的实验（检查 MAE 列是否存在）
    if "MAE" not in df.columns or df.empty:
        print("警告: 没有有效的实验结果")
        return pd.DataFrame()
    
    df_valid = df[~df["MAE"].isna()]
    
    # 按消融配置分组，计算均值和标准差
    summary = df_valid.groupby("ablation").agg({
        "MAE": ["mean", "std"],
        "MSE": ["mean", "std"],
        "RMSE": ["mean", "std"],
        "MAPE": ["mean", "std"],
        "num_params": "first",
        "train_time": "mean",
    }).round(4)
    
    # 展平列名
    summary.columns = [
        "MAE_mean", "MAE_std",
        "MSE_mean", "MSE_std",
        "RMSE_mean", "RMSE_std",
        "MAPE_mean", "MAPE_std",
        "Params", "Time_s"
    ]
    
    # 计算相对于 full 的性能变化
    if "full" in summary.index:
        full_mae = summary.loc["full", "MAE_mean"]
        summary["MAE_delta%"] = ((summary["MAE_mean"] - full_mae) / full_mae * 100).round(2)
    
    return summary.reset_index()


def print_ablation_table(summary: pd.DataFrame):
    """打印消融实验结果表格。"""
    
    if summary.empty:
        print("\n" + "=" * 60)
        print("消融实验失败，没有有效结果")
        print("=" * 60)
        return
    
    print("\n" + "=" * 100)
    print("消融实验结果汇总")
    print("=" * 100)
    
    # 消融名称到中文描述的映射
    ablation_desc = {
        "full": "完整模型 (基准)",
        "no_fft": "- FFT 周期提取",
        "no_reshape_2d": "- 1D→2D 重塑",
        "no_depthwise": "- 深度可分离卷积",
        "no_multi_scale": "- 多尺度处理",
        "no_multi_period": "- 多周期融合",
        "no_amp_weight": "- 振幅加权融合",
        "no_residual": "- 残差连接",
        "no_revin": "- RevIN 归一化",
        "baseline_mlp": "MLP 基线",
    }
    
    print(f"{'配置':<25} {'描述':<20} {'MAE':<15} {'RMSE':<15} {'参数量':<12} {'Δ MAE%':<10}")
    print("-" * 100)
    
    for _, row in summary.iterrows():
        name = row["ablation"]
        desc = ablation_desc.get(name, name)
        mae_str = f"{row['MAE_mean']:.4f}±{row['MAE_std']:.4f}"
        rmse_str = f"{row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}"
        params_str = f"{int(row['Params']):,}"
        delta_str = f"{row.get('MAE_delta%', 0):+.2f}%" if 'MAE_delta%' in row else "-"
        
        print(f"{name:<25} {desc:<20} {mae_str:<15} {rmse_str:<15} {params_str:<12} {delta_str:<10}")
    
    print("=" * 100)


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="TPLC 核心组件消融实验")
    
    # 消融配置
    parser.add_argument(
        "--config", type=str, default="full",
        help="消融配置名称，或 'all' 运行全部消融"
    )
    
    # 数据集
    parser.add_argument(
        "--team", type=str, default="single_file",
        choices=list(DATASETS.keys()),
        help="数据集名称"
    )
    
    # 序列参数
    parser.add_argument("--seq-len", type=int, default=96, help="输入序列长度")
    parser.add_argument("--pred-len", type=int, default=24, help="预测序列长度")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=32, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    
    # 随机种子
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42],
        help="随机种子列表（多个种子会取平均）"
    )
    
    # 输出
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="结果输出目录"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 创建实验配置
    exp_config = ExperimentConfig(
        team=args.team,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seeds=args.seeds,
    )
    
    print("\n" + "=" * 60)
    print("TPLC 核心组件消融实验")
    print("=" * 60)
    print(f"数据集: {args.team}")
    print(f"序列长度: {args.seq_len} → 预测长度: {args.pred_len}")
    print(f"训练轮数: {args.epochs}")
    print(f"随机种子: {args.seeds}")
    print("=" * 60)
    
    # 确定要运行的消融实验
    if args.config == "all":
        ablation_names = ALL_ABLATION_NAMES
    else:
        ablation_names = [args.config]
    
    print(f"待运行消融实验: {ablation_names}")
    
    # 运行实验
    results_df = run_all_experiments(ablation_names, exp_config)
    
    # 汇总结果
    summary = summarize_results(results_df)
    
    # 打印结果表格
    print_ablation_table(summary)
    
    # 保存结果
    output_dir = args.output_dir or str(ABLATION_DIR / "results")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    detail_path = Path(output_dir) / f"ablation_detail_{timestamp}.csv"
    results_df.to_csv(detail_path, index=False)
    print(f"\n详细结果已保存: {detail_path}")
    
    # 保存汇总结果（如果有有效结果）
    if not summary.empty:
        summary_path = Path(output_dir) / f"ablation_summary_{timestamp}.csv"
        summary.to_csv(summary_path, index=False)
        print(f"汇总结果已保存: {summary_path}")
    else:
        print("没有汇总结果可保存（所有实验都失败了）")
    
    return results_df, summary


if __name__ == "__main__":
    main()
