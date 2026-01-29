"""统一对比实验脚本：所有基线模型使用相同配置。

功能：
1. 统一训练配置（epochs, lr, early_stop 等）
2. 支持多随机种子运行取平均
3. 支持选择要运行的模型
4. 自动汇总结果到 CSV/JSON

使用示例：
    # 运行所有模型，单次
    python run_all_baselines.py

    # 运行所有模型，3个种子取平均
    python run_all_baselines.py --seeds 42 123 456

    # 只运行指定模型
    python run_all_baselines.py --models tplcnet lstm timesnet

    # 自定义配置
    python run_all_baselines.py --epochs 30 --lr 0.001 --pred-len 72
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 添加路径
scheme_root = Path(__file__).resolve().parent
tplc_net_root = scheme_root / 'TPLC_Net'
baselines_root = scheme_root / 'baselines'

sys.path.insert(0, str(tplc_net_root))
sys.path.insert(0, str(baselines_root))

# 导入工具
from tplc_algo.pipeline import prepare_greenhouse_datasets, make_loaders
from tplc_algo.train import Trainer, TrainConfig
from tplc_algo.utils import seed_everything
from tplc_algo.config import TPLCConfig
from tplc_algo.exp_utils import create_run_dir, save_json


@dataclass
class UnifiedConfig:
    """统一实验配置"""
    # 数据
    team: str = 'AICU'
    seq_len: int = 288
    pred_len: int = 72
    stride: int = 1
    batch_size: int = 32
    
    # 训练（所有模型统一）
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    
    # 稳定性配置（所有模型统一）
    grad_clip_max_norm: float = 1.0
    use_amp: bool = True
    lr_scheduler: str = 'plateau'
    plateau_patience: int = 3
    plateau_factor: float = 0.5
    early_stop_patience: int = 6
    
    # 实验设置
    seeds: List[int] = field(default_factory=lambda: [42])
    models: List[str] = field(default_factory=lambda: ['tplcnet', 'lstm', 'timesnet', 'patchtst', 'timemixer', 'timemixer_pp', 'transformer', 'tritracknet'])
    
    # 模型特定超参数（可覆盖）
    # TPLCNet
    tplc_hidden_dim: int = 32
    tplc_top_k_periods: int = 4
    tplc_num_scales: int = 1
    
    # LSTM
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    
    # TimesNet
    timesnet_d_model: int = 32
    timesnet_d_ff: int = 64
    timesnet_e_layers: int = 2
    timesnet_top_k: int = 5
    
    # PatchTST
    patchtst_d_model: int = 64
    patchtst_n_heads: int = 4
    patchtst_e_layers: int = 2
    patchtst_patch_len: int = 16
    
    # TimeMixer/TimeMixer++
    timemixer_d_model: int = 32
    timemixer_e_layers: int = 2


def load_models(cfg: UnifiedConfig, input_dim: int, target_dim: int) -> Dict[str, torch.nn.Module]:
    """按需加载模型"""
    models = {}
    
    if 'tplcnet' in cfg.models:
        from tplc_algo.models import TPLCNet
        models['tplcnet'] = lambda: TPLCNet(
            input_dim=input_dim,
            target_dim=target_dim,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            hidden_dim=cfg.tplc_hidden_dim,
            top_k_periods=cfg.tplc_top_k_periods,
            num_scales=cfg.tplc_num_scales,
        )
    
    if 'lstm' in cfg.models:
        from LSTM import LSTMForecaster
        models['lstm'] = lambda: LSTMForecaster(
            input_dim=input_dim,
            target_dim=target_dim,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            hidden_dim=cfg.lstm_hidden_dim,
            num_layers=cfg.lstm_num_layers,
            dropout=0.1,
        )
    
    if 'timesnet' in cfg.models:
        from TimesNet import TimesNetForecaster
        models['timesnet'] = lambda: TimesNetForecaster(
            input_dim=input_dim,
            target_dim=target_dim,
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            d_model=cfg.timesnet_d_model,
            d_ff=cfg.timesnet_d_ff,
            e_layers=cfg.timesnet_e_layers,
            top_k=cfg.timesnet_top_k,
            num_kernels=6,
            dropout=0.1,
        )
    
    if 'patchtst' in cfg.models:
        try:
            from PatchTST import PatchTSTForecaster
            models['patchtst'] = lambda: PatchTSTForecaster(
                input_dim=input_dim,
                target_dim=target_dim,
                seq_len=cfg.seq_len,
                pred_len=cfg.pred_len,
                d_model=cfg.patchtst_d_model,
                n_heads=cfg.patchtst_n_heads,
                e_layers=cfg.patchtst_e_layers,
                d_ff=cfg.patchtst_d_model * 2,
                patch_len=cfg.patchtst_patch_len,
                stride=cfg.patchtst_patch_len // 2,
                dropout=0.1,
            )
        except ImportError as e:
            print(f"⚠️ PatchTST 导入失败: {e}")
    
    if 'timemixer' in cfg.models:
        try:
            from TimeMixer import TimeMixerForecaster, TimeMixerConfig
            def create_timemixer():
                tm_cfg = TimeMixerConfig(
                    seq_len=cfg.seq_len,
                    pred_len=cfg.pred_len,
                    enc_in=input_dim,
                    c_out=target_dim,
                    d_model=cfg.timemixer_d_model,
                    d_ff=cfg.timemixer_d_model * 2,
                    e_layers=cfg.timemixer_e_layers,
                    down_sampling_layers=2,
                    down_sampling_method='avg',
                    channel_independence=False,
                )
                return TimeMixerForecaster(tm_cfg)
            models['timemixer'] = create_timemixer
        except ImportError as e:
            print(f"⚠️ TimeMixer 导入失败: {e}")
    
    if 'timemixer_pp' in cfg.models:
        try:
            # TimeMixer++ 需要特殊处理路径
            timemixer_pp_path = baselines_root / 'TimeMixer++'
            if str(timemixer_pp_path) not in sys.path:
                sys.path.append(str(timemixer_pp_path))
            from timemixer_pp_algo.model import TimeMixerPPForecaster
            from timemixer_pp_algo.config import TimeMixerPPConfig
            def create_timemixer_pp():
                tm_cfg = TimeMixerPPConfig(
                    seq_len=cfg.seq_len,
                    pred_len=cfg.pred_len,
                    enc_in=input_dim,
                    c_out=target_dim,
                    d_model=cfg.timemixer_d_model,
                    d_ff=cfg.timemixer_d_model * 2,
                    e_layers=cfg.timemixer_e_layers,
                    num_scales=2,
                    top_k=3,
                    dropout=0.1,
                    channel_independence=False,
                )
                return TimeMixerPPForecaster(tm_cfg)
            models['timemixer_pp'] = create_timemixer_pp
        except ImportError as e:
            print(f"⚠️ TimeMixer++ 导入失败: {e}")
    
    if 'transformer' in cfg.models:
        try:
            from Transformer import TransformerForecaster
            models['transformer'] = lambda: TransformerForecaster(
                input_dim=input_dim,
                target_dim=target_dim,
                seq_len=cfg.seq_len,
                pred_len=cfg.pred_len,
                d_model=64,
                n_heads=4,
                d_ff=128,
                e_layers=2,
                d_layers=1,
                dropout=0.1,
            )
        except ImportError as e:
            print(f"⚠️ Transformer 导入失败: {e}")
    
    if 'tritracknet' in cfg.models:
        try:
            # TriTrackNet 需要特殊处理路径
            tritracknet_path = baselines_root / 'TriTrackNet' / 'TriTrackNet'
            if str(tritracknet_path) not in sys.path:
                sys.path.insert(0, str(tritracknet_path))
            
            from TriTrackNetWrapper import TriTrackNetWrapper
            models['tritracknet'] = lambda: TriTrackNetWrapper(
                input_dim=input_dim,
                target_dim=target_dim,
                seq_len=cfg.seq_len,
                pred_len=cfg.pred_len,
            )
        except ImportError as e:
            print(f"⚠️ TriTrackNet 导入失败: {e}")
    
    return models


@torch.no_grad()
def compute_raw_metrics(model, loader, device, target_scaler, target_dim) -> Dict[str, float]:
    """计算反标准化后的指标"""
    model.eval()
    y_true_list, y_pred_list = [], []
    
    for x, y in loader:
        x = x.to(device)
        y_hat = model(x).cpu().numpy()
        y_true_list.append(y.numpy())
        y_pred_list.append(y_hat)
    
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)
    
    y_true_raw = target_scaler.inverse_transform(y_true.reshape(-1, target_dim)).reshape(y_true.shape)
    y_pred_raw = target_scaler.inverse_transform(y_pred.reshape(-1, target_dim)).reshape(y_pred.shape)
    
    return {
        'mae_raw': float(np.mean(np.abs(y_true_raw - y_pred_raw))),
        'rmse_raw': float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2))),
    }


def train_single_model(
    model_name: str,
    model: torch.nn.Module,
    cfg: UnifiedConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    target_scaler,
    target_dim: int,
    run_dir: Path,
    seed: int,
) -> Dict[str, Any]:
    """训练并评估单个模型"""
    import time
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_dir = run_dir / model_name / f'seed_{seed}'
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = model_dir / 'best.pt'
    
    trainer = Trainer(
        model=model,
        cfg=TrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=device,
            ckpt_path=ckpt_path,
            grad_clip_max_norm=cfg.grad_clip_max_norm,
            use_amp=cfg.use_amp and device == 'cuda',
            lr_scheduler=cfg.lr_scheduler,
            plateau_patience=cfg.plateau_patience,
            plateau_factor=cfg.plateau_factor,
            early_stop_patience=cfg.early_stop_patience,
            show_progress=True,
            progress_granularity='epoch',
        ),
    )
    
    # 记录训练时间
    start_time = time.time()
    history = trainer.fit(train_loader, val_loader=val_loader)
    train_time = time.time() - start_time
    
    metrics = trainer.evaluate(test_loader)
    
    # 计算原始尺度指标
    raw_metrics = compute_raw_metrics(model, test_loader, device, target_scaler, target_dim)
    metrics.update(raw_metrics)
    
    # 添加参数量和训练时间
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    metrics['train_time'] = train_time
    
    # 保存
    save_json(model_dir / 'metrics.json', metrics)
    save_json(model_dir / 'history.json', history)
    
    print(f"  📊 {model_name} (seed={seed}): MAE={metrics['mae']:.4f}, 参数={trainable_params/1e3:.1f}K, 耗时={train_time:.1f}s")
    
    return {
        'model': model_name,
        'seed': seed,
        **metrics,
    }


def run_experiment(cfg: UnifiedConfig) -> pd.DataFrame:
    """运行完整实验"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n{'='*60}")
    print(f"统一对比实验")
    print(f"{'='*60}")
    print(f"设备: {device}")
    print(f"模型: {cfg.models}")
    print(f"随机种子: {cfg.seeds}")
    print(f"训练配置: epochs={cfg.epochs}, lr={cfg.lr}, early_stop={cfg.early_stop_patience}")
    print(f"{'='*60}\n")
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = create_run_dir(
        f'unified_compare_{cfg.team}_{timestamp}',
        base_dir=scheme_root / 'compare_result'
    )
    print(f"实验目录: {run_dir}\n")
    
    # 保存配置
    save_json(run_dir / 'config.json', asdict(cfg))
    
    # 准备数据（只需一次）
    dataset_root = scheme_root / 'datasets' / '自主温室挑战赛'
    selected_features = list(TPLCConfig.feature_cols) if TPLCConfig.feature_cols else None
    target_cols_cfg = list(TPLCConfig.target_cols) if TPLCConfig.target_cols else None
    
    print(f"加载数据: {dataset_root / cfg.team}")
    if selected_features:
        print(f"输入特征: {len(selected_features)} 个")
    if target_cols_cfg:
        print(f"目标变量: {target_cols_cfg}")
    
    prepared = prepare_greenhouse_datasets(
        dataset_root=dataset_root,
        team=cfg.team,
        seq_len=cfg.seq_len,
        pred_len=cfg.pred_len,
        stride=cfg.stride,
        selected_features=selected_features,
        target_cols=target_cols_cfg,
        missing_rate_threshold=0.7,
        drop_constant=True,
        protect_target_cols=True,
    )
    
    input_dim = len(prepared.feature_cols)
    target_dim = len(prepared.target_cols)
    print(f"Input dim: {input_dim}, Target dim: {target_dim}")
    
    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=cfg.batch_size)
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}\n")
    
    # 加载模型工厂
    model_factories = load_models(cfg, input_dim, target_dim)
    
    # 运行实验
    all_results = []
    
    for model_name in cfg.models:
        if model_name not in model_factories:
            print(f"⚠️ 跳过 {model_name}（未找到或导入失败）")
            continue
        
        for seed in cfg.seeds:
            print(f"\n{'='*40}")
            print(f"训练 {model_name} (seed={seed})")
            print(f"{'='*40}")
            
            seed_everything(seed)
            
            # 创建新模型实例
            model = model_factories[model_name]()
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"参数量: {param_count:,}")
            
            result = train_single_model(
                model_name=model_name,
                model=model,
                cfg=cfg,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                target_scaler=prepared.target_scaler,
                target_dim=target_dim,
                run_dir=run_dir,
                seed=seed,
            )
            all_results.append(result)
            
            print(f"✅ {model_name} (seed={seed}): MAE_raw={result['mae_raw']:.4f}, RMSE_raw={result['rmse_raw']:.4f}")
    
    # 汇总结果
    results_df = pd.DataFrame(all_results)
    
    # 检查是否有结果
    if len(results_df) == 0:
        print("\n❌ 没有模型成功运行，请检查模型导入和配置！")
        return None
    
    # 按模型聚合（多种子取平均）
    if len(cfg.seeds) > 1:
        summary_df = results_df.groupby('model').agg({
            'loss': ['mean', 'std'],
            'mae': ['mean', 'std'],
            'rmse': ['mean', 'std'],
            'mae_raw': ['mean', 'std'],
            'rmse_raw': ['mean', 'std'],
        }).round(4)
        summary_df.columns = ['_'.join(col) for col in summary_df.columns]
        summary_df = summary_df.reset_index()
    else:
        summary_df = results_df[['model', 'loss', 'mae', 'rmse', 'mae_raw', 'rmse_raw']].round(4)
    
    # 保存结果
    results_df.to_csv(run_dir / 'all_results.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(run_dir / 'summary.csv', index=False, encoding='utf-8-sig')
    
    print(f"\n{'='*60}")
    print("实验完成！汇总结果：")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\n结果保存至: {run_dir}")
    
    return summary_df


def parse_args():
    parser = argparse.ArgumentParser(description='统一对比实验')
    
    # 数据配置
    parser.add_argument('--team', type=str, default='AICU')
    parser.add_argument('--seq-len', type=int, default=288)
    parser.add_argument('--pred-len', type=int, default=72)
    parser.add_argument('--batch-size', type=int, default=32)
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early-stop', type=int, default=6)
    
    # 实验配置
    parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                        help='随机种子列表，如 --seeds 42 123 456')
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['tplcnet', 'lstm', 'timesnet'],
                        help='要运行的模型，如 --models tplcnet lstm timesnet patchtst')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = UnifiedConfig(
        team=args.team,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        early_stop_patience=args.early_stop,
        seeds=args.seeds,
        models=args.models,
    )
    
    run_experiment(cfg)


if __name__ == '__main__':
    main()
