"""TimeMixer 温室数据训练脚本。

使用方法：
    python run_timemixer_greenhouse.py

说明：
    - 数据加载：复用 TPLC_Net 项目的数据加载逻辑
    - 模型：TimeMixer 预测模型
    - 训练：复用 TPLC_Net 的 Trainer
"""

from pathlib import Path
import sys

# 添加 TPLC_Net 到路径以复用数据加载和训练工具
tplc_path = Path(__file__).parent.parent.parent / 'TPLC_Net'
if str(tplc_path) not in sys.path:
    sys.path.insert(0, str(tplc_path))

import numpy as np
import torch
import matplotlib.pyplot as plt

# 中文支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入 TPLC_Net 的工具
from tplc_algo.pipeline import prepare_greenhouse_datasets, make_loaders
from tplc_algo.train import Trainer, TrainConfig
from tplc_algo.utils import seed_everything
from tplc_algo.exp_utils import (
    create_run_dir,
    save_config_json,
    save_env_json,
    save_history_csv,
    save_metrics_json,
    save_figure,
)

# 导入 TimeMixer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from timemixer import TimeMixerForecaster, TimeMixerConfig

# 导入特征配置
from tplc_algo.config import TPLCConfig


def main():
    # ========= 1) 配置 =========
    seed_everything(42)

    # 数据配置
    dataset_root = (Path(__file__).parent.parent.parent / 'datasets' / '自主温室挑战赛').resolve()
    team = 'AICU'
    seq_len = 288
    pred_len = 72
    stride = 1
    batch_size = 32

    # 模型配置
    d_model = 64
    d_ff = 128
    e_layers = 2
    down_sampling_layers = 2
    down_sampling_window = 2
    down_sampling_method = 'avg'
    decomp_method = 'moving_avg'
    moving_avg = 25
    top_k = 5
    channel_independence = True
    dropout = 0.1

    # 训练配置
    epochs = 20
    lr = 1e-3
    weight_decay = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 稳定性配置
    use_amp = (device == 'cuda')
    grad_clip_max_norm = 1.0
    lr_scheduler = 'plateau'
    plateau_patience = 3
    plateau_factor = 0.5
    early_stop_patience = 6

    # 结果目录
    exp_name = f"timemixer_greenhouse_{team}"
    run_dir = create_run_dir(exp_name, base_dir=Path(__file__).parent / 'results')
    ckpt_path = run_dir / 'checkpoints' / 'timemixer_best.pt'
    
    print('='*60)
    print(f'TimeMixer 温室数据预测实验')
    print('='*60)
    print(f'device: {device}')
    print(f'team: {team}')
    print(f'dataset_root: {dataset_root}')
    print(f'run_dir: {run_dir}')
    print('='*60)

    # ========= 2) 数据准备 =========
    print('\n准备数据...')
    # 从 TPLCConfig 读取特征配置（与 run_compare_experiment.ipynb 保持一致）
    selected_features = list(TPLCConfig.feature_cols) if TPLCConfig.feature_cols else None
    target_cols_cfg = list(TPLCConfig.target_cols) if TPLCConfig.target_cols else None

    if selected_features:
        print(f'Config: 使用指定输入特征 ({len(selected_features)} 个)')
    if target_cols_cfg:
        print(f'Config: 预测目标变量: {target_cols_cfg}')

    prepared = prepare_greenhouse_datasets(
        dataset_root=dataset_root,
        team=team,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        selected_features=selected_features,
        target_cols=target_cols_cfg,
        missing_rate_threshold=0.7,
        drop_constant=True,
        protect_target_cols=True,
    )

    feature_cols = prepared.feature_cols
    target_cols = prepared.target_cols
    target_scaler = prepared.target_scaler

    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=batch_size)

    print(f'input_dim: {len(feature_cols)}')
    print(f'target_dim: {len(target_cols)} {target_cols}')
    print(f'train batches: {len(train_loader)}')
    print(f'val batches: {len(val_loader)}')
    print(f'test batches: {len(test_loader)}')

    # ========= 3) 构建模型 =========
    print('\n构建 TimeMixer 模型...')
    config = TimeMixerConfig(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=len(feature_cols),
        c_out=len(target_cols),
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        dropout=dropout,
        down_sampling_layers=down_sampling_layers,
        down_sampling_window=down_sampling_window,
        down_sampling_method=down_sampling_method,
        decomp_method=decomp_method,
        moving_avg=moving_avg,
        top_k=top_k,
        channel_independence=channel_independence,
    )
    
    model = TimeMixerForecaster(config)
    
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    # ========= 4) 训练 =========
    print('\n开始训练...')
    trainer = Trainer(
        model=model,
        cfg=TrainConfig(
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            ckpt_path=ckpt_path,
            use_amp=use_amp,
            grad_clip_max_norm=grad_clip_max_norm,
            lr_scheduler=lr_scheduler,
            plateau_patience=plateau_patience,
            plateau_factor=plateau_factor,
            early_stop_patience=early_stop_patience,
            show_progress=True,
            progress_granularity='batch',
        )
    )
    
    history = trainer.fit(train_loader, val_loader=val_loader)

    # ========= 5) 保存训练信息 =========
    config_payload = {
        'team': team,
        'dataset_root': str(dataset_root),
        'seq_len': seq_len,
        'pred_len': pred_len,
        'stride': stride,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'model': {
            'd_model': d_model,
            'd_ff': d_ff,
            'e_layers': e_layers,
            'down_sampling_layers': down_sampling_layers,
            'down_sampling_window': down_sampling_window,
            'down_sampling_method': down_sampling_method,
            'decomp_method': decomp_method,
            'moving_avg': moving_avg,
            'top_k': top_k,
            'channel_independence': channel_independence,
            'dropout': dropout,
        },
        'train': {
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'device': device,
            'use_amp': use_amp,
            'grad_clip_max_norm': grad_clip_max_norm,
            'lr_scheduler': lr_scheduler,
        },
    }
    
    save_config_json(run_dir, config_payload)
    save_env_json(run_dir)
    save_history_csv(run_dir, history)

    # ========= 6) 测试集评估 =========
    print('\n测试集评估...')
    test_metrics = trainer.evaluate(test_loader)

    # 反标准化后的指标
    model.eval()
    y_true_list = []
    y_pred_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_hat = model(x).cpu().numpy()
            y_true_list.append(y.numpy())
            y_pred_list.append(y_hat)

    y_true = np.concatenate(y_true_list, axis=0)
    y_pred = np.concatenate(y_pred_list, axis=0)

    y_true_raw = target_scaler.inverse_transform(
        y_true.reshape(-1, len(target_cols))
    ).reshape(y_true.shape)
    y_pred_raw = target_scaler.inverse_transform(
        y_pred.reshape(-1, len(target_cols))
    ).reshape(y_pred.shape)

    test_metrics['mae_raw'] = float(np.mean(np.abs(y_true_raw - y_pred_raw)))
    test_metrics['rmse_raw'] = float(np.sqrt(np.mean((y_true_raw - y_pred_raw) ** 2)))

    print('\n测试集指标:')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.6f}')

    save_metrics_json(run_dir, test_metrics)

    # ========= 7) 可视化 =========
    print('\n生成可视化...')
    
    # Loss 曲线
    fig = plt.figure(figsize=(7, 4))
    plt.plot(history['train_loss'], label='train_loss')
    if len(history.get('val_loss', [])) > 0:
        plt.plot(history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('训练/验证损失')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.tight_layout()
    save_figure(fig, run_dir / 'figures' / 'loss_curve.png')
    plt.close()

    # 单样本预测可视化
    model.eval()
    x, y = next(iter(test_loader))
    x = x.to(device)
    with torch.no_grad():
        y_hat = model(x).cpu().numpy()
    y_true = y.numpy()

    y_hat_inv = target_scaler.inverse_transform(
        y_hat.reshape(-1, len(target_cols))
    ).reshape(y_hat.shape)
    y_true_inv = target_scaler.inverse_transform(
        y_true.reshape(-1, len(target_cols))
    ).reshape(y_true.shape)

    # 为每个目标变量生成预测曲线
    for var_idx, var_name in enumerate(target_cols):
        fig = plt.figure(figsize=(10, 4))
        sample = 0
        plt.plot(y_true_inv[sample, :, var_idx], label='真实', marker='o', markersize=3)
        plt.plot(y_hat_inv[sample, :, var_idx], label='预测', marker='s', markersize=3)
        plt.title(f'目标变量：{var_name}（样本 {sample}）')
        plt.xlabel('时间步')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        save_figure(fig, run_dir / 'figures' / f'pred_curve_{var_name}.png')
        plt.close()

    print(f'\n实验完成！结果保存在: {run_dir}')


if __name__ == '__main__':
    main()
