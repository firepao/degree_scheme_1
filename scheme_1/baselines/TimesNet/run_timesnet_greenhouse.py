"""TimesNet 温室数据训练脚本。

使用方法：
    python run_timesnet_greenhouse.py

说明：
    - 数据加载：复用 TPLC_Net 项目的数据加载逻辑
    - 模型：TimesNet 预测模型
    - 训练：复用 TPLC_Net 的 Trainer
"""

from pathlib import Path
import sys

# 添加 TPLC_Net 到路径以复用数据加载和训练工具
tplc_path = Path(__file__).parent.parent.parent / 'TPLC_Net'
if str(tplc_path) not in sys.path:
    sys.path.insert(0, str(tplc_path))

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

# 导入 TimesNet
from baselines.TimesNet import TimesNetForecaster


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

    # TimesNet 模型配置
    d_model = 64
    d_ff = 128
    e_layers = 2
    top_k = 3
    num_kernels = 6
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

    # 实验目录
    exp_name = f"timesnet_greenhouse_{team}"
    run_dir = create_run_dir(exp_name, base_dir=Path(__file__).parent / 'results')
    ckpt_path = run_dir / 'checkpoints' / 'timesnet_best.pt'

    print(f'Device: {device}')
    print(f'Dataset: {dataset_root}')
    print(f'Team: {team}')
    print(f'Run directory: {run_dir}')

    # ========= 2) 数据准备 =========
    prepared = prepare_greenhouse_datasets(
        dataset_root=dataset_root,
        team=team,
        seq_len=seq_len,
        pred_len=pred_len,
        stride=stride,
        missing_rate_threshold=0.7,
        drop_constant=True,
        protect_target_cols=True,
    )

    feature_cols = prepared.feature_cols
    target_cols = prepared.target_cols
    target_scaler = prepared.target_scaler

    train_loader, val_loader, test_loader = make_loaders(prepared, batch_size=batch_size)

    print(f'Input dim: {len(feature_cols)}')
    print(f'Target dim: {len(target_cols)} -> {target_cols}')
    print(f'Train batches: {len(train_loader)}')

    # ========= 3) 构建 TimesNet 模型 =========
    model = TimesNetForecaster(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        top_k=top_k,
        num_kernels=num_kernels,
        dropout=dropout,
    )

    print(f'\nTimesNet 模型参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # ========= 4) 训练 =========
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
            progress_granularity='epoch',
        )
    )

    print('\n开始训练...')
    history = trainer.fit(train_loader, val_loader=val_loader)

    # ========= 5) 保存配置和历史 =========
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
            'top_k': top_k,
            'num_kernels': num_kernels,
            'dropout': dropout,
        },
        'train': {
            'batch_size': batch_size,
            'epochs': epochs,
            'lr': lr,
            'weight_decay': weight_decay,
            'device': device,
        }
    }

    save_config_json(run_dir, config_payload)
    save_env_json(run_dir)
    save_history_csv(run_dir, history)

    # ========= 6) 测试集评估 =========
    print('\n在测试集上评估...')
    test_metrics = trainer.evaluate(test_loader)

    # 反标准化后的指标
    import numpy as np
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

    print('\n测试集指标：')
    for k, v in test_metrics.items():
        print(f'  {k}: {v:.4f}')

    save_metrics_json(run_dir, test_metrics)

    # ========= 7) 可视化 =========
    # Loss 曲线
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='训练损失')
    if len(history.get('val_loss', [])) > 0:
        plt.plot(history['val_loss'], label='验证损失')
    plt.legend()
    plt.title('TimesNet 训练曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, run_dir / 'figures' / 'loss_curve.png')
    plt.show()

    # 单样本预测
    sample_idx = 0
    var_idx = 0
    fig = plt.figure(figsize=(10, 4))
    plt.plot(y_true_raw[sample_idx, :, var_idx], label='真实值', marker='o', markersize=3)
    plt.plot(y_pred_raw[sample_idx, :, var_idx], label='预测值', marker='x', markersize=3)
    plt.title(f'TimesNet 预测示例：{target_cols[var_idx]}')
    plt.xlabel('时间步')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, run_dir / 'figures' / f'pred_curve_{target_cols[var_idx]}.png')
    plt.show()

    print(f'\n所有结果已保存到：{run_dir}')


if __name__ == '__main__':
    main()
