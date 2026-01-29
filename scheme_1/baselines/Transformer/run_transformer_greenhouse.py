"""Transformer 温室数据统一训练脚本。

与其他基线保持一致：数据管线复用 TPLC_Net，训练评估统一。
"""

from pathlib import Path
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

tplc_path = Path(__file__).parent.parent.parent / 'TPLC_Net'
if str(tplc_path) not in sys.path:
    sys.path.insert(0, str(tplc_path))

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

from .transformer_forecaster import TransformerForecaster

# 导入特征配置
from tplc_algo.config import TPLCConfig


def main():
    seed_everything(42)

    scheme_root = Path(__file__).resolve().parent.parent.parent
    dataset_root = scheme_root / 'datasets' / '自主温室挑战赛'
    team = 'AICU'
    seq_len = 288
    pred_len = 72
    stride = 1
    batch_size = 32

    d_model = 64
    n_heads = 4
    d_ff = 128
    e_layers = 2
    d_layers = 1
    factor = 3
    dropout = 0.1

    epochs = 20
    lr = 1e-3
    weight_decay = 0.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_amp = (device == 'cuda')
    grad_clip_max_norm = 1.0
    lr_scheduler = 'plateau'
    plateau_patience = 3
    plateau_factor = 0.5
    early_stop_patience = 6

    exp_name = f"transformer_greenhouse_{team}"
    run_dir = create_run_dir(exp_name, base_dir=Path(__file__).parent / 'results')
    ckpt_path = run_dir / 'checkpoints' / 'transformer_best.pt'

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

    model = TransformerForecaster(
        input_dim=len(feature_cols),
        target_dim=len(target_cols),
        seq_len=seq_len,
        pred_len=pred_len,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        e_layers=e_layers,
        d_layers=d_layers,
        factor=factor,
        dropout=dropout,
    )

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

    history = trainer.fit(train_loader, val_loader=val_loader)

    test_metrics = trainer.evaluate(test_loader)

    # 反标准化指标
    model.eval()
    y_true_list, y_pred_list = [], []
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

    save_metrics_json(run_dir, test_metrics)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='训练损失')
    if len(history.get('val_loss', [])) > 0:
        plt.plot(history['val_loss'], label='验证损失')
    plt.legend()
    plt.title('Transformer 训练曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, run_dir / 'figures' / 'loss_curve.png')
    plt.close()

    print(f'完成，结果保存在：{run_dir}')


if __name__ == '__main__':
    main()
