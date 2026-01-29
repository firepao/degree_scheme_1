"""TPLC 模型消融实验主脚本。

使用方式:
    python run_ablation.py --config baseline        # 基线模型
    python run_ablation.py --config +revin          # + RevIN
    python run_ablation.py --config +inception      # + Inception Block
    python run_ablation.py --config +decomp         # + 季节-趋势分解
    python run_ablation.py --config +stacked        # + 多层堆叠
    python run_ablation.py --config all             # 运行所有配置
    python run_ablation.py --config full            # 完整改进模型
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加项目路径
ABLATION_DIR = Path(__file__).parent.resolve()
TPLC_NET_DIR = ABLATION_DIR.parent
sys.path.insert(0, str(TPLC_NET_DIR))

# 导入 tplc_algo 模块
from tplc_algo.pipeline import make_loaders, prepare_greenhouse_datasets
from tplc_algo.train import Trainer, TrainConfig
from tplc_algo.utils import seed_everything
from tplc_algo.exp_utils import create_run_dir, save_config_json, save_metrics_json, save_history_csv

# 导入消融模块
from revin import RevIN
from inception import Inception_Block_V1
from decomposition import DFT_series_decomp, series_decomp
from stacked_block import TPLCBlock

# ============================================================
# 配置定义
# ============================================================

@dataclass
class AblationConfig:
    """消融实验配置。"""
    # 实验名称
    name: str = "baseline"
    
    # 模型配置
    input_dim: int = 10
    target_dim: int = 10
    seq_len: int = 96
    pred_len: int = 24
    hidden_dim: int = 64
    num_scales: int = 2
    top_k_periods: int = 3
    dropout: float = 0.1
    
    # 消融开关
    use_revin: bool = False           # RevIN 归一化
    use_inception: bool = False       # Inception Block（替代 DepthwiseSeparable）
    use_decomp: bool = False          # 季节-趋势分解
    decomp_method: str = "dft"        # 分解方法: "dft" 或 "moving_avg"
    decomp_kernel: int = 25           # Moving Avg 核大小
    use_stacked: bool = False         # 多层堆叠
    e_layers: int = 2                 # 堆叠层数
    
    # 训练配置
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    early_stop_patience: int = 6
    
    # 数据配置
    dataset_type: str = "single_file"
    dataset_path: str = r"D:\学位会\数据集\温室环境数据(4万条)\GreenhouseClimate1.csv"
    
    # 其他
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# 预定义配置
ABLATION_CONFIGS = {
    "baseline": AblationConfig(name="baseline"),
    "+revin": AblationConfig(name="+revin", use_revin=True),
    "+inception": AblationConfig(name="+inception", use_inception=True),
    "+decomp": AblationConfig(name="+decomp", use_decomp=True),
    "+decomp_ma": AblationConfig(name="+decomp_ma", use_decomp=True, decomp_method="moving_avg"),
    "+stacked": AblationConfig(name="+stacked", use_stacked=True, e_layers=2),
    "+stacked3": AblationConfig(name="+stacked3", use_stacked=True, e_layers=3),
    # 组合配置
    "+revin+inception": AblationConfig(name="+revin+inception", use_revin=True, use_inception=True),
    "+revin+decomp": AblationConfig(name="+revin+decomp", use_revin=True, use_decomp=True),
    "+revin+stacked": AblationConfig(name="+revin+stacked", use_revin=True, use_stacked=True),
    # 完整模型
    "full": AblationConfig(
        name="full",
        use_revin=True,
        use_inception=True,
        use_decomp=True,
        use_stacked=True,
        e_layers=2,
    ),
}


# ============================================================
# 消融模型定义
# ============================================================

class TPLCNet_Ablation(nn.Module):
    """TPLC 消融实验模型。
    
    通过配置开关控制各模块的启用/禁用。
    """
    
    def __init__(self, cfg: AblationConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # 1. RevIN 归一化
        self.revin = None
        if cfg.use_revin:
            self.revin = RevIN(num_features=cfg.input_dim, affine=True)
        
        # 2. 季节-趋势分解
        self.decomp = None
        if cfg.use_decomp:
            if cfg.decomp_method == "dft":
                self.decomp = DFT_series_decomp(top_k=cfg.top_k_periods)
            else:
                self.decomp = series_decomp(kernel_size=cfg.decomp_kernel)
        
        # 3. Embedding 层
        self.embedding = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        
        # 4. 主干网络
        if cfg.use_stacked:
            # 多层堆叠
            self.blocks = nn.ModuleList([
                TPLCBlock(
                    d_model=cfg.hidden_dim,
                    seq_len=cfg.seq_len,
                    pred_len=cfg.pred_len,
                    top_k=cfg.top_k_periods,
                    num_scales=cfg.num_scales,
                    use_inception=cfg.use_inception,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.e_layers)
            ])
            self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        else:
            # 单层
            self.blocks = nn.ModuleList([
                TPLCBlock(
                    d_model=cfg.hidden_dim,
                    seq_len=cfg.seq_len,
                    pred_len=cfg.pred_len,
                    top_k=cfg.top_k_periods,
                    num_scales=cfg.num_scales,
                    use_inception=cfg.use_inception,
                    dropout=cfg.dropout,
                )
            ])
            self.layer_norm = nn.LayerNorm(cfg.hidden_dim)
        
        # 5. 预测头
        self.projection = nn.Linear(cfg.hidden_dim, cfg.target_dim)
        
        # 6. 时间维度投影（seq_len -> pred_len）
        self.time_proj = nn.Linear(cfg.seq_len, cfg.pred_len)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, input_dim]
        Returns:
            y: [B, pred_len, target_dim]
        """
        # 1. RevIN 归一化
        if self.revin is not None:
            x = self.revin(x, mode='norm')
        
        # 2. 季节-趋势分解
        x_season, x_trend = None, None
        if self.decomp is not None:
            x_season, x_trend = self.decomp(x)
            # 这里简单处理：只用季节性成分进入主干
            # 趋势性成分后面加回来
            x_main = x_season
        else:
            x_main = x
            x_trend = torch.zeros_like(x)
        
        # 3. Embedding
        enc_out = self.embedding(x_main)  # [B, seq_len, hidden_dim]
        
        # 4. 主干网络（TPLC Blocks）
        for block in self.blocks:
            enc_out = self.layer_norm(block(enc_out))
        
        # 5. 时间投影
        # [B, seq_len, hidden_dim] -> [B, hidden_dim, seq_len] -> [B, hidden_dim, pred_len]
        enc_out = enc_out.transpose(1, 2)
        enc_out = self.time_proj(enc_out)
        enc_out = enc_out.transpose(1, 2)  # [B, pred_len, hidden_dim]
        
        # 6. 预测投影
        y = self.projection(enc_out)  # [B, pred_len, target_dim]
        
        # 7. 加回趋势（如果使用分解）
        if self.decomp is not None and x_trend is not None:
            # 趋势投影到 pred_len
            trend_proj = x_trend[:, -self.cfg.pred_len:, :self.cfg.target_dim]
            if trend_proj.shape[1] < self.cfg.pred_len:
                trend_proj = F.pad(trend_proj, (0, 0, self.cfg.pred_len - trend_proj.shape[1], 0))
            y = y + trend_proj
        
        # 8. RevIN 反归一化
        if self.revin is not None:
            y = self.revin(y, mode='denorm')
        
        return y


# ============================================================
# 实验运行器
# ============================================================

class AblationRunner:
    """消融实验运行器。"""
    
    def __init__(self, cfg: AblationConfig):
        self.cfg = cfg
        self.results_dir = ABLATION_DIR / "results"
        self.results_dir.mkdir(exist_ok=True)
        
    def load_data(self):
        """加载数据。"""
        print(f"\n{'='*60}")
        print(f"加载数据: {self.cfg.dataset_path}")
        print(f"{'='*60}")
        
        dataset_root = Path(self.cfg.dataset_path).parent
        
        # 核心特征
        selected_features = [
            'Tair', 'Rhair', 'CO2air', 'Tot_PAR', 'PipeGrow',
            'VentLee', 'Ventwind', 'HumDef', 'water_sup', 'EnScr'
        ]
        
        prepared = prepare_greenhouse_datasets(
            dataset_root=dataset_root,
            team='.',
            seq_len=self.cfg.seq_len,
            pred_len=self.cfg.pred_len,
            stride=1,
            base_table=Path(self.cfg.dataset_path).name,
            extra_tables=(),
            selected_features=selected_features,
            missing_rate_threshold=0.7,
            drop_constant=True,
            protect_target_cols=True,
        )
        
        self.feature_cols = prepared.feature_cols
        self.target_cols = prepared.target_cols
        self.target_scaler = prepared.target_scaler
        
        # 更新配置中的维度
        self.cfg.input_dim = len(self.feature_cols)
        self.cfg.target_dim = len(self.target_cols)
        
        train_loader, val_loader, test_loader = make_loaders(
            prepared, batch_size=self.cfg.batch_size
        )
        
        print(f"input_dim = {self.cfg.input_dim}")
        print(f"target_dim = {self.cfg.target_dim}")
        print(f"train batches: {len(train_loader)}")
        print(f"val batches: {len(val_loader)}")
        print(f"test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def build_model(self) -> nn.Module:
        """构建模型。"""
        print(f"\n{'='*60}")
        print(f"构建模型: {self.cfg.name}")
        print(f"{'='*60}")
        print(f"  use_revin: {self.cfg.use_revin}")
        print(f"  use_inception: {self.cfg.use_inception}")
        print(f"  use_decomp: {self.cfg.use_decomp} ({self.cfg.decomp_method})")
        print(f"  use_stacked: {self.cfg.use_stacked} (e_layers={self.cfg.e_layers})")
        
        model = TPLCNet_Ablation(self.cfg)
        
        # 打印模型参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        return model
    
    def run(self) -> Dict[str, Any]:
        """运行单次实验。"""
        seed_everything(self.cfg.seed)
        
        # 1. 加载数据
        train_loader, val_loader, test_loader = self.load_data()
        
        # 2. 构建模型
        model = self.build_model()
        
        # 3. 创建结果目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"{self.cfg.name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = run_dir / "checkpoints" / "best.pt"
        
        # 4. 训练
        print(f"\n{'='*60}")
        print(f"开始训练")
        print(f"{'='*60}")
        
        trainer = Trainer(
            model=model,
            cfg=TrainConfig(
                epochs=self.cfg.epochs,
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
                device=self.cfg.device,
                ckpt_path=ckpt_path,
                use_amp=(self.cfg.device == 'cuda'),
                grad_clip_max_norm=1.0,
                lr_scheduler='plateau',
                plateau_patience=3,
                plateau_factor=0.5,
                early_stop_patience=self.cfg.early_stop_patience,
                show_progress=True,
                progress_granularity='epoch',
            )
        )
        
        start_time = time.time()
        history = trainer.fit(train_loader, val_loader=val_loader)
        train_time = time.time() - start_time
        
        # 5. 测试
        print(f"\n{'='*60}")
        print(f"测试评估")
        print(f"{'='*60}")
        
        metrics = trainer.evaluate(test_loader)
        metrics['train_time'] = train_time
        metrics['config'] = self.cfg.name
        
        # 补全缺失的指标 (Trainer.evaluate 返回 loss, mae, rmse)
        if 'mse' not in metrics and 'loss' in metrics:
            metrics['mse'] = metrics['loss']
        elif 'mse' not in metrics and 'rmse' in metrics:
            metrics['mse'] = metrics['rmse'] ** 2
        if 'mape' not in metrics:
            metrics['mape'] = float('nan')
        
        print(f"  MAE:  {metrics.get('mae', -1):.4f}")
        print(f"  MSE:  {metrics.get('mse', -1):.4f}")
        print(f"  RMSE: {metrics.get('rmse', -1):.4f}")
        print(f"  MAPE: {metrics.get('mape', float('nan')):.4f}")
        print(f"  训练时间: {train_time:.2f}s")
        
        # 6. 保存结果
        save_config_json(run_dir, asdict(self.cfg))
        save_metrics_json(run_dir, metrics)
        save_history_csv(run_dir, history)
        
        return metrics


def run_single_config(config_name: str) -> Dict[str, Any]:
    """运行单个配置。"""
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"未知配置: {config_name}，可用配置: {list(ABLATION_CONFIGS.keys())}")
    
    cfg = ABLATION_CONFIGS[config_name]
    runner = AblationRunner(cfg)
    return runner.run()


def run_all_configs() -> pd.DataFrame:
    """运行所有配置并生成对比表格。"""
    results = []
    
    for config_name in ABLATION_CONFIGS:
        print(f"\n{'#'*70}")
        print(f"# 运行配置: {config_name}")
        print(f"{'#'*70}")
        
        try:
            metrics = run_single_config(config_name)
            results.append(metrics)
        except Exception as e:
            print(f"[ERROR] 配置 {config_name} 失败: {e}")
            results.append({
                'config': config_name,
                'mae': float('nan'),
                'mse': float('nan'),
                'rmse': float('nan'),
                'mape': float('nan'),
                'error': str(e),
            })
    
    # 汇总表格
    df = pd.DataFrame(results)
    df = df.sort_values('mae', ascending=True)
    
    # 保存
    results_dir = ABLATION_DIR / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(results_dir / f"ablation_summary_{timestamp}.csv", index=False)
    
    print(f"\n{'='*70}")
    print("消融实验汇总")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    
    return df


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TPLC 模型消融实验")
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        help=f"实验配置: {list(ABLATION_CONFIGS.keys())} 或 'all'",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    
    args = parser.parse_args()
    
    if args.config == "all":
        run_all_configs()
    else:
        # 支持 + 开头的配置名
        config_name = args.config
        if config_name not in ABLATION_CONFIGS:
            # 尝试添加 + 前缀
            if f"+{config_name}" in ABLATION_CONFIGS:
                config_name = f"+{config_name}"
        
        if config_name not in ABLATION_CONFIGS:
            print(f"未知配置: {args.config}")
            print(f"可用配置: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
        
        # 覆盖参数
        cfg = ABLATION_CONFIGS[config_name]
        if args.epochs is not None:
            cfg.epochs = args.epochs
        if args.seed is not None:
            cfg.seed = args.seed
        
        runner = AblationRunner(cfg)
        runner.run()


if __name__ == "__main__":
    main()
