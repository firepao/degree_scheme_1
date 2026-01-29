"""TPLC 消融模型 - 支持对核心组件进行开关控制。

消融实验设计:
1. FFT 周期提取 (use_fft): 使用 FFT 动态提取周期 vs 固定周期
2. 1D→2D 重塑 (use_reshape_2d): 将 1D 序列重塑为 2D 图像处理 vs 纯 1D 处理
3. 深度可分离卷积 (use_depthwise): 使用深度可分离卷积 vs 标准卷积
4. 多尺度处理 (use_multi_scale): 多尺度并行处理 vs 单尺度
5. 多周期融合 (use_multi_period): 多个周期融合 vs 单个周期
6. 振幅加权融合 (use_amp_weight): 使用 FFT 振幅作为周期权重 vs 非振幅权重
7. 多尺度融合 (use_scale_weight): 等权求和 vs 可学习权重
8. 残差连接 (use_residual): 残差连接 vs 无残差
9. RevIN 归一化 (use_revin): 使用 RevIN 归一化 vs 不使用
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from components import (
    extract_topk_periods,
    PeriodInfo,
    reshape_1d_to_2d,
    reshape_2d_to_1d,
    DepthwiseSeparableConv2d,
    StandardConv2d,
    MultiScaleGenerator,
    RevIN,
    MLPPredictor,
    Conv1DPredictor,
)


@dataclass
class AblationConfig:
    """消融实验配置。"""
    
    # 核心消融开关
    use_fft: bool = True           # 使用 FFT 动态周期提取
    use_reshape_2d: bool = True    # 使用 1D→2D 重塑
    use_depthwise: bool = True     # 使用深度可分离卷积
    use_multi_scale: bool = True   # 使用多尺度
    use_multi_period: bool = True  # 使用多周期融合
    use_amp_weight: bool = True    # 使用 FFT 振幅作为周期权重
    use_scale_weight: bool = False # 多尺度融合是否使用可学习权重
    use_residual: bool = True      # 使用残差连接
    use_revin: bool = True         # 使用 RevIN 归一化
    
    # 默认固定周期（当 use_fft=False 时使用）
    fixed_periods: List[int] = field(default_factory=lambda: [24, 12, 8])
    
    # 模型超参数
    hidden_dim: int = 32
    num_scales: int = 2
    top_k_periods: int = 3
    dropout: float = 0.1
    
    @classmethod
    def full(cls) -> "AblationConfig":
        """完整模型配置。"""
        return cls()
    
    @classmethod
    def no_fft(cls) -> "AblationConfig":
        """不使用 FFT 周期提取。"""
        return cls(use_fft=False)
    
    @classmethod
    def no_reshape_2d(cls) -> "AblationConfig":
        """不使用 1D→2D 重塑。"""
        return cls(use_reshape_2d=False)
    
    @classmethod
    def no_depthwise(cls) -> "AblationConfig":
        """不使用深度可分离卷积。"""
        return cls(use_depthwise=False)
    
    @classmethod
    def no_multi_scale(cls) -> "AblationConfig":
        """不使用多尺度处理。"""
        return cls(use_multi_scale=False, num_scales=0)
    
    @classmethod
    def no_multi_period(cls) -> "AblationConfig":
        """不使用多周期融合。"""
        return cls(use_multi_period=False, top_k_periods=1)

    @classmethod
    def no_amp_weight(cls) -> "AblationConfig":
        """不使用 FFT 振幅权重进行多周期融合。"""
        return cls(use_amp_weight=False)
    
    @classmethod
    def no_residual(cls) -> "AblationConfig":
        """不使用残差连接。"""
        return cls(use_residual=False)
    
    @classmethod
    def no_revin(cls) -> "AblationConfig":
        """不使用 RevIN 归一化。"""
        return cls(use_revin=False)
    
    @classmethod
    def baseline_mlp(cls) -> "AblationConfig":
        """基线 MLP 模型（所有创新模块关闭）。"""
        return cls(
            use_fft=False,
            use_reshape_2d=False,
            use_depthwise=False,
            use_multi_scale=False,
            use_multi_period=False,
            use_residual=False,
            use_revin=False,
        )


class TPLCNet_Ablation(nn.Module):
    """TPLC 消融模型 - 支持对各个核心组件进行独立开关。
    
    核心消融组件:
    1. FFT 周期提取: 动态发现数据中的主要周期
    2. 1D→2D 重塑: 将时序数据按周期展开为 2D 图像
    3. 深度可分离卷积: 比标准卷积参数更少、效率更高
    4. 多尺度处理: 在不同时间粒度上捕获模式
    5. 多周期融合: 结合多个周期的信息
    6. 残差连接: 促进梯度流动、加速收敛
    7. RevIN 归一化: 处理时序数据的分布偏移
    """
    
    def __init__(
        self,
        input_dim: int,
        target_dim: int,
        seq_len: int,
        pred_len: int,
        config: Optional[AblationConfig] = None,
    ):
        super().__init__()
        
        self.config = config or AblationConfig.full()
        cfg = self.config
        
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_dim = cfg.hidden_dim
        
        # ---- 特殊情况: 纯 MLP 基线 ----
        if self._is_mlp_baseline():
            self.mlp_predictor = MLPPredictor(
                input_dim=input_dim,
                target_dim=target_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                hidden_dim=cfg.hidden_dim * 2,
            )
            return
        
        # ---- 特殊情况: 不使用 2D 重塑 (纯 1D 卷积) ----
        if not cfg.use_reshape_2d:
            self.conv1d_predictor = Conv1DPredictor(
                input_dim=input_dim,
                target_dim=target_dim,
                seq_len=seq_len,
                pred_len=pred_len,
                hidden_dim=cfg.hidden_dim,
            )
            if cfg.use_revin:
                self.revin = RevIN(target_dim)
            if cfg.use_residual:
                self.residual_proj = nn.Linear(seq_len, pred_len)
            return
        
        # ---- 标准 TPLC 架构 ----
        
        # RevIN 归一化
        if cfg.use_revin:
            self.revin = RevIN(target_dim)
        
        # 多尺度生成器
        if cfg.use_multi_scale and cfg.num_scales > 0:
            self.multi_scale = MultiScaleGenerator(num_scales=cfg.num_scales)
            self.num_scales = cfg.num_scales + 1  # 包含原始尺度
        else:
            self.num_scales = 1
        
        # 2D 卷积（根据配置选择深度可分离 or 标准卷积）
        if cfg.use_depthwise:
            self.conv2d = DepthwiseSeparableConv2d(
                in_channels=input_dim,
                out_channels=cfg.hidden_dim,
                kernel_size=3,
            )
        else:
            self.conv2d = StandardConv2d(
                in_channels=input_dim,
                out_channels=cfg.hidden_dim,
                kernel_size=3,
            )
        
        # 周期数
        self.top_k = cfg.top_k_periods if cfg.use_multi_period else 1
        
        # 时间投影（每个尺度有多个周期）
        self.time_projs = nn.ModuleDict()
        for s in range(self.num_scales):
            scale_len = seq_len // (2 ** s)
            for k in range(self.top_k):
                key = f"s{s}_p{k}"
                self.time_projs[key] = nn.Linear(scale_len, pred_len)
        
        # 通道投影
        self.channel_proj = nn.Linear(cfg.hidden_dim, target_dim)
        
        # 周期融合权重（仅在不使用振幅权重时启用）
        if cfg.use_multi_period and self.top_k > 1 and not cfg.use_amp_weight:
            self.period_logits = nn.Parameter(torch.zeros(self.top_k))

        # 尺度融合权重（可选）
        if cfg.use_multi_scale and self.num_scales > 1 and cfg.use_scale_weight:
            self.scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        
        # Dropout
        self.dropout = nn.Dropout(cfg.dropout)
        
        # 残差连接
        if cfg.use_residual:
            self.residual_proj = nn.Linear(seq_len, pred_len)
    
    def _is_mlp_baseline(self) -> bool:
        """检查是否为纯 MLP 基线配置。"""
        cfg = self.config
        return (
            not cfg.use_fft
            and not cfg.use_reshape_2d
            and not cfg.use_depthwise
            and not cfg.use_multi_scale
            and not cfg.use_multi_period
        )
    
    def _get_period_info(self, x: torch.Tensor) -> PeriodInfo:
        """获取周期与振幅信息（FFT 提取或固定周期）。"""
        cfg = self.config
        
        if cfg.use_fft:
            # 动态 FFT 提取
            period_info = extract_topk_periods(x, top_k=self.top_k)
        else:
            # 使用固定周期，并用均匀振幅占位
            periods = cfg.fixed_periods[:self.top_k]
            while len(periods) < self.top_k:
                periods.append(periods[-1] if periods else 12)
            period_info = PeriodInfo(
                periods=torch.tensor(periods, dtype=torch.long, device=x.device),
                amplitudes=torch.ones(self.top_k, dtype=torch.float32, device=x.device),
            )

        return period_info
    
    def _process_single_scale(
        self,
        x: torch.Tensor,
        scale_idx: int,
        period_info: PeriodInfo,
    ) -> torch.Tensor:
        """处理单个尺度的数据。
        
        Args:
            x: [B, C, L] 输入
            scale_idx: 尺度索引
            periods: 周期列表
        Returns:
            [B, pred_len, target_dim] 该尺度的预测
        """
        cfg = self.config
        b, c, l = x.shape
        
        period_outputs = []
        
        periods = period_info.periods.tolist()

        for k, period in enumerate(periods):
            # 1D → 2D 重塑
            z, orig_len = reshape_1d_to_2d(x, period)  # [B, C, rows, period]
            
            # 2D 卷积
            z = self.conv2d(z)  # [B, hidden_dim, rows, period]
            z = self.dropout(z)
            
            # 2D → 1D 还原
            out = reshape_2d_to_1d(z, orig_len)  # [B, hidden_dim, L]
            
            # 时间投影
            time_proj = self.time_projs[f"s{scale_idx}_p{k}"]
            out = time_proj(out)  # [B, hidden_dim, pred_len]
            out = out.transpose(1, 2)  # [B, pred_len, hidden_dim]
            
            # 通道投影
            out = self.channel_proj(out)  # [B, pred_len, target_dim]
            
            period_outputs.append(out)
        
        # 周期融合
        if cfg.use_multi_period and len(period_outputs) > 1:
            if cfg.use_amp_weight and cfg.use_fft:
                period_weights = F.softmax(period_info.amplitudes, dim=0)
            elif hasattr(self, "period_logits"):
                period_weights = F.softmax(self.period_logits, dim=0)
            else:
                period_weights = torch.full(
                    (len(period_outputs),),
                    1.0 / len(period_outputs),
                    device=period_outputs[0].device,
                )
            scale_out = sum(w * o for w, o in zip(period_weights, period_outputs))
        else:
            scale_out = period_outputs[0]
        
        return scale_out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [B, seq_len, input_dim] 输入序列
        Returns:
            [B, pred_len, target_dim] 预测序列
        """
        cfg = self.config
        
        # ---- 特殊情况: MLP 基线 ----
        if self._is_mlp_baseline():
            return self.mlp_predictor(x)
        
        # ---- RevIN 归一化 ----
        if cfg.use_revin:
            # 只对目标维度做归一化
            x_target = x[..., :self.target_dim]
            x_target = self.revin(x_target, mode='norm')
            x = torch.cat([x_target, x[..., self.target_dim:]], dim=-1)
        
        # ---- 特殊情况: 纯 1D 卷积 ----
        if not cfg.use_reshape_2d:
            out = self.conv1d_predictor(x)
            if cfg.use_residual:
                # 残差连接
                x_res = x[..., :self.target_dim].transpose(1, 2)
                res = self.residual_proj(x_res).transpose(1, 2)
                out = out + res
            if cfg.use_revin:
                out = self.revin(out, mode='denorm')
            return out
        
        # ---- 标准 TPLC 流程 ----
        
        # 保存残差
        if cfg.use_residual:
            x_res = x[..., :self.target_dim].clone()
        
        # [B, L, C] → [B, C, L]
        x = x.transpose(1, 2)
        
        # 提取周期
        period_info = self._get_period_info(x)
        
        # 生成多尺度
        if cfg.use_multi_scale and self.num_scales > 1:
            scales = self.multi_scale(x)
        else:
            scales = [x]
        
        # 处理每个尺度
        scale_outputs = []
        for s, scale_x in enumerate(scales):
            # 调整周期（对于下采样的尺度）
            scale_periods = [max(1, p // (2 ** s)) for p in period_info.periods.tolist()]
            scale_period_info = PeriodInfo(
                periods=torch.tensor(scale_periods, dtype=torch.long, device=x.device),
                amplitudes=period_info.amplitudes,
            )
            scale_out = self._process_single_scale(scale_x, s, scale_period_info)
            scale_outputs.append(scale_out)
        
        # 尺度融合
        if cfg.use_multi_scale and len(scale_outputs) > 1:
            if cfg.use_scale_weight and hasattr(self, "scale_logits"):
                scale_weights = F.softmax(self.scale_logits, dim=0)
                out = sum(w * o for w, o in zip(scale_weights, scale_outputs))
            else:
                out = sum(scale_outputs)
        else:
            out = scale_outputs[0]
        
        # 残差连接
        if cfg.use_residual:
            res = self.residual_proj(x_res.transpose(1, 2)).transpose(1, 2)
            out = out + res
        
        # RevIN 反归一化
        if cfg.use_revin:
            out = self.revin(out, mode='denorm')
        
        return out


# ============================================================
# 便捷工厂函数
# ============================================================

def create_ablation_model(
    input_dim: int,
    target_dim: int,
    seq_len: int,
    pred_len: int,
    ablation_name: str,
) -> TPLCNet_Ablation:
    """根据消融实验名称创建模型。
    
    Args:
        input_dim: 输入特征维度
        target_dim: 输出目标维度
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        ablation_name: 消融实验名称
            - "full": 完整模型
            - "no_fft": 不使用 FFT 周期提取
            - "no_reshape_2d": 不使用 1D→2D 重塑
            - "no_depthwise": 不使用深度可分离卷积
            - "no_multi_scale": 不使用多尺度
            - "no_multi_period": 不使用多周期融合
            - "no_amp_weight": 不使用 FFT 振幅权重
            - "no_residual": 不使用残差连接
            - "no_revin": 不使用 RevIN 归一化
            - "baseline_mlp": MLP 基线
    
    Returns:
        配置好的消融模型
    """
    config_map = {
        "full": AblationConfig.full,
        "no_fft": AblationConfig.no_fft,
        "no_reshape_2d": AblationConfig.no_reshape_2d,
        "no_depthwise": AblationConfig.no_depthwise,
        "no_multi_scale": AblationConfig.no_multi_scale,
        "no_multi_period": AblationConfig.no_multi_period,
        "no_amp_weight": AblationConfig.no_amp_weight,
        "no_residual": AblationConfig.no_residual,
        "no_revin": AblationConfig.no_revin,
        "baseline_mlp": AblationConfig.baseline_mlp,
    }
    
    if ablation_name not in config_map:
        raise ValueError(
            f"未知的消融实验名称: {ablation_name}. "
            f"可选: {list(config_map.keys())}"
        )
    
    config = config_map[ablation_name]()
    
    return TPLCNet_Ablation(
        input_dim=input_dim,
        target_dim=target_dim,
        seq_len=seq_len,
        pred_len=pred_len,
        config=config,
    )
