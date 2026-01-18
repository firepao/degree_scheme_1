"""TimeMixer 模型核心实现。

基于官方实现简化：专注于预测任务，适配温室数据。

参考：
- 论文：TimeMixer (ICLR 2024)
- 代码：https://github.com/thuml/Time-Series-Library
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 支持相对导入和绝对导入
try:
    from .layers import DataEmbedding_wo_pos, Normalize, series_decomp
except ImportError:
    # 尝试显式从当前包导入（解决 script 模式运行时的相对导入问题）
    try:
        from TimeMixer.layers import DataEmbedding_wo_pos, Normalize, series_decomp
    except ImportError:
         # 最后尝试直接导入（注意：可能与 TSLib layers 冲突）
        from layers import DataEmbedding_wo_pos, Normalize, series_decomp


# ========== DFT 分解 ==========

class DFT_series_decomp(nn.Module):
    """基于 DFT 的序列分解。
    
    保留 top-k 个频率成分作为季节性，其余作为趋势性。
    """
    
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, C] 输入序列
        
        Returns:
            x_season: [B, T, C] 季节性成分
            x_trend: [B, T, C] 趋势性成分
        """
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[:, 0, :] = 0  # 去除直流分量
        
        # 找到振幅最大的 top_k 个频率
        top_k_freq, top_list = torch.topk(freq, k=self.top_k, dim=1)
        
        # 过滤掉低振幅频率
        xf[freq <= top_k_freq.min(dim=1, keepdim=True)[0]] = 0
        
        x_season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        x_trend = x - x_season
        
        return x_season, x_trend


# ========== 多尺度混合模块 ==========

class MultiScaleSeasonMixing(nn.Module):
    """自底向上的季节性混合。
    
    从高分辨率到低分辨率逐层混合季节性模式。
    """
    
    def __init__(
        self,
        seq_len: int,
        down_sampling_window: int,
        down_sampling_layers: int,
    ):
        super().__init__()
        
        self.down_sampling_layers_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** (i + 1)),
                ),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            season_list: 多尺度季节性列表 [x_0, x_1, ..., x_n]
                每个 x_i: [B, C, T_i]
        
        Returns:
            out_season_list: 混合后的多尺度季节性 [y_0, y_1, ..., y_n]
        """
        # 从高到低混合
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]  # [B, T, C]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers_list[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """自顶向下的趋势性混合。
    
    从低分辨率到高分辨率逐层混合趋势性模式。
    """
    
    def __init__(
        self,
        seq_len: int,
        down_sampling_window: int,
        down_sampling_layers: int,
    ):
        super().__init__()
        
        self.up_sampling_layers_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    seq_len // (down_sampling_window ** (i + 1)),
                    seq_len // (down_sampling_window ** i),
                ),
                nn.GELU(),
                nn.Linear(
                    seq_len // (down_sampling_window ** i),
                    seq_len // (down_sampling_window ** i),
                ),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            trend_list: 多尺度趋势性列表 [x_0, x_1, ..., x_n]
        
        Returns:
            out_trend_list: 混合后的多尺度趋势性 [y_0, y_1, ..., y_n]
        """
        # 从低到高混合
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers_list[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


# ========== 过去可分解混合（PDM）==========

class PastDecomposableMixing(nn.Module):
    """过去可分解混合模块（PDM Block）。
    
    核心思想：
    1. 将多尺度序列分解为季节性和趋势性
    2. 分别进行多尺度混合
    3. 融合后输出
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        d_model: int,
        d_ff: int,
        down_sampling_window: int,
        down_sampling_layers: int,
        dropout: float,
        channel_independence: bool,
        decomp_method: str,
        moving_avg: int,
        top_k: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.channel_independence = channel_independence
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 分解方法
        if decomp_method == 'moving_avg':
            self.decomposition = series_decomp(moving_avg)
        elif decomp_method == 'dft_decomp':
            self.decomposition = DFT_series_decomp(top_k)
        else:
            raise ValueError(f'未知分解方法: {decomp_method}')
        
        # 跨通道层（通道混合模式）
        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )
        
        # 多尺度季节性混合
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
        )
        
        # 多尺度趋势性混合
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len=seq_len,
            down_sampling_window=down_sampling_window,
            down_sampling_layers=down_sampling_layers,
        )
        
        # 输出跨通道层（通道独立模式也需要）
        if channel_independence:
            # 通道独立模式：d_model -> d_model
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )
        else:
            # 通道混合模式：d_model -> d_model
            self.out_cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

    def forward(self, x_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            x_list: 多尺度输入列表，每个 [B, T_i, d_model]
        
        Returns:
            out_list: 多尺度输出列表，每个 [B, T_i, d_model]
        """
        length_list = [x.size(1) for x in x_list]
        
        # 1. 分解：季节性 + 趋势性
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decomposition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))  # [B, C, T]
            trend_list.append(trend.permute(0, 2, 1))
        
        # 2. 多尺度混合
        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)
        
        # 3. 融合 + 残差
        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            # 应用输出跨通道层并添加残差连接
            out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        
        return out_list


# ========== TimeMixer 主模型 ==========

@dataclass
class TimeMixerConfig:
    """TimeMixer 配置。"""
    seq_len: int = 288
    pred_len: int = 72
    enc_in: int = 10  # 输入特征数
    c_out: int = 4    # 输出特征数
    d_model: int = 64
    d_ff: int = 128
    e_layers: int = 2
    dropout: float = 0.1
    
    # 多尺度配置
    down_sampling_layers: int = 2
    down_sampling_window: int = 2
    down_sampling_method: str = 'avg'  # 'avg', 'max', 'conv'
    
    # 分解配置
    decomp_method: str = 'moving_avg'  # 'moving_avg', 'dft_decomp'
    moving_avg: int = 25
    top_k: int = 5
    
    # 通道配置
    channel_independence: bool = True
    
    # 嵌入配置
    embed: str = 'timeF'
    freq: str = 'h'
    use_norm: int = 1


class TimeMixerForecaster(nn.Module):
    """TimeMixer 预测器（简化版）。
    
    专注于长期/短期预测任务。
    """
    
    def __init__(self, config: TimeMixerConfig):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.c_out = config.c_out
        self.channel_independence = config.channel_independence
        self.down_sampling_window = config.down_sampling_window
        
        # 预处理：序列分解
        self.preprocess = series_decomp(config.moving_avg)
        
        # 嵌入层
        if config.channel_independence:
            self.enc_embedding = DataEmbedding_wo_pos(
                c_in=1,
                d_model=config.d_model,
                embed_type=config.embed,
                freq=config.freq,
                dropout=config.dropout,
            )
        else:
            self.enc_embedding = DataEmbedding_wo_pos(
                c_in=config.enc_in,
                d_model=config.d_model,
                embed_type=config.embed,
                freq=config.freq,
                dropout=config.dropout,
            )
        
        # PDM Blocks（编码器）
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=config.seq_len,
                pred_len=config.pred_len,
                d_model=config.d_model,
                d_ff=config.d_ff,
                down_sampling_window=config.down_sampling_window,
                down_sampling_layers=config.down_sampling_layers,
                dropout=config.dropout,
                channel_independence=config.channel_independence,
                decomp_method=config.decomp_method,
                moving_avg=config.moving_avg,
                top_k=config.top_k,
            )
            for _ in range(config.e_layers)
        ])
        
        # 归一化层（每个尺度一个）
        self.normalize_layers = nn.ModuleList([
            Normalize(
                config.enc_in,
                affine=True,
                non_norm=True if config.use_norm == 0 else False,
            )
            for _ in range(config.down_sampling_layers + 1)
        ])
        
        # 如果通道独立且输入输出维度不同，需要额外的归一化层用于输出
        if config.channel_independence and config.enc_in != config.c_out:
            self.output_normalize_layer = Normalize(
                config.c_out,
                affine=True,
                non_norm=True if config.use_norm == 0 else False,
            )
        else:
            self.output_normalize_layer = None
        
        # 预测头（每个尺度一个）
        self.predict_layers = nn.ModuleList([
            nn.Linear(
                config.seq_len // (config.down_sampling_window ** i),
                config.pred_len,
            )
            for i in range(config.down_sampling_layers + 1)
        ])
        
        # 投影层
        if config.channel_independence:
            self.projection_layer = nn.Linear(config.d_model, 1, bias=True)
            # 如果输入和输出维度不同，需要额外的投影层
            if config.enc_in != config.c_out:
                self.channel_projection = nn.Linear(config.enc_in, config.c_out, bias=True)
            else:
                self.channel_projection = None
        else:
            self.projection_layer = nn.Linear(config.d_model, config.c_out, bias=True)
            self.channel_projection = None
            
            # 修复：如果有输入输出维度差异，增加残差投影层
            if config.enc_in != config.c_out:
                self.residual_channel_projection = nn.Linear(config.enc_in, config.c_out, bias=True)
            else:
                self.residual_channel_projection = None

            # 额外的残差和回归层
            self.out_res_layers = nn.ModuleList([
                nn.Linear(
                    config.seq_len // (config.down_sampling_window ** i),
                    config.seq_len // (config.down_sampling_window ** i),
                )
                for i in range(config.down_sampling_layers + 1)
            ])
            
            self.regression_layers = nn.ModuleList([
                nn.Linear(
                    config.seq_len // (config.down_sampling_window ** i),
                    config.pred_len,
                )
                for i in range(config.down_sampling_layers + 1)
            ])

    def _multi_scale_process_inputs(
        self,
        x_enc: torch.Tensor,
    ) -> list[torch.Tensor]:
        """多尺度下采样。
        
        Args:
            x_enc: [B, T, C] 输入序列
        
        Returns:
            x_enc_list: 多尺度序列列表
        """
        # 选择下采样方法
        if self.config.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.down_sampling_window)
        elif self.config.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.config.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False,
            )
        else:
            return [x_enc]
        
        # 转换维度：[B, T, C] -> [B, C, T]
        x_enc = x_enc.permute(0, 2, 1)
        
        x_enc_ori = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]  # 添加原始尺度
        
        # 逐层下采样
        for i in range(self.config.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        
        return x_enc_sampling_list

    def _pre_enc(
        self,
        x_list: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """预编码：序列分解。"""
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def _out_projection(
        self,
        dec_out: torch.Tensor,
        i: int,
        out_res: torch.Tensor
    ) -> torch.Tensor:
        """输出投影（通道混合模式）。"""
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        
        # 修复：维度对齐
        if hasattr(self, 'residual_channel_projection') and self.residual_channel_projection is not None:
             out_res = self.residual_channel_projection(out_res)
             
        dec_out = dec_out + out_res
        return dec_out

    def _future_multi_mixing(
        self,
        B: int,
        enc_out_list: list[torch.Tensor],
        x_list: tuple[list[torch.Tensor], list[torch.Tensor] | None]
    ) -> list[torch.Tensor]:
        """未来多预测器混合（解码器）。"""
        dec_out_list = []
        
        if self.channel_independence:
            x_list_0 = x_list[0]
            for i, enc_out in enumerate(enc_out_list):
                # 时间维映射：T -> pred_len
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                # 通道维映射：d_model -> 1
                dec_out = self.projection_layer(dec_out)
                # 重塑：[B*enc_in, pred, 1] -> [B, pred, enc_in]
                dec_out = dec_out.reshape(B, self.enc_in, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)
        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec_out = self._out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)
        
        return dec_out_list

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: [B, seq_len, enc_in] 输入序列
        
        Returns:
            y: [B, pred_len, c_out] 预测序列
        """
        # 1. 多尺度下采样
        x_enc_list = self._multi_scale_process_inputs(x)
        
        # 2. 归一化 + 通道独立处理
        x_list = []
        for i, x_enc in enumerate(x_enc_list):
            B, T, N = x_enc.size()
            x_enc = self.normalize_layers[i](x_enc, 'norm')
            if self.channel_independence:
                # 将每个通道视为独立样本：[B, T, N] -> [B*N, T, 1]
                x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x_enc)
        
        # 3. 嵌入
        enc_out_list = []
        x_list = self._pre_enc(x_list)
        for x in x_list[0]:
            enc_out = self.enc_embedding(x, None)  # [B*N, T, d_model] 或 [B, T, d_model]
            enc_out_list.append(enc_out)
        
        # 4. PDM Blocks（编码器）
        for pdm_block in self.pdm_blocks:
            enc_out_list = pdm_block(enc_out_list)
        
        # 5. 未来多预测器混合（解码器）
        dec_out_list = self._future_multi_mixing(B, enc_out_list, x_list)
        
        # 6. 多尺度聚合
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        
        # 7. 反归一化（在通道投影前，使用输入维度的归一化层）
        # 修复：仅当维度匹配或有专门的输出归一化层时才执行反归一化
        if self.enc_in == self.c_out:
            dec_out = self.normalize_layers[0](dec_out, 'denorm')
        elif self.channel_independence and self.output_normalize_layer is not None:
             dec_out = self.output_normalize_layer(dec_out, 'denorm')
        
        # 8. 通道投影（如果需要）
        if self.channel_projection is not None:
            dec_out = self.channel_projection(dec_out)  # [B, pred, enc_in] -> [B, pred, c_out]
        
        return dec_out
