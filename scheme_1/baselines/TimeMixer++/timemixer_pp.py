from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 尝试导入辅助层，如果失败则假设在同级目录或路径已设置
try:
    from .layers import DataEmbedding_wo_pos, Normalize
except ImportError:
    try:
        from layers import DataEmbedding_wo_pos, Normalize
    except ImportError:
        # Fallback if layers.py is not found, define simple dummies or rely on user to fix path
        pass

# ========== 配置类 ==========

@dataclass
class TimeMixerPPConfig:
    seq_len: int = 96           # 输入序列长度
    pred_len: int = 96          # 预测序列长度
    enc_in: int = 7             # 输入特征数
    c_out: int = 7              # 输出特征数
    d_model: int = 16           # 嵌入维度
    d_ff: int = 32              # 前馈维度
    e_layers: int = 2           # MixerBlock 层数
    num_scales: int = 3         # 多尺度数量
    top_k: int = 3              # MRTI/MRM 中的 Top-K 频率
    dropout: float = 0.1
    down_sampling_window: int = 2
    down_sampling_layers: int = 2  # 通常 = num_scales - 1
    channel_independence: bool = False # 是否通道独立
    num_kernels: int = 6        # Inception 卷积核数

# ========== 核心组件 ==========

class Inception_Block_V1(nn.Module):
    """TimesNet 中的 Inception Block V1，用于 2D 卷积混合。"""
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class ChannelMixing(nn.Module):
    """通道混合模块：在最粗尺度上应用变量间自注意力。"""
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        # 注意：这里假设 d_model 包含了通道信息(Channel Mixing across features)
        # 或者如果是 iTransformer 风格，x 应该是 [B, N_vars, D]。
        # 由于 TimeMixer 主要是 [B, T, D]，这里的 ChannelMixing 理解为特征维度的混合
        # 或者我们将其视为 Temporal Self-Attention (如 Transformer)
        # 论文描述 "Variate-wise self-attention" -> 变量间。
        # 如果架构是 Channel Independent，则 x 是 [B*C, T, 1] 或 [B*C, T, D']。
        # 此时无法做 Variate-wise。
        # 假设此模块仅在 Channel Dependence (mixing) 模式下启用。
        
        # 简单实现为标准 Self-Attention，捕捉全局依赖
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class MultiResolutionTimeImaging(nn.Module):
    """MRTI: 将 1D 序列转换为多分辨率 2D 图像。"""
    def __init__(self, top_k: int = 3):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[int], List[torch.Tensor]]:
        """
        Args:
           x: [B, T, D]
        Returns:
           images: List of [B, D, F, P] (2D images) for each k
           periods: List of periods
           amplitudes: Top-k amplitudes (for MRM weights)
        """
        B, T, D = x.shape
        xf = torch.fft.rfft(x, dim=1)
        # 计算平均振幅以找周期
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.top_k)
        top_list = top_list.detach().cpu().numpy()
        
        periods = []
        images = []
        # 保存振幅用于 MRM 加权: [B, K, D] -> 需对齐
        # 这里提取 top-k 频率对应的振幅
        # xf: [B, L//2+1, D]
        # select top_k indices
        amp = abs(xf) # [B, L//2+1, D]
        # top_list is indices of freq.
        # We need amplitudes for each k.
        
        # 修正: top_list 是全局平均的 top频率索引
        # 我们收集每个 k 对应的 2D 图像
        
        # Amplitudes for MRM: [B, D, K] (or similar to weight the K images)
        # Pick amplitudes at top_list frequencies
        # amp[:, top_list, :] -> [B, K, D]
        top_amps = amp[:, top_list, :] # [B, K, D]

        for i in range(self.top_k):
            period = T // top_list[i]
            periods.append(period)
            
            # Padding
            if T % period != 0:
                length = ((T // period) + 1) * period
                padding = torch.zeros([B, (length - T), D], device=x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = T
                out = x
                
            # Reshape: [B, Length, D] -> [B, F, P, D] -> [B, D, F, P]
            # F = Length // P
            out = out.reshape(B, length // period, period, D).permute(0, 3, 1, 2).contiguous()
            images.append(out)
            
        return images, periods, top_amps

class TimeImageDecomposition(nn.Module):
    """TID: 双轴注意力分解季节性和趋势性。"""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        # Column Attention (Seasonality) - Attention over Height (F)
        # Row Attention (Trend) - Attention over Width (P)
        # 实际上：
        # Image: [B, D, F, P]
        # Seasonality (F axis? or P axis?):
        # 季节性通常是周期内的模式 (P 轴)。
        # 趋势性是周期周期间的变化 (F 轴)。
        # 论文定义: 
        # TID applies dual-axis attention to disentangle seasonal and trend.
        # Check TimesNet: 2D Conv works on (F, P).
        # We assume one axis captures intra-period (Season) and other inter-period (Trend).
        
        # 为了简化，我们使用标准 MultiheadAttention
        # 并假设 Last Dim 是要 Attention 的维度
        
        self.col_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.row_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True, dropout=dropout)
        self.norm_s = nn.LayerNorm(d_model) # Applied on permuted
        self.norm_t = nn.LayerNorm(d_model)

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image: [B, D, F, P]
        Returns:
            season: [B, D, F, P]
            trend: [B, D, F, P]
        """
        B, D, F, P = image.shape
        
        # Seasonality: Attention along P (columns of the period view?) or F?
        # Typically P is the "Time step in a period".
        # Let's apply Attention over P.
        # Reshape to [B*F, P, D] ? No, D is feature.
        # Attention expects [Batch, Seq, Embed].
        # If we fix D as Embed: [B*F, P, D] -> Attn over P -> Seasonality
        
        # Seasonality (Intra-period): [B*F, P, D]
        # But wait, image is [B, D, F, P].
        # Permute to [B, F, P, D]. Flatten B*F -> [B*F, P, D].
        x_p = image.permute(0, 2, 3, 1).reshape(B*F, P, D)
        s, _ = self.col_attn(x_p, x_p, x_p)
        s = self.norm_s(x_p + s)
        season = s.reshape(B, F, P, D).permute(0, 3, 1, 2) # Back to [B, D, F, P]
        
        # Trend (Inter-period): [B*P, F, D]
        # We want to see how things change across periods (F axis).
        x_f = image.permute(0, 3, 2, 1).reshape(B*P, F, D)
        t, _ = self.row_attn(x_f, x_f, x_f)
        t = self.norm_t(x_f + t)
        trend = t.reshape(B, P, F, D).permute(0, 3, 2, 1) # Back to [B, D, F, P]
        
        return season, trend

class MultiScaleMixing(nn.Module):
    """MCM: 多尺度混合 (2D Conv / 2D TransConv)。"""
    def __init__(self, d_model: int, d_ff: int, num_kernels: int = 6):
        super().__init__()
        # Seasonality Mixing: 2D Conv (Inception)
        self.season_mixer = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )
        # Trend Mixing: 2D Transposed Conv?
        # For simplicity and robustness (as shapes vary), we use Inception for Trend too
        # unless strict TransConv is required for upsizing.
        # But here inputs are already same scale images (or handled externally).
        # In TimeMixer++, MCM aggregates across scales.
        # If we just process the image, we treat it as "intra-scale mixing"
        # and rely on the architecture to sum them up?
        # 论文: "MCM hierarchically aggregates these patterns across scales."
        # This implies inputs from other scales are added.
        # Due to complexity of matching resolutions of different scales in 2D,
        # we implement the "MixerBlock" to handle the `s = s + Conv(s_prev)` logic.
        # This module will just be the Learnable Conv Layer.
        
        self.trend_mixer = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, season: torch.Tensor, trend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s_out = self.season_mixer(season)
        t_out = self.trend_mixer(trend)
        return s_out, t_out

class MultiResolutionMixing(nn.Module):
    """MRM: 多分辨率自适应融合。"""
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=1) # Dim 1 is K (resolutions)

    def forward(self, images_list: list[torch.Tensor], amps: torch.Tensor, origin_len: int) -> torch.Tensor:
        """
        Args:
            images_list: List of K images [B, D, F_k, P_k]
            amps: [B, K, D] (Amplitudes for weighting)
            origin_len: Original sequence length T
        Returns:
            Combined sequence [B, T, D]
        """
        # 1. Reshape images back to sequences
        # 2. Weighted Sum
        
        B, K, D = amps.shape
        weights = self.softmax(amps).unsqueeze(-1) # [B, K, D, 1] for broadcasting?
        # Sequence reconstruction
        
        preds = []
        for k, img in enumerate(images_list):
            # img: [B, D, F, P] -> [B, D, F*P] -> [B, F*P, D]
            # Be careful with padding removal
            B, D, F, P = img.shape
            seq = img.permute(0, 2, 3, 1).reshape(B, -1, D)
            seq = seq[:, :origin_len, :]
            preds.append(seq)
            
        stack_preds = torch.stack(preds, dim=1) # [B, K, T, D]
        
        # Weights: [B, K, D]. Expand to [B, K, T, D]
        w = weights.permute(0, 1, 3, 2).repeat(1, 1, origin_len, 1) # [B, K, T, D]
        
        # Weighted sum dim 1
        return torch.sum(stack_preds * w, dim=1)


class MixerBlock(nn.Module):
    """TimeMixer++ Block: MRTI -> TID -> MCM -> MRM."""
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        self.mrti = MultiResolutionTimeImaging(top_k=config.top_k)
        self.tid = TimeImageDecomposition(d_model=config.d_model, dropout=config.dropout)
        self.mcm = MultiScaleMixing(d_model=config.d_model, d_ff=config.d_ff, num_kernels=config.num_kernels)
        self.mrm = MultiResolutionMixing()
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        
        # 1. MRTI
        images, periods, amps = self.mrti(x)
        
        # 2. TID & 3. MCM (Applied on each resolution image)
        # Note: MCM in paper aggregates across scales. 
        # But here we are inside one block processing one scale input?
        # Or 'x' is one scale.
        # Paper Fig 2: Input is 'Multi-scale Time Series'. 
        # MixerBlock processes them.
        # If we simplify: apply TID and Conv mixing on the images derived from x
        
        processed_images = []
        for img in images:
            # img: [B, D, F, P]
            season, trend = self.tid(img)
            
            # Here we just apply the convolution mixing on S and T
            # For full MCM (cross-scale), we'd need inputs from other scales.
            # Assuming simplified "Intra-scale" mixing for this standalone block 
            # or that x already contains scale info.
            
            s_out, t_out = self.mcm(season, trend)
            
            # Combine Season and Trend
            processed_images.append(s_out + t_out)
            
        # 4. MRM
        y = self.mrm(processed_images, amps, x.shape[1])
        
        return self.norm(x + self.dropout(y))


class TimeMixerPPForecaster(nn.Module):
    """TimeMixer++ 预测模型主体。"""
    
    def __init__(self, config: TimeMixerPPConfig):
        super().__init__()
        self.config = config
        
        # 1. Embedding & Normalize
        self.normalize = Normalize(config.enc_in, affine=True, non_norm=False)
        self.embedding = DataEmbedding_wo_pos(
            c_in=config.enc_in,
            d_model=config.d_model,
            dropout=config.dropout
        )
        
        # 2. Multi-scale Downsampling
        self.down_sampling_layers = nn.ModuleList()
        for i in range(config.num_scales):
            # Scale 0: Original
            # Scale 1: Downsample 2^1
            # ...
            if i == 0:
                self.down_sampling_layers.append(nn.Identity())
            else:
                self.down_sampling_layers.append(
                    nn.Sequential(
                        nn.AvgPool1d(kernel_size=config.down_sampling_window**i, stride=config.down_sampling_window**i)
                    )
                )

        # 3. Mixing Blocks (Shared or Separate per scale?)
        # Paper: "Processing multi-scale time series... MCM hierarchically aggregates"
        # We'll use one stack of MixerLayers per scale, or allow interaction?
        # For simplicity, we process each scale independently then ensemble, 
        # OR we perform the "MCM" cross-scale logic.
        # Given "TimeMixer" (original) architecture, it processes scales and mixes them.
        # TimeMixer++ enhances the block internals (MRTI/TID/MRM).
        # We will apply the TimeMixer++ Block to EACH scale.
        
        self.mixer_blocks = nn.ModuleList([
             nn.ModuleList([MixerBlock(config) for _ in range(config.e_layers)])
             for _ in range(config.num_scales)
        ])
        
        # Channel Mixing (only for coarsest scale)
        self.channel_mixing = ChannelMixing(config.d_model, dropout=config.dropout)
        
        # 4. Output Projection
        self.projection = nn.Linear(config.d_model, config.c_out)
        self.out_len_layers = nn.ModuleList([
            nn.Linear(config.seq_len // (config.down_sampling_window ** i), config.pred_len)
            for i in range(config.num_scales)
        ])
        
    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, C]
        
        # Norm
        x = self.normalize(x, 'norm')
        
        # Multi-scale Inputs
        x_scales = []
        for i in range(self.config.num_scales):
            # Downsample (AvgPool on [B, C, T])
            x_i = x.permute(0, 2, 1) # [B, C, T]
            x_i = self.down_sampling_layers[i](x_i)
            x_i = x_i.permute(0, 2, 1) # [B, T, C]
            
            # Embedding
            enc_out = self.embedding(x_i, None) # [B, T, D]
            
            # Channel Mixing on Coarsest Scale (Last one)
            if i == self.config.num_scales - 1:
                 enc_out = self.channel_mixing(enc_out)
            
            x_scales.append(enc_out)
            
        # Encoder Processing
        enc_outs = []
        for i in range(self.config.num_scales):
            out = x_scales[i]
            for layer in self.mixer_blocks[i]:
                out = layer(out)
            enc_outs.append(out)
            
        # Output Projection & Ensemble
        # [B, T_i, D] -> [B, T_i, C_out] -> [B, C_out, T_i] -> Linear -> [B, C_out, Pred] -> [B, Pred, C_out]
        
        y_final = torch.zeros([x.shape[0], self.config.pred_len, self.config.c_out], device=x.device)
        
        for i in range(self.config.num_scales):
            out = self.projection(enc_outs[i]) # [B, T_i, C_out]
            out = self.out_len_layers[i](out.permute(0, 2, 1)).permute(0, 2, 1) # [B, Pred, C_out]
            y_final = y_final + out
            
        # Denorm
        y_final = self.normalize(y_final, 'denorm')
        
        return y_final

