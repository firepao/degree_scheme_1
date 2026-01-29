from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from typing import List, Tuple, Optional
from .config import TimeMixerPPConfig

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
        # cuFFT 在半精度下仅支持 2 的幂次方维度，需要先转为 float32
        xf = torch.fft.rfft(x.float(), dim=1)
        # 计算平均振幅以找周期
        frequency_list = abs(xf).mean(0).mean(-1)
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.top_k)
        top_list = top_list.detach().cpu().numpy()
        
        periods = []
        images = []
        
        amp = abs(xf) 
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
        
        # Seasonality: Intra-period
        x_p = image.permute(0, 2, 3, 1).reshape(B*F, P, D)
        s, _ = self.col_attn(x_p, x_p, x_p)
        s = self.norm_s(x_p + s)
        season = s.reshape(B, F, P, D).permute(0, 3, 1, 2) # Back to [B, D, F, P]
        
        # Trend: Inter-period
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
        # Trend Mixing: 2D Transposed Conv (Inception for now)
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
        B, K, D = amps.shape
        weights = self.softmax(amps).unsqueeze(-1) # [B, K, D, 1]
        
        preds = []
        for k, img in enumerate(images_list):
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
        # 1. MRTI
        images, periods, amps = self.mrti(x)
        
        # 2. TID & 3. MCM
        processed_images = []
        for img in images:
            season, trend = self.tid(img)
            s_out, t_out = self.mcm(season, trend)
            processed_images.append(s_out + t_out)
            
        # 4. MRM
        y = self.mrm(processed_images, amps, x.shape[1])
        
        return self.norm(x + self.dropout(y))