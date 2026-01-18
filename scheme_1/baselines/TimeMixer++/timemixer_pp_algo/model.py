from __future__ import annotations

import torch
import torch.nn as nn
from .config import TimeMixerPPConfig
from .layers import DataEmbedding_wo_pos, Normalize
from .modules import MixerBlock, ChannelMixing

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

        # 3. Mixing Blocks
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
            
        # Denorm (pass target_idx if available)
        y_final = self.normalize(y_final, 'denorm', target_idx=self.config.target_idx)
        
        return y_final