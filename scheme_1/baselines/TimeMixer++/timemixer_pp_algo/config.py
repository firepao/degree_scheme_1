from dataclasses import dataclass

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
    target_idx: list[int] | None = None # 目标变量在输入特征中的索引（用于非对称 Denorm）
