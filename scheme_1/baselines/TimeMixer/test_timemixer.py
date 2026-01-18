"""测试 TimeMixer 模型是否可以正常运行。"""

import torch
from timemixer import TimeMixerForecaster, TimeMixerConfig

# 测试配置
config = TimeMixerConfig(
    seq_len=288,
    pred_len=72,
    enc_in=101,  # 输入特征数
    c_out=3,     # 输出特征数
    d_model=64,
    d_ff=128,
    e_layers=2,
    dropout=0.1,
    down_sampling_layers=2,
    down_sampling_window=2,
    down_sampling_method='avg',
    decomp_method='moving_avg',
    moving_avg=25,
    top_k=5,
    channel_independence=True,
)

# 创建模型
model = TimeMixerForecaster(config)
model.eval()

# 创建测试输入
batch_size = 4
x = torch.randn(batch_size, config.seq_len, config.enc_in)

print(f"输入形状: {x.shape}")
print(f"期望输出形状: [{batch_size}, {config.pred_len}, {config.c_out}]")

# 前向传播
try:
    with torch.no_grad():
        y = model(x)
    print(f"实际输出形状: {y.shape}")
    print("✓ 模型前向传播成功！")
    
    # 验证输出形状
    assert y.shape == (batch_size, config.pred_len, config.c_out), \
        f"输出形状不匹配！期望 {(batch_size, config.pred_len, config.c_out)}，实际 {y.shape}"
    print("✓ 输出形状正确！")
    
    # 测试梯度回传
    model.train()
    y = model(x)
    loss = y.sum()
    loss.backward()
    print("✓ 梯度回传成功！")
    
    print("\n所有测试通过！模型可以正常使用。")
    
except Exception as e:
    print(f"\n✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
