
import torch
import sys
import os

# Ensure the current directory is in path to find layers
sys.path.append(os.path.join(os.getcwd(), 'degree_code', 'scheme_1', 'baselines', 'TimeMixer++'))

try:
    from timemixer_pp import TimeMixerPPConfig, TimeMixerPPForecaster
except ImportError:
    # Try local import if running from the folder directly
    from timemixer_pp import TimeMixerPPConfig, TimeMixerPPForecaster

def test_timemixer_pp():
    config = TimeMixerPPConfig(
        seq_len=96,
        pred_len=24,
        enc_in=7,
        c_out=7,
        d_model=16,
        d_ff=32,
        e_layers=2,
        num_scales=3,
        top_k=3,
        dropout=0.1,
        down_sampling_window=2,
        down_sampling_layers=2,
        channel_independence=False
    )
    
    model = TimeMixerPPForecaster(config)
    
    # Create dummy input [B, T, C]
    x = torch.randn(2, 96, 7)
    
    print("Input shape:", x.shape)
    
    # Forward pass
    y = model(x)
    
    print("Output shape:", y.shape)
    
    expected_shape = (2, 24, 7)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    print("Test passed!")

if __name__ == "__main__":
    test_timemixer_pp()
