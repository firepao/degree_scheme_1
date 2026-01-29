#!/usr/bin/env python
"""简单测试消融实验数据准备流程。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tplc_algo.pipeline import prepare_greenhouse_datasets, make_loaders

def test_data_pipeline():
    print("测试数据管道...")
    
    try:
        # 测试数据准备
        prepared_data = prepare_greenhouse_datasets(
            dataset_root=Path(r"D:\degree_code_scheme_1\scheme_1\datasets\自主温室挑战赛"),
            team="AICU",
            seq_len=96,
            pred_len=24,
        )

        print(f"特征维度: {len(prepared_data.feature_cols)}")
        print(f"目标维度: {len(prepared_data.target_cols)}")
        print(f"训练集大小: {len(prepared_data.train_ds)}")

        # 测试数据加载器
        train_loader, val_loader, test_loader = make_loaders(prepared_data, batch_size=32)
        sample = next(iter(train_loader))
        
        print(f"批次形状 - x: {sample[0].shape}, y: {sample[1].shape}")
        print("数据测试成功！")
        
        return prepared_data, train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_data_pipeline()