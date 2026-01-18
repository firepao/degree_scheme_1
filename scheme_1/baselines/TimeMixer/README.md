# TimeMixer 温室数据预测

基于官方 TimeMixer 实现的温室数据预测模型。

## 模型特点

TimeMixer 是一个**多尺度时间序列预测模型**，核心思想包括：

1. **可分解混合（Decomposable Mixing）**：将序列分解为季节性（Season）和趋势性（Trend）成分
2. **多尺度处理**：通过下采样生成多个时间尺度的表示
3. **自底向上季节混合**：Bottom-up 方式处理季节性模式
4. **自顶向下趋势混合**：Top-down 方式处理趋势性模式
5. **通道独立/通道混合**：支持两种建模方式

## 文件结构

```
TimeMixer/
├── __init__.py              # 模块导出
├── README.md                # 说明文档
├── layers.py                # 辅助层（Embedding、Normalization、Decomposition）
├── timemixer.py             # 核心模型实现
├── run_timemixer_greenhouse.py    # 训练脚本
└── run_timemixer_greenhouse.ipynb # Notebook 版本
```

## 使用方法

### 训练脚本

```bash
python run_timemixer_greenhouse.py
```

### Notebook

打开 `run_timemixer_greenhouse.ipynb` 进行交互式训练和分析。

## 核心配置

- **seq_len**: 输入序列长度（默认 288，约 1 天）
- **pred_len**: 预测长度（默认 72，约 6 小时）
- **d_model**: 模型隐藏维度（默认 64）
- **d_ff**: FFN 隐藏维度（默认 128）
- **e_layers**: PDM Block 层数（默认 2）
- **down_sampling_layers**: 多尺度层数（默认 2）
- **down_sampling_window**: 下采样窗口（默认 2）
- **channel_independence**: 是否通道独立（默认 True）

## 参考

- 论文: TimeMixer (ICLR 2024)
- 官方代码: [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
