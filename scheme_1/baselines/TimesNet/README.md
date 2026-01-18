# TimesNet 温室数据预测

本目录包含 TimesNet 模型在温室环境数据上的复现和应用。

## 简介

**TimesNet** 是一个强大的时间序列分析模型，通过将 1D 时间序列转换为 2D 表示来捕获周期性模式。

**论文：** [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://openreview.net/pdf?id=ju_Uqw384Oq)  
**源码：** https://github.com/thuml/Time-Series-Library

### 核心思想

1. **FFT 提取周期**：通过快速傅里叶变换（FFT）自动发现时间序列的主要周期
2. **1D → 2D 转换**：将时间序列按周期重塑为 2D 表示
3. **2D 卷积**：使用 Inception 卷积块提取周期内和周期间的信息
4. **多尺度融合**：对多个周期的结果进行自适应加权融合

## 文件结构

```
baselines/TimesNet/
├── __init__.py                      # 模块导出
├── layers.py                        # 层实现（Embedding、Inception Block）
├── timesnet.py                      # TimesNet 模型核心代码
├── run_timesnet_greenhouse.py       # 训练脚本
├── run_timesnet_greenhouse.ipynb    # 训练 Notebook（推荐）
├── README.md                        # 本文件
└── results/                         # 实验结果（自动创建）
    └── timesnet_greenhouse_AICU_YYYYMMDD_HHMMSS/
        ├── config.json              # 实验配置
        ├── environment.json         # 环境信息
        ├── history.csv              # 训练历史
        ├── artifacts/
        │   └── metrics.json         # 测试指标
        ├── checkpoints/
        │   └── timesnet_best.pt     # 最佳模型
        └── figures/                 # 可视化图表
            ├── loss_curve.png
            ├── pred_curve_*.png
            ├── error_distribution.png
            └── error_per_step.png
```

## 快速开始

### 1. 环境准备

确保已安装以下依赖（参考 `TPLC_Net/requirements.txt`）：

```bash
torch>=1.12.0
numpy
pandas
matplotlib
scikit-learn
tqdm
```

### 2. 使用 Notebook（推荐）

打开 `run_timesnet_greenhouse.ipynb`，逐个单元格执行：

1. **配置参数**：设置数据路径、模型超参数、训练配置
2. **数据准备**：加载并预处理温室数据
3. **构建模型**：创建 TimesNet 模型实例
4. **训练**：训练模型并保存结果
5. **评估**：在测试集上评估性能
6. **可视化**：查看训练曲线、预测结果、误差分布

### 3. 使用脚本

```bash
cd D:/degree_code/scheme_1/baselines/TimesNet
python run_timesnet_greenhouse.py
```

## 模型配置

### 关键超参数

```python
# 模型结构
d_model = 64        # Embedding 维度
d_ff = 128          # Feedforward 维度
e_layers = 2        # TimesBlock 层数
top_k = 3           # 提取 top-k 个主要周期
num_kernels = 6     # Inception Block 卷积核数量

# 数据配置
seq_len = 288       # 输入序列长度（约 1 天）
pred_len = 72       # 预测长度（约 6 小时）

# 训练配置
batch_size = 32
lr = 1e-3
epochs = 20
```

### 超参数调优建议

- `top_k`：温室数据通常有昼夜周期，建议 2-4
- `e_layers`：层数越多模型越复杂，建议 2-3
- `d_model` / `d_ff`：根据数据复杂度调整，建议 64-128
- `seq_len`：应包含至少 1-2 个完整周期

## 实验结果

实验结果自动保存到 `results/` 目录，包含：

### 1. 配置文件

- `config.json`：完整的实验配置
- `environment.json`：Python 版本、依赖版本等

### 2. 训练历史

- `history.csv`：每个 epoch 的训练/验证损失

### 3. 测试指标

- `metrics.json`：测试集上的 MSE、MAE、RMSE（包含原始尺度）

### 4. 可视化

- `loss_curve.png`：训练/验证损失曲线
- `pred_curve_*.png`：各目标变量的预测示例
- `error_distribution.png`：误差分布直方图
- `error_per_step.png`：不同预测步长的误差变化

## 与 TPLC-Net 的对比

| 特性              | TimesNet           | TPLC-Net           |
|-------------------|--------------------|--------------------|
| 核心思想          | FFT + 2D 卷积       | 多尺度周期学习      |
| 周期提取          | 自动（FFT）         | 自动（学习）        |
| 参数量            | 中等               | 较大               |
| 训练速度          | 快                 | 中等               |
| 适用场景          | 明显周期性数据      | 复杂多周期数据      |

## 常见问题

### Q1: FFT 提取不到有效周期怎么办？

**A:** 检查以下几点：
- `seq_len` 是否足够长（至少包含 1-2 个完整周期）
- 数据是否已正确标准化
- 尝试增加 `top_k` 值

### Q2: 模型训练不收敛？

**A:** 尝试：
- 降低学习率（如 `lr=5e-4`）
- 增加 `grad_clip_max_norm`
- 检查数据预处理是否正确

### Q3: 预测精度不高？

**A:** 考虑：
- 增加 `e_layers`（更深的网络）
- 增大 `d_model` 和 `d_ff`（更大的容量）
- 调整 `top_k`（捕获更多周期信息）
- 增加训练数据或数据增强

## 参考资料

1. **论文**：Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. ICLR 2023.
2. **官方实现**：https://github.com/thuml/Time-Series-Library
3. **Tutorial**：`D:/Time_Series_Library/Time-Series-Library/tutorial/TimesNet_tutorial.ipynb`

## 引用

如果使用本实现，请引用原论文：

```bibtex
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```

## 联系方式

如有问题，请参考：
- 原项目 issues：https://github.com/thuml/Time-Series-Library/issues
- TPLC_Net 项目文档
