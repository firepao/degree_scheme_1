# TPLC 核心组件消融实验指南

本文档详细说明 TPLC 模型的核心组件及其消融实验设计。

## 📋 消融实验概览

| 消融名称 | 关闭的组件 | 说明 |
|---------|-----------|------|
| `full` | - | 完整模型（作为基准） |
| `no_fft` | FFT 周期提取 | 使用固定周期替代动态 FFT |
| `no_reshape_2d` | 1D→2D 重塑 | 使用纯 1D 卷积替代 2D |
| `no_depthwise` | 深度可分离卷积 | 使用标准 Conv2D 替代 |
| `no_multi_scale` | 多尺度处理 | 仅使用原始尺度 |
| `no_multi_period` | 多周期融合 | 仅使用 Top-1 周期 |
| `no_amp_weight` | 振幅加权融合 | 不使用 FFT 振幅作为周期权重 |
| `no_residual` | 残差连接 | 移除残差路径（训练增强，非核心创新） |
| `no_revin` | RevIN 归一化 | 移除分布对齐（训练增强，非核心创新） |
| `baseline_mlp` | 所有组件 | MLP 基线模型 |

## 🔧 核心组件详解

### 1. FFT 周期提取 (`use_fft`)

**作用**: 从输入序列中动态发现主要周期。

```python
# 开启（默认）: 使用 FFT 提取 top-k 周期
periods = extract_topk_periods(x, top_k=3)  # 例如 [24, 12, 8]

# 关闭: 使用固定周期
periods = [24, 12, 8]  # 预设值
```

**消融假设**: FFT 动态周期提取能够自适应不同数据的周期特性，优于固定周期。

---

### 2. 1D→2D 重塑 (`use_reshape_2d`)

**作用**: 将 1D 时序数据按周期展开为 2D 图像，利用 2D 卷积捕获行内（周期内）和行间（周期间）模式。

```
原始 1D: [T₁, T₂, T₃, ..., T₉₆]
         ↓ reshape with period=24
2D 图像: [[T₁,  T₂,  ..., T₂₄],   ← 第 1 个周期
         [T₂₅, T₂₆, ..., T₄₈],   ← 第 2 个周期
         [T₄₉, T₅₀, ..., T₇₂],   ← 第 3 个周期
         [T₇₃, T₇₄, ..., T₉₆]]   ← 第 4 个周期
```

**消融假设**: 2D 视角能够同时捕获周期内变化和跨周期趋势。

---

### 3. 深度可分离卷积 (`use_depthwise`)

**作用**: 使用 Depthwise + Pointwise 卷积替代标准卷积，减少参数量同时保持表达能力。

```
标准 Conv2D:    参数 = C_in × C_out × K × K
深度可分离:     参数 = C_in × K × K + C_in × C_out
                     (depthwise)     (pointwise)
```

**消融假设**: 深度可分离卷积在参数效率和泛化能力上优于标准卷积。

---

### 4. 多尺度处理 (`use_multi_scale`)

**作用**: 通过下采样生成多个时间粒度的表示，捕获不同时间尺度的模式。

```
Scale 0 (原始):   L = 96
Scale 1 (2x下采样): L = 48
Scale 2 (4x下采样): L = 24
```

**消融假设**: 多尺度能够捕获从细粒度到粗粒度的多层次时序模式。

---

### 5. 多周期融合 (`use_multi_period`)

**作用**: 对多个显著周期分别处理后加权融合。

```python
# Top-3 周期: [24, 12, 8]
# 每个周期独立处理后通过振幅权重融合
output = a₁×output_p24 + a₂×output_p12 + a₃×output_p8
```

**消融假设**: 多周期信息互补，融合后优于单一周期。

---

### 6. 振幅加权融合 (`use_amp_weight`)

**作用**: 使用 FFT 提取的振幅作为多周期融合权重，体现各周期重要性。

```python
# FFT 振幅 -> Softmax 权重
weights = softmax(amplitudes)
output = sum(w * out_k for w, out_k in zip(weights, outputs))
```

**消融假设**: 振幅加权能够更准确地强调主周期信息。

---

### 7. 多尺度融合 (`use_scale_weight`)

**作用**: 多尺度预测结果的融合方式。

```python
# 论文中为等权求和（默认）
output = sum(outputs_scale)

# 可选：可学习权重融合
output = sum(w * out_s for w, out_s in zip(weights, outputs_scale))
```

**消融假设**: 多尺度融合有助于整合不同时间粒度的预测能力。

---

### 8. 残差连接 (`use_residual`)

**作用**: 添加跳跃连接促进梯度流动，加速收敛。

```python
output = model_output + residual_proj(input)
```

**消融假设**: 残差连接有助于训练更深/更复杂的网络。

---

### 9. RevIN 归一化 (`use_revin`)

**作用**: 可逆实例归一化，处理时序数据的分布偏移问题。

```python
# 归一化
x_norm = (x - mean) / std
# 预测后反归一化
y = y_raw * std + mean
```

**消融假设**: RevIN 能够对齐训练和测试数据的分布差异。

---

## 🚀 运行消融实验

### 快速开始

```bash
cd scheme_1/TPLC_Net/ablation

# 运行完整模型（作为基准）
python run_ablation_new.py --config full --epochs 20

# 运行单个消融
python run_ablation_new.py --config no_fft --epochs 20

# 运行全部消融实验
python run_ablation_new.py --config all --epochs 20 --seeds 42 123 456
```

### 使用 VS Code Task

在 VS Code 命令面板中选择:
- `消融实验: 核心组件 (full)`
- `消融实验: 核心组件 (all)`
- `消融实验: 核心组件 (自定义)`

### 自定义参数

```bash
python run_ablation_new.py \
    --config all \
    --team letsgrow \
    --seq-len 96 \
    --pred-len 24 \
    --epochs 30 \
    --seeds 42 123 456 789 2024 \
    --output-dir ./results
```

---

## 📊 结果解读

消融实验会输出两个 CSV 文件:

1. **详细结果** (`ablation_detail_*.csv`): 每次运行的原始指标
2. **汇总结果** (`ablation_summary_*.csv`): 按配置分组的均值±标准差

### 结果表格示例

```
配置                      描述                  MAE             RMSE            参数量        Δ MAE%
========================================================================================
full                     完整模型 (基准)        0.1234±0.0012   0.1567±0.0015   45,678       +0.00%
no_fft                   - FFT 周期提取         0.1345±0.0018   0.1678±0.0021   45,678       +9.00%
no_reshape_2d            - 1D→2D 重塑           0.1567±0.0025   0.1890±0.0030   38,234       +27.00%
no_depthwise             - 深度可分离卷积       0.1289±0.0014   0.1612±0.0017   89,012       +4.46%
no_multi_scale           - 多尺度处理           0.1312±0.0016   0.1645±0.0019   32,456       +6.32%
no_multi_period          - 多周期融合           0.1378±0.0019   0.1701±0.0022   41,234       +11.67%
no_residual              - 残差连接             0.1401±0.0021   0.1734±0.0024   44,567       +13.54%
no_revin                 - RevIN 归一化         0.1267±0.0013   0.1589±0.0016   45,456       +2.67%
baseline_mlp             MLP 基线               0.1890±0.0035   0.2234±0.0042   25,678       +53.16%
```

### 分析要点

1. **Δ MAE%**: 相对于完整模型的 MAE 变化百分比，越高说明该组件越重要
2. **参数量对比**: 某些消融（如 no_depthwise）会增加参数量
3. **稳定性**: 标准差反映模型对不同随机种子的敏感度

---

## 📁 文件结构

```
ablation/
├── components.py         # 核心组件实现
├── ablation_model.py     # 消融模型类
├── run_ablation_new.py   # 消融实验运行脚本
├── ABLATION_GUIDE_NEW.md # 本文档
├── results/              # 实验结果输出目录
│
# 旧版模块（保留兼容）
├── revin.py
├── inception.py
├── decomposition.py
├── stacked_block.py
└── run_ablation.py
```

---

## 🔬 设计理念

TPLC 模型的核心创新在于：

1. **周期感知**: 通过 FFT 自动发现数据中的周期性
2. **维度提升**: 将 1D 时序转换为 2D 图像，利用成熟的 CV 技术
3. **高效参数化**: 深度可分离卷积减少过拟合风险
4. **多尺度多周期**: 从多个视角捕获时序模式

消融实验的目的是量化验证每个组件的贡献，为论文提供实验支撑。
