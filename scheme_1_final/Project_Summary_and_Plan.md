# 丹参温室智能控制系统 (Scheme 1) - 项目总览与规划

## 1. 项目代码架构与逻辑 (Project Structure & Code Logic)

本项目的核心目标是构建一个集**环境预测**、**智能控制**与**经济效益优化**为一体的温室管理系统。代码主要分布在以下三个核心目录中，逐步递进实现最终的 ISSA-PID-MPC 框架。

### 1.1 目录结构说明

```text
degree_code_scheme_1/
├── scheme_1/                # [预测层] 环境变量预测模型 (TPLC-Net)
│   ├── TPLC_Net/            # 模型核心代码
│   ├── run_compare_experiment.py  # 自动化对比实验脚本
│   └── ...
│
├── scheme_1_PID/            # [控制层-基础] ISSA 优化 PID 参数
│   ├── greenhouse_env.py    # 虚拟温室环境模拟器
│   ├── issa.py              # 改进麻雀搜索算法 (ISSA) 实现
│   ├── pid.py               # PID 控制器实现
│   └── Module4_Demo.ipynb   # ISSA-PID 优化演示
│
└── scheme_1_final/          # [集成层-进阶] 经济效益导向的 ISSA-MPC 框架
    ├── issa_pid_mpc_framework.py  # (核心) 抽象框架接口定义 (Predictor, Solver, CostFunc)
    ├── mpc_framework.py           # MPC 通用组件实现
    ├── ISSA_PID_MPC_Framework_Guide.ipynb # 框架使用与开发指南
    ├── Project_Summary_and_Plan.md # 本文档
    └── Future_plan.md             # 详细开发任务书
```

### 1.2 核心代码逻辑流 (System Architecture)

整个系统的数据流与控制流如下所示：

```mermaid
graph TD
    subgraph Environment [温室环境]
        Sensors(传感器数据) --> Predictor
        Actuators(执行机构) --控制动作--> Sensors
    end

    subgraph Prediction_Layer [预测层 (scheme_1)]
        Predictor[TPLC-Net 模型] --预测未来环境--> Optimizer
    end

    subgraph Optimization_Layer [决策层 (scheme_1_final)]
        YieldModel[作物生长模型] --产量预估--> ProfitFunc
        EnergyModel[能耗模型] --成本计算--> ProfitFunc
        ProfitFunc[经济效益目标函数 J] --> Optimizer[ISSA/MPC 优化器]
        Optimizer --最优设定点/参数--> Controller
    end

    subgraph Control_Layer [执行层 (scheme_1_PID)]
        Controller[PID 控制器] --PWM/模拟量--> Actuators
    end
```

### 1.3 `scheme_1_final` 关键文件详解

*   **`issa_pid_mpc_framework.py`**:
    *   **作用**: 定义了系统的核心抽象基类 (Abstract Base Classes)，解耦了预测、优化和成本计算模块。
    *   **核心类**:
        *   `EnvironmentPredictor`: 预测器接口 (Oracle)，用于推演未来状态。
        *   `OptimizationSolver`: 优化器接口，定义如何寻找最优解。
        *   `CostFunction`: 代价函数接口，未来将在此处实现 "负利润" 计算。
*   **`mpc_framework.py`**:
    *   **作用**: 提供了模型预测控制 (MPC) 的通用实现模板，展示了如何利用预测器和代价函数进行滚动时域优化。
*   **`ISSA_PID_MPC_Framework_Guide.ipynb`**:
    *   **作用**: 交互式教程，演示如何通过继承上述接口来构建具体的控制系统。

---

## 2. Scheme 1 方案总结 (Summary of Current Scheme)

目前方案一 (Scheme 1) 已经完成了从环境预测到基础智能控制的功能闭环。

### 2.1 环境预测模型：TPLC-Net (D:\degree_code\scheme_1)
针对温室数据的多尺度耦合特性，提出了 **TPLC-Net (Time-Frequency Pattern Learning with Lightweight Convolution)**。
*   **创新点**:
    *   **时频域结合**: 利用 FFT 将 1D 序列转换为 2D 图像，捕获周期性模式。
    *   **多尺度架构**: 分别处理短期波动与长期趋势。
    *   **轻量化设计**: 采用深度可分离卷积，极大降低了计算量，适合边缘端部署。
*   **状态**: 已完成与 LSTM, PatchTST, TimesNet 等基线模型的对比实验，证明了其在多变量长时序预测上的优越性。

### 2.2 基础智能控制：ISSA-PID (D:\degree_code\scheme_1_PID)
实现了基于改进麻雀搜索算法 (ISSA) 的 PID 参数自整定策略。
*   **创新点**:
    *   **ISSA 算法**: 引入 t-分布变异策略，增强了算法跳出局部最优的能力。
    *   **自适应调参**: 能够根据环境变化自动寻找最优的 $K_p, K_i, K_d$ 参数。
*   **状态**: 已在虚拟温室环境中验证，能够使 PID 控制器快速、稳定地追踪温度和湿度设定曲线 (Setpoints)。

---

## 3. 后续工作计划 (Future Roadmap)

接下来的工作重心将从“追踪设定点”升级为“最大化种植效益”，实现真正的 **ISSA-MPC**。

### 阶段一：机理模型引入 (Step 1 & 2)
为了计算经济效益，必须引入生物与物理模型。
*   **作物生长模型 (`TomatoYieldModel`)**:
    *   基于 Vanthoor 模型，输入光、温、CO2，输出**产量增量 (kg)**。
*   **能耗成本模型 (`EnergyCostModel`)**:
    *   基于热力学公式，输入加热、通风动作，输出**电力与燃料成本 (¥)**。

### 阶段二：经济目标函数构建 (Step 3)
将优化目标从 MSE (均方误差) 替换为 **净利润 (Net Profit)**。
*   $J = \text{Revenue}(\text{Yield}) - \text{Cost}(\text{Energy})$
*   优化器将寻找能使 $J$ 最大化的环境设定点轨迹，而非简单地维持恒定温度。

### 阶段三：完整框架集成 (Step 4)
实现 `scheme_1_final` 中的完整闭环：
1.  **预测**: TPLC-Net 提供未来环境趋势。
2.  **决策**: ISSA 在滚动窗口内寻找最佳设定点序列，最大化预测利润。
3.  **执行**: 底层 PID 控制器负责将环境高精度控制在生成的设定点上。
