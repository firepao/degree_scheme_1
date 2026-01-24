# Role
你是一个精通温室环境控制、生物物理模型建模以及群智能优化算法（Swarm Intelligence）的 Python 专家。

# Context
我正在开发一个智能温室控制系统。
1.  **现有代码**：我已经实现了基于物理的温室环境模型 (`VirtualGreenhouse`)，改进的麻雀搜索算法 (`ISSAOptimizer`)，以及基础的 PID 控制器。目前的代码逻辑是使用 ISSA 优化 PID 参数，以追踪固定的温度/湿度设定点。
2.  **参考主要文献**：我上传了一篇名为《利用 PSO-MPC 算法优化温室控制中的作物产量和减少能源消耗》的论文。
3.  **目标**：我需要按照这篇论文的实验设计思路（特别是 3.2 和 3.3 节），实现“作物产量模型”和“能耗模型”，并将优化目标从“误差最小化”转变为“经济利润最大化”。

# Task
请基于我现有的代码结构，分步帮我生成以下模块的代码，并演示如何整合它们：

## Step 1: 实现作物生长模型 (Crop Yield Model)
参考论文公式 (2) 或基于 Vanthoor et al. 的简化模型，创建一个 `TomatoYieldModel` 类。
* **输入**：当前的温室环境状态（温度 Temperature, 光合有效辐射 PAR, CO2 浓度）。
* **逻辑**：
    * 计算光合作用速率 (Photosynthesis rate)。
    * 计算同化物在叶、茎、果实中的分配。
    * 计算干物质积累量 (Dry matter accumulation)。
* **输出**：当前时间步长的作物产量增量 (Yield increase in kg)。

## Step 2: 实现能耗模型 (Energy Consumption Model)
参考论文公式 (4)，创建一个 `EnergyCostModel` 类。
* **输入**：控制动作（加热器功率/阀门开度, 通风率, 补光灯开启状态）。
* **参数**：需定义加热价格、电力价格等系数（可以先用占位符变量）。
* **输出**：当前时间步长的能源消耗成本 (Cost in currency)。

## Step 3: 定义经济目标函数 (Economic Objective Function)
修改现有的适应度函数 `fitness_function`。
* **旧逻辑**：最小化设定点追踪误差 (MSE)。
* **新逻辑**：最大化利润 J。
    * `J = sum(Yield_Increase * Price_Crop) - sum(Energy_Cost)`
    * 你需要定义一个函数，该函数接收一组控制序列（或一组环境设定点序列），调用 Step 1 和 Step 2 的模型进行模拟，返回负的利润值（因为优化器通常是求最小化）。

## Step 4: 整合为 ISSA-MPC 架构
编写一个新的主循环代码块，演示如何使用 `ISSAOptimizer` 来直接优化**未来 N 步的最佳设定点 (Optimal Setpoints)**。
* 优化器应该寻找能让 `Economic Objective Function` 最优的温度/湿度设定点轨迹。
* 将优化出的最佳设定点传给底层的 PID 控制器去执行（或者如果建议直接控制，请说明理由）。

# Requirements
1.  **代码风格**：使用 Python，风格需与我现有的 `Module4_Demo.ipynb` 保持一致，使用 NumPy 进行向量化计算。
2.  **注释**：在关键公式处添加注释，说明对应论文中的哪个公式或逻辑。
3.  **模块化**：请将作物模型和能耗模型封装为独立的类或函数，方便我在主程序中调用。
4.  **鲁棒性**：处理好除零错误或物理上不可能的数值（如负产量）。

请先解释你的实现思路，然后给出完整的代码块。