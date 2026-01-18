import numpy as np

class ISSAOptimizer:
    """
    改进麻雀搜索算法 (ISSA) 实现
    用于PID参数的滚动优化
    特性: 自适应t分布变异
    """
    def __init__(self, pop_size=20, dim=6, max_iter=50, lb=0, ub=10):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # 种群比例配置
        self.p_percent = 0.2  # 发现者比例
        self.w_percent = 0.1  # 警戒者比例 (original paper suggest 10-20%)
        
        self.p_num = int(self.pop_size * self.p_percent)
        self.w_num = int(self.pop_size * self.w_percent)
        
    def optimize(self, fitness_func):
        """
        运行优化主循环
        Args:
            fitness_func: 函数 f(x) -> cost (越小越好)
        """
        # 1. 初始化种群
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([fitness_func(x) for x in X])
        
        # 记录全局最优
        best_idx = np.argmin(fitness)
        best_X = X[best_idx].copy()
        best_fit = fitness[best_idx]
        
        loss_history = []
        
        for t in range(self.max_iter):
            # 对种群按适应度排序 (为了区分发现者和加入者)
            sort_idx = np.argsort(fitness)
            X = X[sort_idx]
            fitness = fitness[sort_idx]
            
            # 更新全局最优 (排序后第一个就是最好的)
            if fitness[0] < best_fit:
                best_fit = fitness[0]
                best_X = X[0].copy()
            
            # --- 2. 发现者 (Producer) 更新 ---
            # 这一部分负责全局搜索
            R2 = np.random.rand()  # 预警值
            
            for i in range(self.p_num):
                if R2 < 0.8: # 安全状态，广泛搜索
                    # X_new = X * exp(-i / (alpha * max_iter))
                    X[i] = X[i] * np.exp(-i / (np.random.rand() * self.max_iter))
                else: # 危险状态，向安全区移动
                    # X_new = X + Q * L
                    X[i] = X[i] + np.random.normal() * np.ones(self.dim)
                    
            # 边界处理
            X = np.clip(X, self.lb, self.ub)
            
            # --- 3. 加入者 (Scrounger) 更新 ---
            # 跟随发现者
            # 排序后的 best 位于 index 0
            best_pos = X[0]
            worst_pos = X[-1]
            
            for i in range(self.p_num, self.pop_size):
                if i > self.pop_size / 2:
                    # 适应度太差，饥饿，飞往其他地方
                    X[i] = np.random.normal() * np.exp((self.worst_idx(i, self.pop_size) - X[i]) / (i**2))
                    # 注意：上述公式是原始SSA的变体示意，常用实现如下:
                    # Q * exp((X_worst - X_curr) / i^2)
                    # 这里简化处理:
                    X[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    # 靠近最优发现者
                    # | X_i - X_best |
                    A = np.random.choice([1, -1], (self.dim, 1))
                    # A_plus = A^T * (A * A^T)^(-1) -> simply A here for 1D logic approx
                    # 标准公式: X_new = X_best + |X_i - X_best| * A+ * L
                    # 简化实现:
                    diff = np.abs(X[i] - best_pos)
                    X[i] = best_pos + diff * np.random.uniform(-1, 1) # simple approach
                    
            X = np.clip(X, self.lb, self.ub)

            # --- 4. 警戒者 (Vigilante) 更新 ---
            # 随机选取一部分
            perm = np.random.permutation(self.pop_size)
            vigilante_indices = perm[:self.w_num]
            
            for i in vigilante_indices:
                f_i = fitness[i]
                f_g = best_fit # 全局最优
                f_w = fitness[-1] # 当前最差
                
                if f_i > f_g:
                    # 处于外围，向最优靠近
                    beta = np.random.normal(0, 1) # 步长控制
                    X[i] = best_X + beta * np.abs(X[i] - best_X)
                elif f_i == f_g:
                    # 处于中心，意识到危险，随机游走
                    k = np.random.uniform(-1, 1)
                    epsilon = 1e-8 # 避免除零
                    X[i] = X[i] + k * (np.abs(X[i] - X[-1]) / (f_i - f_w + epsilon))
                    
            X = np.clip(X, self.lb, self.ub)
            
            # 重新计算适应度
            fitness = np.array([fitness_func(x) for x in X])
            
            # 更新当前最优
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_fit:
                best_fit = fitness[curr_best_idx]
                best_X = X[curr_best_idx].copy()
                
            # --- 5. 自适应t分布变异 (Adaptive t-distribution Mutation) ---
            # 针对全局最优位置进行变异，尝试跳出局部最优
            # 自由度 dof = 当前迭代次数 t (防止早期收敛太快，后期收敛正态分布)
            
            dof = t + 1
            # 生成变异扰动: t分布随机数
            t_val = np.random.standard_t(dof, size=self.dim)
            
            # 变异后的新位置
            X_new_best = best_X + best_X * t_val
            X_new_best = np.clip(X_new_best, self.lb, self.ub)
            
            # 贪婪策略：如果变异后更好，则接受
            fit_mutated = fitness_func(X_new_best)
            if fit_mutated < best_fit:
                best_X = X_new_best
                best_fit = fit_mutated
                # 可以选择是否更新种群中最差的个体为这个新最优，增加多样性
                worst_idx = np.argmax(fitness)
                X[worst_idx] = best_X
                fitness[worst_idx] = best_fit

            loss_history.append(best_fit)
            
        return best_X, loss_history

    def worst_idx(self, i, total):
        # 辅助计算
        return 0 # simplified
