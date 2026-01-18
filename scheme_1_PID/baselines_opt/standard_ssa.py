import numpy as np

class StandardSSAOptimizer:
    """
    标准麻雀搜索算法 (SSA) 实现
    用于作为消融实验的基准 (Ablation Study Baseline)
    不包含：t-分布变异、柯西变异等改进策略
    """
    def __init__(self, pop_size=20, dim=6, max_iter=50, lb=0, ub=10):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # 种群比例配置
        self.p_percent = 0.2  # 发现者比例
        self.w_percent = 0.1  # 警戒者比例
        
        self.p_num = int(self.pop_size * self.p_percent)
        self.w_num = int(self.pop_size * self.w_percent)
        
    def optimize(self, fitness_func):
        """
        运行优化主循环
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
            # 排序
            sort_idx = np.argsort(fitness)
            X = X[sort_idx]
            fitness = fitness[sort_idx]
            
            # 更新全局最优
            if fitness[0] < best_fit:
                best_fit = fitness[0]
                best_X = X[0].copy()
            
            # --- 发现者 (Producer) 更新 ---
            R2 = np.random.rand()
            for i in range(self.p_num):
                if R2 < 0.8:
                    X[i] = X[i] * np.exp(-i / (np.random.rand() * self.max_iter))
                else:
                    X[i] = X[i] + np.random.normal() * np.ones(self.dim)
            X = np.clip(X, self.lb, self.ub)
            
            # --- 加入者 (Scrounger) 更新 ---
            best_pos = X[0] # 当前种群最优(排序后)
            worst_pos = X[-1]
            
            for i in range(self.p_num, self.pop_size):
                if i > self.pop_size / 2:
                    # 饥饿，随机飞
                    X[i] = np.random.normal() * np.exp((self.worst_idx(i) - X[i]) / (i**2)) # 原始公式示意
                    # 简化实现:
                    X[i] = np.random.uniform(self.lb, self.ub, self.dim)
                else:
                    # 飞往最优
                    # X_new = X_best + |X_i - X_best| * A+ * L
                    diff = np.abs(X[i] - best_pos)
                    X[i] = best_pos + diff * np.random.uniform(-1, 1)
            X = np.clip(X, self.lb, self.ub)

            # --- 警戒者 (Vigilante) 更新 ---
            perm = np.random.permutation(self.pop_size)
            vigilante_indices = perm[:self.w_num]
            
            for i in vigilante_indices:
                f_i = fitness[i]
                f_g = best_fit 
                f_w = fitness[-1]
                
                if f_i > f_g:
                    beta = np.random.normal(0, 1)
                    X[i] = best_X + beta * np.abs(X[i] - best_X)
                elif f_i == f_g:
                    k = np.random.uniform(-1, 1)
                    epsilon = 1e-8
                    X[i] = X[i] + k * (np.abs(X[i] - X[-1]) / (f_i - f_w + epsilon))
            X = np.clip(X, self.lb, self.ub)
            
            # --- 评估新种群 ---
            fitness = np.array([fitness_func(x) for x in X])
            
            # 更新当前最优
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_fit:
                best_fit = fitness[curr_best_idx]
                best_X = X[curr_best_idx].copy()
                
            loss_history.append(best_fit)
            # 注意：标准SSA没有最后的 t-分布变异步骤
            
        return best_X, loss_history

    def worst_idx(self, i):
        return 0
