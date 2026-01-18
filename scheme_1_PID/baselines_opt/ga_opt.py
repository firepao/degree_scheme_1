import numpy as np

class GAOptimizer:
    """
    遗传算法 (Genetic Algorithm) 实现
    经典进化算法基准
    """
    def __init__(self, pop_size=20, dim=6, max_iter=50, lb=0, ub=10):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
    def optimize(self, fitness_func):
        # 1. 初始化种群
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([fitness_func(x) for x in X])
        
        best_idx = np.argmin(fitness)
        best_X = X[best_idx].copy()
        best_fit = fitness[best_idx]
        
        loss_history = []
        
        for t in range(self.max_iter):
            # --- 选择 (Tournament Selection) ---
            new_X = []
            for _ in range(self.pop_size):
                # 随机选3个，通过锦标赛选最好的
                candidates_idx = np.random.choice(self.pop_size, 3, replace=False)
                c_data = X[candidates_idx]
                c_fit = fitness[candidates_idx]
                winner_idx = np.argmin(c_fit)
                new_X.append(c_data[winner_idx])
            new_X = np.array(new_X)
            
            # --- 交叉 (Crossover) ---
            for i in range(0, self.pop_size, 2):
                if i+1 < self.pop_size and np.random.rand() < self.crossover_rate:
                    # 简单的算术交叉
                    alpha = np.random.rand()
                    c1 = alpha * new_X[i] + (1-alpha) * new_X[i+1]
                    c2 = (1-alpha) * new_X[i] + alpha * new_X[i+1]
                    new_X[i] = c1
                    new_X[i+1] = c2
            
            # --- 变异 (Mutation) ---
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    # 随机选择一个维度进行变异
                    dim_idx = np.random.randint(0, self.dim)
                    new_X[i][dim_idx] = np.random.uniform(self.lb, self.ub)
                    
            # 边界处理
            X = np.clip(new_X, self.lb, self.ub)
            
            # --- 评估 ---
            fitness = np.array([fitness_func(x) for x in X])
            
            # 更新全局最优
            curr_best_idx = np.argmin(fitness)
            if fitness[curr_best_idx] < best_fit:
                best_fit = fitness[curr_best_idx]
                best_X = X[curr_best_idx].copy()
                
            loss_history.append(best_fit)
            
        return best_X, best_fit, loss_history # 保持接口一致 (ga return 3 values similar to PSO above)
        # 注意: standard_ssa 返回了 2 values, 我需要统一接口
