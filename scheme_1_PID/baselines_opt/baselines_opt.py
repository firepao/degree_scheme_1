import numpy as np

class PSOOptimizer:
    """
    粒子群优化算法 (PSO) 基准实现
    用于与 ISSA 进行对比
    """
    def __init__(self, pop_size=20, dim=6, max_iter=50, lb=0, ub=10):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        
        # PSO 参数
        self.w = 0.7   # 惯性权重
        self.c1 = 1.5  # 认知系数 (个体学习)
        self.c2 = 1.5  # 社会系数 (群体学习)

    def optimize(self, fitness_func):
        """
        运行优化主循环
        Args:
            fitness_func: 函数 f(x) -> cost (越小越好)
        Returns:
            best_X: 全局最优位置
            best_fit: 全局最优适应度
            loss_history: 迭代收敛曲线
        """
        # 1. 初始化种群
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        
        # 计算初始适应度
        fitness = np.array([fitness_func(x) for x in X])
        
        # 初始化个体历史最优 (pbest)
        pbest_X = X.copy()
        pbest_fit = fitness.copy()
        
        # 初始化全局最优 (gbest)
        best_idx = np.argmin(fitness)
        gbest_X = X[best_idx].copy()
        gbest_fit = fitness[best_idx]
        
        loss_history = []
        
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                # 速度更新
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                V[i] = self.w * V[i] + \
                       self.c1 * r1 * (pbest_X[i] - X[i]) + \
                       self.c2 * r2 * (gbest_X - X[i])
                
                # 位置更新
                X[i] = X[i] + V[i]
                
                # 边界处理
                X[i] = np.clip(X[i], self.lb, self.ub)
                
                # 评估
                cost = fitness_func(X[i])
                
                # 更新 Pbest
                if cost < pbest_fit[i]:
                    pbest_fit[i] = cost
                    pbest_X[i] = X[i].copy()
                    
                # 更新 Gbest
                if cost < gbest_fit:
                    gbest_fit = cost
                    gbest_X = X[i].copy()
            
            loss_history.append(gbest_fit)
            # print(f"PSO Iter {t}: Best Score = {gbest_fit:.6f}")
            
        return gbest_X, gbest_fit, loss_history
