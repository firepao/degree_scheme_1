import numpy as np
import math
import random

def calculate_2k_crowding_distance_entropy(fitness, k=2, epsilon=1e-10):
    """
    计算2k拥挤距离熵 (2k Crowding-Distance Entropy)
    
    参数:
        fitness: shape (N, M) 的适应度矩阵，N为个体数，M为目标数
        k: 邻居对数，默认为2
        epsilon: 防止除零的小常数
    
    返回:
        cde_values: shape (N,) 的熵值数组
    """
    N, M = fitness.shape
    cde_values = np.zeros(N)
    
    # 既然是基于非支配排序后的层级，这里假设传入的fitness是同一个Front里的
    # 如果只有1个或2个点，直接设为无穷大
    if N <= 2:
        return np.full(N, np.inf)
    
    for r in range(M):
        # 对第r个目标进行排序
        sorted_indices = np.argsort(fitness[:, r])
        # 获取排序后的目标值
        obj_values = fitness[sorted_indices, r]
        
        # 边界点设为无穷大
        cde_values[sorted_indices[0]] = np.inf
        cde_values[sorted_indices[-1]] = np.inf
        
        # 计算最大最小值差，用于归一化（虽然公式里主要是为了相对距离，但通常CD会归一化，
        # 不过论文公式(7)(8)用的是局部距离，不需要全局归一化，直接计算相对比率）
        
        for i in range(1, N - 1):
            # 如果已经被标记为无穷大（比如在其他目标上是边界），则跳过或累加无穷大（通常保持无穷大）
            if np.isinf(cde_values[sorted_indices[i]]):
                continue
                
            sum_entropy = 0.0
            
            for s in range(1, k + 1):
                # 检查索引边界
                idx_prev = i - s
                idx_next = i + s
                
                # 如果超出边界，论文中说 "Make sure to always select boundary points"，
                # 这里我们只计算有效范围内的。如果k太大导致越界，就只算能算的部分，
                # 或者严格按照如果有越界就把它当做边界处理？
                # 简单起见，如果越界，就认为它是边界的一种扩展，或者仅计算可用部分。
                # 但根据Standard CD，越界意味着它是边界。
                # 为了鲁棒性，如果越界，我们停止当前s的循环或不做累加（因为它不够k对邻居）
                # 但通常 N >> k。
                
                if idx_prev < 0 or idx_next >= N:
                    continue
                
                # 获取值 (F(i+s) 和 F(i-s))，注意obj_values是排序后的
                f_next = obj_values[idx_next]
                f_prev = obj_values[idx_prev]
                f_curr = obj_values[i]
                
                # 分母: |F(i+s) - F(i-s)|
                denom = abs(f_next - f_prev) + epsilon
                
                # 相对距离 d_{i, i-s} 和 d_{i, i+s} (Eq 7, 8)
                # 注意：排序后 obj_values 是递增的（假设是最小化问题，或者仅仅是排序）
                # 绝对值处理
                d_prev = abs(f_curr - f_prev) / denom
                d_next = abs(f_next - f_curr) / denom # 公式里是 |f(i+s) - f(i)|
                
                # 熵项 (Eq 9 的部分)
                # E = -(d_prev * ln(d_prev) + d_next * ln(d_next))
                # 论文Eq 10/11: CDE += |f_next - f_prev| * E
                # 也就是 CDE += |f_next - f_prev| * (-1) * (...)
                
                term_prev = d_prev * np.log(d_prev + epsilon)
                term_next = d_next * np.log(d_next + epsilon)
                entropy = -(term_prev + term_next)
                
                # 距离权重
                dist_weight = abs(f_next - f_prev)
                
                sum_entropy += dist_weight * entropy
            
            # 累加到对应的个体 (Eq 11: 1/k * sum)
            cde_values[sorted_indices[i]] += sum_entropy / k
            
    return cde_values

def fast_non_dominated_sort(fitness):
    """
    快速非支配排序
    """
    N, M = fitness.shape
    fronts = []
    domination_count = np.zeros(N, dtype=int)
    dominated_solutions = [[] for _ in range(N)]
    rank = np.zeros(N, dtype=int)
    
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            # 判断 p 是否支配 q
            # 最小化问题：p支配q 意味着 p的所有目标 <= q的所有目标，且至少有一个 <
            better = fitness[p] <= fitness[q]
            strictly_better = fitness[p] < fitness[q]
            
            if np.all(better) and np.any(strictly_better):
                dominated_solutions[p].append(q)
            elif np.all(fitness[q] <= fitness[p]) and np.any(fitness[q] < fitness[p]):
                domination_count[p] += 1
                
    first_front = []
    for p in range(N):
        if domination_count[p] == 0:
            rank[p] = 0
            first_front.append(p)
    
    fronts.append(first_front)
    
    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        fronts.append(next_front)
        i += 1
        
    return fronts[:-1], rank # 最后一个是空的

class MOSSA:
    def __init__(self, obj_func, n_dim, lb, ub, pop_size=100, max_iter=100, n_obj=2):
        self.obj_func = obj_func      # 目标函数，返回 [f1, f2, ...]
        self.n_dim = n_dim            # 变量维度
        self.lb = np.array(lb)        # 下界
        self.ub = np.array(ub)        # 上界
        self.pop_size = pop_size      # 种群大小
        self.max_iter = max_iter      # 最大迭代次数
        self.n_obj = n_obj            # 目标数量
        
        # 初始化位置
        self.X = self.lb + (self.ub - self.lb) * np.random.rand(self.pop_size, self.n_dim)
        self.fitness = np.zeros((self.pop_size, self.n_obj))
        
        # 计算初始适应度
        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.X[i])
            
        # 初始存档（即当前种群）
        self.archive_X = self.X.copy()
        self.archive_fitness = self.fitness.copy()
        
    def check_bounds(self, x):
        return np.maximum(self.lb, np.minimum(x, self.ub))
    
    def update_positions(self, t):
        """
        更新种群位置
        """
        # 计算自适应参数
        # ST: Eq 15
        ST_min, ST_max = 0.5, 1.0 # 假设值，论文未给出具体数值，通常ST在SSA中是0.8，这里设为自适应
        ST = (ST_max + ST_min) / 2 + (ST_max - ST_min) / 2 * np.tanh(-4 + 8 * t / self.max_iter)

        # pE: Eq 16
        pE_min, pE_max = 0.1, 0.3 # 假设范围，标准SSA是0.2
        pE_val = (pE_max + pE_min) / 2 + (pE_max - pE_min) / 2 * np.tanh(-4 + 8 * t / self.max_iter)
        
        # pD: Eq 19
        pD_min, pD_max = 0.1, 0.3
        pD_val = (pD_max + pD_min) / 2 + (pD_max - pD_min) / 2 * np.tanh(-4 + 8 * (self.max_iter - t) / self.max_iter)
        
        m = self.pop_size
        m_e = int(m * pE_val)
        m_d = int(m * pD_val)
        
        # 根据适应度对种群进行简单的排序来确定发现者和追随者
        # 对于多目标，这里需要一个简单的标准。通常使用非支配排序后的Rank，或者拥挤距离。
        # 为了简单且符合SSA逻辑，我们使用目前的Archive策略里的排序结果。
        # 也就是 self.X 应该是已经排好序的（因为每轮末尾都会重组）
        # 所以直接取前 m_e 个作为 Explorers
        
        X_new = np.zeros_like(self.X)
        
        # 当前最优和最差 (在 Archive/Sort 后，index 0 是最好， index -1 是最差)
        X_best = self.X[0]
        X_worst = self.X[-1]
        
        # 1. 更新 Explorers (Eq 13)
        for i in range(m_e):
            R2 = np.random.rand()
            if R2 < ST:
                # 论文 Eq 13: X * (1 +/- exp(...))
                # +/- 是随机的吗？通常是。
                sign = 1 if np.random.rand() > 0.5 else -1
                X_new[i] = self.X[i] * (1 + sign * np.exp(-t / (np.random.rand() * self.max_iter)))
            else:
                Q = np.random.randn() # 正态分布
                # L 是一行全1，这里直接广播
                X_new[i] = self.X[i] + Q # * L (Broadcasting)
        
        # 2. 更新 Followers (Eq 17)
        # 分割点
        split_idx = m_e + (m - m_e) // 2
        
        for i in range(m_e, m):
            if i <= split_idx:
                # Eq 17 part 1
                # A+ calculation: A is 1xdim with 1 or -1. A+ = A^T(AA^T)^-1. 
                # 对于 1D A， A A^T 是 scalar (dim). (AA^T)^-1 is 1/dim.
                # A^T (AA^T)^-1 = A^T / dim.
                # 所以 A+ * L 实际上就是 A vector / dim.
                A = np.random.choice([1, -1], size=self.n_dim)
                A_plus = A / self.n_dim # 简化计算
                
                # |X_i - X_best| * A+ * L
                # 注意这里是矩阵乘法概念，但在element-wise实现时：
                dist = np.abs(self.X[i] - X_best)
                # 每一个维度 d: dist[d] * A_plus[d] ? 
                # 标准SSA代码实现通常是 element-wise multiply
                X_new[i] = X_best + dist * A_plus 
            else:
                # Eq 17 part 2
                Q = np.random.randn()
                # exp fraction
                denom = (i ** 2) if i != 0 else 1 # 论文写的是 i^2
                term = np.exp((X_worst - self.X[i]) / denom)
                X_new[i] = Q * term # 这里可能要是 X_new[i] = term? 不，公式是 Q * exp(...)

        # 3. 更新 Defenders (Eq 18)
        # 随机选择 m_d 个个体作为 defenders
        defender_indices = np.random.choice(m, m_d, replace=False)
        # 我们需要先计算X_new里的defenders，也就是原来的X对应位置变了，还是在X_new基础上变？
        # SSA通常是一次性更新。defenders是在初始位置上更新。
        # 这里为了不覆盖之前的更新（Explorer/Follower），我们应该把defender逻辑应用到那些被选中的人身上。
        # 如果被选中者是 Explorer，它就按 Defender 逻辑动。
        # 简单起见，按标准代码逻辑，defender循环覆盖之前的 update。
        
        # 计算F(X_i)用于公式18。注意是旧的 fitness
        
        for i in defender_indices:
            u = np.random.rand()
            if u >= 0.5:
                # Eq 18 part 1
                # sum over objectives
                sum_term = 0
                for r in range(self.n_obj):
                    num = np.abs(self.X[i] - X_worst) # Vector
                    den = (self.fitness[i, r] - self.fitness[-1, r]) + 1e-10 # scalar
                    sum_term += num / den
                
                kappa = np.random.uniform(-1, 1)
                X_new[i] = self.X[i] + (kappa / self.n_obj) * sum_term
            else:
                # Eq 18 part 2
                beta = np.random.randn() # normal (0, 1)
                X_new[i] = X_best + beta * np.abs(self.X[i] - X_best)

        # 边界检查
        self.X = self.check_bounds(X_new)
        
    def evaluate(self):
        for i in range(self.pop_size):
            self.fitness[i] = self.obj_func(self.X[i])

    def optimal_strategy(self):
        """
        位置最优策略：合并，排序，截断
        """
        # 合并 Pop 和 Archive
        combined_X = np.vstack((self.X, self.archive_X))
        combined_fitness = np.vstack((self.fitness, self.archive_fitness))
        
        # 去重 (可选，防止过多重复解)
        # combined_X, unique_indices = np.unique(combined_X, axis=0, return_index=True)
        # combined_fitness = combined_fitness[unique_indices]
        
        N = combined_X.shape[0]
        
        # 1. 快速非支配排序
        fronts, rank = fast_non_dominated_sort(combined_fitness)
        
        # 2. 构建新的 Archive
        new_X = []
        new_fitness = []
        
        # 依次加入 Front
        for front in fronts:
            if len(new_X) + len(front) <= self.pop_size:
                # 整个 Front 都能加入
                for idx in front:
                    new_X.append(combined_X[idx])
                    new_fitness.append(combined_fitness[idx])
            else:
                # 需要截断，使用 2k CDE 排序
                # 提取当前 Front 的 fitness
                front_indices = np.array(front)
                front_fit = combined_fitness[front_indices]
                
                # 计算 CDE
                cde_values = calculate_2k_crowding_distance_entropy(front_fit, k=2)
                
                # 降序排列 (CDE 越大越好)
                sorted_front_indices = np.argsort(cde_values)[::-1]
                
                num_needed = self.pop_size - len(new_X)
                selected_indices = front_indices[sorted_front_indices[:num_needed]]
                
                for idx in selected_indices:
                    new_X.append(combined_X[idx])
                    new_fitness.append(combined_fitness[idx])
                
                break # 填满了
                
        self.X = np.array(new_X)
        self.fitness = np.array(new_fitness)
        
        # 更新 Archive (在本算法逻辑中，Archive 和下一代 Pop 是同一个东西)
        self.archive_X = self.X.copy()
        self.archive_fitness = self.fitness.copy()

    def run(self):
        for t in range(2, self.max_iter + 1):
            # 1. Update Positions
            self.update_positions(t)
            
            # 2. Evaluate
            self.evaluate()
            
            # 3. Optimal Strategy (Sort & Update)
            self.optimal_strategy()
            
            if t % 10 == 0:
                print(f"Iter {t}/{self.max_iter} completed.")
                
        # 返回 Pareto 前沿 (Rank 0)
        # 因为每轮最后都做了一次 Sort，且 Archive 就是 self.X (或其子集)
        # 我们可以再做一次非支配排序只返回 Rank 0
        fronts, rank = fast_non_dominated_sort(self.fitness)
        pareto_front_indices = fronts[0]
        
        return self.X[pareto_front_indices], self.fitness[pareto_front_indices]

