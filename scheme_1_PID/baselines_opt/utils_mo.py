import numpy as np

# Reuse utilities from MOSSA (assuming they are generic)
try:
    from MOSSA import fast_non_dominated_sort, calculate_crowding_distance
except ImportError:
    # Minimal fallback implementation if import fails
    def fast_non_dominated_sort(fitness):
        N, M = fitness.shape
        fronts = []
        domination_count = np.zeros(N, dtype=int)
        dominated_solutions = [[] for _ in range(N)]
        ranks = np.zeros(N, dtype=int)
        
        front0 = []
        for p in range(N):
            for q in range(N):
                # Check dominance: p dominates q
                diff = fitness[p] - fitness[q]
                if np.all(diff <= 0) and np.any(diff < 0):
                    dominated_solutions[p].append(q)
                elif np.all(diff >= 0) and np.any(diff > 0):
                    domination_count[p] += 1
            if domination_count[p] == 0:
                ranks[p] = 0
                front0.append(p)
        fronts.append(front0)
        
        i = 0
        while i < len(fronts) and len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in dominated_solutions[p]:
                    domination_count[q] -= 1
                    if domination_count[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            if len(next_front) > 0:
                fronts.append(next_front)
            i += 1
        return fronts, ranks

    def calculate_crowding_distance(fitness, front_indices):
        l = len(front_indices)
        dists = np.zeros(l)
        if l == 0: return dists
        M = fitness.shape[1]
        
        for m in range(M):
            # Sort by objective m
            sorted_idx = np.argsort(fitness[front_indices, m])
            # Boundary
            dists[sorted_idx[0]] = np.inf
            dists[sorted_idx[-1]] = np.inf
            
            f_min = fitness[front_indices[sorted_idx[0]], m]
            f_max = fitness[front_indices[sorted_idx[-1]], m]
            
            if f_max == f_min: continue
            
            for i in range(1, l-1):
                dists[sorted_idx[i]] += (fitness[front_indices[sorted_idx[i+1]], m] - 
                                         fitness[front_indices[sorted_idx[i-1]], m]) / (f_max - f_min)
        return dists

class MOOptimizer:
    def __init__(self, pop_size=50, dim=6, max_iter=100, lb=0, ub=10):
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
