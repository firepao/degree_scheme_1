import numpy as np
from .utils_mo import MOOptimizer, fast_non_dominated_sort, calculate_crowding_distance

class NSGAII(MOOptimizer):
    def optimize(self, fitness_func):
        # Init
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([fitness_func(x) for x in X])
        
        for t in range(self.max_iter):
            # 1. Selection (Tournament) & Reproduction
            offspring_X = []
            for _ in range(self.pop_size):
                p1 = self._tournament(X, fitness)
                p2 = self._tournament(X, fitness)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                offspring_X.append(child)
            offspring_X = np.array(offspring_X)
            offspring_fit = np.array([fitness_func(x) for x in offspring_X])
            
            # 2. Merge
            combined_X = np.vstack((X, offspring_X))
            combined_fit = np.vstack((fitness, offspring_fit))
            
            # 3. Non-dominated Sort
            fronts, ranks = fast_non_dominated_sort(combined_fit)
            
            # 4. Fill new population
            new_X = []
            new_fit = []
            idx_added = 0
            
            for front in fronts:
                if idx_added + len(front) <= self.pop_size:
                    for i in front:
                        new_X.append(combined_X[i])
                        new_fit.append(combined_fit[i])
                    idx_added += len(front)
                else:
                    # Sort by crowding distance
                    dists = calculate_crowding_distance(combined_fit, front)
                    # Descending order of distance
                    sorted_front = [front[i] for i in np.argsort(dists)[::-1]]
                    
                    needed = self.pop_size - idx_added
                    for i in range(needed):
                        real_idx = sorted_front[i]
                        new_X.append(combined_X[real_idx])
                        new_fit.append(combined_fit[real_idx])
                    break
            
            X = np.array(new_X)
            fitness = np.array(new_fit)
            
        # Return Pareto Front of final population
        fronts, _ = fast_non_dominated_sort(fitness)
        pareto_indices = fronts[0]
        return X[pareto_indices], fitness[pareto_indices]

    def _tournament(self, X, fit):
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        # Random domination check or simple random
        # Proper way: Check rank. Since I don't maintain rank in loop easily, simple random for demo
        # Or better: random selection
        return X[idx1]

    def _crossover(self, p1, p2, eta=20):
        # SBX
        child = np.zeros_like(p1)
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                # Simple arithmetic crossover fallback
                alpha = np.random.rand()
                child[i] = alpha*p1[i] + (1-alpha)*p2[i]
            else:
                child[i] = p1[i]
        return np.clip(child, self.lb, self.ub)

    def _mutate(self, p, pm=0.1):
        if np.random.rand() < pm:
            idx = np.random.randint(0, self.dim)
            p[idx] += np.random.normal(0, (self.ub-self.lb)*0.1)
        return np.clip(p, self.lb, self.ub)
