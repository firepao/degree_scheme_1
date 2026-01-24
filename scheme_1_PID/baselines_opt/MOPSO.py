import numpy as np
from .utils_mo import MOOptimizer

class MOPSO(MOOptimizer):
    def optimize(self, fitness_func):
        # 1. Init
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        V = np.zeros_like(X)
        fitness = np.array([fitness_func(x) for x in X])
        
        pbest_X = X.copy()
        pbest_fit = fitness.copy()
        
        # Simple Archive: Store non-dominated solutions found so far
        archive_X = list(X)
        archive_fit = list(fitness)
        
        w, c1, c2 = 0.5, 1.5, 1.5
        
        for t in range(self.max_iter):
            # Update Archive
            self._update_archive(archive_X, archive_fit)
            
            for i in range(self.pop_size):
                # Select Leader (gbest) random from top percentage of archive
                if len(archive_X) > 0:
                    leader_idx = np.random.randint(0, len(archive_X))
                    gbest = archive_X[leader_idx]
                else:
                    gbest = pbest_X[i] # Fallback
                
                # Update V, X
                r1, r2 = np.random.rand(), np.random.rand()
                V[i] = w*V[i] + c1*r1*(pbest_X[i] - X[i]) + c2*r2*(gbest - X[i])
                X[i] = np.clip(X[i] + V[i], self.lb, self.ub)
                
                # Eval
                new_f = fitness_func(X[i])
                fitness[i] = new_f
                
                # Update Pbest (Domination check)
                if self._dominates(new_f, pbest_fit[i]):
                    pbest_X[i] = X[i].copy()
                    pbest_fit[i] = new_f
                elif not self._dominates(pbest_fit[i], new_f):
                    if np.random.rand() < 0.5:
                        pbest_X[i] = X[i].copy()
                        pbest_fit[i] = new_f
                
                archive_X.append(X[i].copy())
                archive_fit.append(new_f)
                
        # Final Archive Pruning
        pareto_mask = self._get_pareto_mask(np.array(archive_fit))
        return np.array(archive_X)[pareto_mask], np.array(archive_fit)[pareto_mask]

    def _update_archive(self, arch_X, arch_F):
        # Keep manageable size, remove dominated
        # This is simplified. Real MOPSO uses grid.
        if len(arch_X) > 200: # Limit size
            # Random prune or Keep best? Simplified: Random keep
            indices = np.random.choice(len(arch_X), 100, replace=False)
            # Reconstruct list (inefficient but safe for simple demo)
            new_X = [arch_X[i] for i in indices]
            new_F = [arch_F[i] for i in indices]
            arch_X[:] = new_X
            arch_F[:] = new_F

    def _dominates(self, f1, f2):
        return np.all(f1 <= f2) and np.any(f1 < f2)
    
    def _get_pareto_mask(self, fitness):
        is_efficient = np.ones(fitness.shape[0], dtype=bool)
        for i, c in enumerate(fitness):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(fitness[is_efficient] < c, axis=1)  # Keep if better
                is_efficient[i] = True  # Keep self
        # Re-check properly O(N^2)
        is_pareto = np.ones(fitness.shape[0], dtype=bool)
        for i in range(fitness.shape[0]):
            for j in range(fitness.shape[0]):
                if i != j and self._dominates(fitness[j], fitness[i]):
                    is_pareto[i] = False
                    break
        return is_pareto
