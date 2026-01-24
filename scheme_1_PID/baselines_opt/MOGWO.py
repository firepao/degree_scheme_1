import numpy as np
from .utils_mo import MOOptimizer

class MOGWO(MOOptimizer):
    def optimize(self, fitness_func):
        # Grey Wolf Optimization for Multi-Objective
        # Simplified: Alpha, Beta, Delta are selected from Archive
        
        X = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([fitness_func(x) for x in X])
        
        archive_X = list(X)
        archive_fit = list(fitness)
        
        for t in range(self.max_iter):
            # Update Archive
            # (In simplified version, archive accumulates non-dominated)
            
            # Select Leaders from Archive
            # Sort Archive by simple metrics or random
            if len(archive_X) < 3:
                # Not enough solutions, use what we have or random
                leaders = [np.random.choice(len(X)) for _ in range(3)]
                Alpha_pos = X[leaders[0]]
                Beta_pos = X[leaders[1]]
                Delta_pos = X[leaders[2]]
            else:
                # Use Crowding Distance to pick diverse leaders? Or simple random
                # Randomly pick 3 from archive
                indices = np.random.choice(len(archive_X), 3, replace=False)
                Alpha_pos = archive_X[indices[0]]
                Beta_pos = archive_X[indices[1]]
                Delta_pos = archive_X[indices[2]]
            
            a = 2 - t * (2 / self.max_iter) 
            
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(), np.random.rand()
                A1 = 2*a*r1 - a
                C1 = 2*r2
                D_alpha = abs(C1*Alpha_pos - X[i])
                X1 = Alpha_pos - A1*D_alpha
                
                r1, r2 = np.random.rand(), np.random.rand()
                A2 = 2*a*r1 - a
                C2 = 2*r2
                D_beta = abs(C2*Beta_pos - X[i])
                X2 = Beta_pos - A2*D_beta
                
                r1, r2 = np.random.rand(), np.random.rand()
                A3 = 2*a*r1 - a
                C3 = 2*r2
                D_delta = abs(C3*Delta_pos - X[i])
                X3 = Delta_pos - A3*D_delta
                
                X[i] = (X1 + X2 + X3) / 3
                X[i] = np.clip(X[i], self.lb, self.ub)
                
                new_f = fitness_func(X[i])
                fitness[i] = new_f
                
                archive_X.append(X[i].copy())
                archive_fit.append(new_f)
        
        # Prune
        pareto_mask = self._get_pareto_mask(np.array(archive_fit))
        return np.array(archive_X)[pareto_mask], np.array(archive_fit)[pareto_mask]

    def _get_pareto_mask(self, fitness):
        # Same as MOPSO
        is_pareto = np.ones(fitness.shape[0], dtype=bool)
        # Limit N for check
        if len(fitness) > 500: # subsample if too large
            indices = np.random.choice(len(fitness), 500, replace=False)
            sub_fit = fitness[indices]
            # ... simpler to strictly check
        
        for i in range(fitness.shape[0]):
            for j in range(fitness.shape[0]):
                if i != j and (np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i])):
                    is_pareto[i] = False
                    break
        return is_pareto
