import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import random
import math

# --- Requirement 1: Core Abstractions (DIP) ---

class EnvironmentPredictor(ABC):
    """
    Abstract Base Class for the Prediction Layer.
    Acts as a Black-box Oracle for the MPC Controller.
    """
    @abstractmethod
    def predict_next_state(self, current_state: np.ndarray, control_action: np.ndarray, external_disturbances: np.ndarray = None) -> np.ndarray:
        """
        Predicts the state at t+1 given state at t and action at t.
        
        Args:
            current_state: State vector at time t [T, H, CO2, ...]
            control_action: Control inputs at time t [u_heat, u_vent, u_co2, ...]
            external_disturbances: (Optional) External factors like outdoor temp, solar radiation.
            
        Returns:
            Predicted state vector at time t+1.
        """
        pass

class CostFunction(ABC):
    """
    Abstract Base Class for the Cost Function using in Optimization.
    """
    @abstractmethod
    def evaluate(self, predicted_states: List[np.ndarray], control_sequence: List[np.ndarray]) -> float:
        """
        Calculates the scalar cost J for a given trajectory.
        
        Args:
            predicted_states: List of state vectors from t+1 to t+N
            control_sequence: List of control actions from t to t+N-1
            
        Returns:
            Scalar cost value (lower is better).
        """
        pass

class OptimizationSolver(ABC):
    """
    Abstract Base Class for the Optimization Layer.
    """
    @abstractmethod
    def solve(self, cost_function: CostFunction, predictor: EnvironmentPredictor, current_state: np.ndarray, horizon: int, bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Finds the optimal control sequence.
        
        Args:
            cost_function: The objective function to minimize.
            predictor: The environment model for forward simulation.
            current_state: The starting state for the horizon.
            horizon: Prediction horizon N.
            bounds: Constraints for each control variable [(min, max), ...].
            
        Returns:
            Optimal control sequence [u_0, u_1, ..., u_{N-1}].
        """
        pass


# --- Requirement 2: Enhanced ISSA Implementation ---

class ISSAOptimizer(OptimizationSolver):
    """
    Improved Sparrow Search Algorithm (ISSA) Optimizer.
    Includes:
    1. Tent Chaos Mapping Initialization
    2. Adaptive Steps & Weights
    3. Cauchy Mutation
    """
    
    def __init__(self, pop_size=30, max_iter=50):
        self.pop_size = pop_size
        self.max_iter = max_iter
        
    def _tent_map_init(self, dim, bounds):
        """
        Tent Chaos Mapping for population initialization.
        x_{n+1} = 2x_n if x_n < 0.5 else 2(1-x_n)
        """
        population = []
        for _ in range(self.pop_size):
            individual = []
            # Generate a chaotic sequence for each dimension to ensure ergodic property
            for d in range(dim):
                lb, ub = bounds[d]
                x = random.random() # Random start
                # Iterate map a few times to lose memory of random start
                for _ in range(10): 
                    if x < 0.5: x = 2 * x
                    else: x = 2 * (1 - x)
                
                # Map back to [lb, ub]
                val = lb + x * (ub - lb)
                individual.append(val)
            population.append(np.array(individual))
        return population

    def _cauchy_mutation(self, best_position, bounds):
        """
        Cauchy Mutation to escape local optima.
        Cauchy distribution has heavier tails than Gaussian.
        """
        dim = len(best_position)
        new_position = best_position.copy()
        for d in range(dim):
            # Standard Cauchy generation
            cauchy_var = math.tan(math.pi * (random.random() - 0.5))
            new_position[d] += cauchy_var * new_position[d]
            
            # Clamp to bounds
            lb, ub = bounds[d % len(bounds)]
            new_position[d] = max(lb, min(ub, new_position[d]))
            
        return new_position
        
    def solve(self, cost_function: CostFunction, predictor: EnvironmentPredictor, current_state: np.ndarray, horizon: int, bounds: List[Tuple[float, float]]) -> List[np.ndarray]:
        """
        Executes the ISSA optimization loop.
        Note: The solution space dimension = horizon * num_control_vars.
        """
        num_controls = len(bounds)
        dim = horizon * num_controls
        
        # 1. Initialization (Tent Map)
        # We flatten the control sequence into a single 1D vector for optimization
        # vector = [u0_0, u0_1, ..., u1_0, u1_1, ...]
        population = self._tent_map_init(dim, bounds * horizon) 
        
        fitness_values = []
        best_fitness = float('inf')
        best_position = None
        
        # Initial evaluation
        for individual in population:
            # Reshape 1D individual back to sequence of control actions
            control_seq = individual.reshape((horizon, num_controls))
            
            # Simulation Step (Forward Pass)
            predicted_states = []
            temp_state = current_state.copy()
            for u in control_seq:
                temp_state = predictor.predict_next_state(temp_state, u)
                predicted_states.append(temp_state)
            
            fit = cost_function.evaluate(predicted_states, list(control_seq))
            fitness_values.append(fit)
            
            if fit < best_fitness:
                best_fitness = fit
                best_position = individual.copy()
                
        # Main Loop
        for iter in range(self.max_iter):
            # 2. Adaptive Weights (Non-linear decay)
            # w decreases from high to low to shift from exploration to exploitation
            omega = 0.9 - 0.5 * (iter / self.max_iter) ** 2
            
            # Sort population
            sorted_indices = np.argsort(fitness_values)
            population = [population[i] for i in sorted_indices]
            fitness_values = [fitness_values[i] for i in sorted_indices]
            
            # Update best
            if fitness_values[0] < best_fitness:
                best_fitness = fitness_values[0]
                best_position = population[0].copy()
                
            # Sparrow Search Logic (Simplified for brevity)
            # Producers (Discoverers) - Top 20%
            num_producers = int(self.pop_size * 0.2)
            for i in range(num_producers):
                # Update location based on simplified rule with adaptive weight
                if random.random() < 0.8:
                    population[i] = population[i] * math.exp(-i / (omega * self.max_iter))
                else:
                    population[i] = population[i] + np.random.normal() 
            
            # Scroungers (Followers)
            for i in range(num_producers, self.pop_size):
                if i > self.pop_size / 2:
                    # Worst ones go towards best
                    population[i] = np.random.normal() * np.exp((list(population[-1]) - list(population[i])) / (i**2))
                else:
                    # Follow producers
                    A = np.random.choice([-1, 1], size=dim) # Simplified matrix A
                    population[i] = best_position + np.abs(population[i] - best_position) * A * 0.5 # L interaction
                    
            # 3. Cauchy Mutation on Best Position
            # Give a chance to jump out of local optima
            if random.random() < 0.1: # 10% chance
                mutated_best = self._cauchy_mutation(best_position, bounds * horizon)
                
                # Evaluate mutated
                control_seq_mut = mutated_best.reshape((horizon, num_controls))
                pred_states_mut = []
                temp_state = current_state.copy()
                for u in control_seq_mut:
                    temp_state = predictor.predict_next_state(temp_state, u)
                    pred_states_mut.append(temp_state)
                
                fit_mut = cost_function.evaluate(pred_states_mut, list(control_seq_mut))
                
                if fit_mut < best_fitness:
                    best_fitness = fit_mut
                    best_position = mutated_best
                    population[0] = mutated_best # Replace current best
                    fitness_values[0] = best_fitness

            # Clamp all to bounds
            for i in range(self.pop_size):
                for d in range(dim):
                    lb, ub = (bounds * horizon)[d]
                    population[i][d] = max(lb, min(ub, population[i][d]))
                    
        return best_position.reshape((horizon, num_controls))


# --- Requirement 3: MPC Controller (The "Glue") ---

class MPCController:
    def __init__(self, predictor: EnvironmentPredictor, optimizer: OptimizationSolver, cost_func: CostFunction, horizon: int, control_bounds: List[Tuple[float, float]]):
        self.predictor = predictor
        self.optimizer = optimizer
        self.cost_func = cost_func
        self.horizon = horizon
        self.control_bounds = control_bounds
        
    def get_next_action(self, current_state: np.ndarray) -> np.ndarray:
        """
        The Rolling Horizon process:
        1. Measure state (passed as input).
        2. Optimize future sequence.
        3. Return first action.
        """
        print(f"\n[MPC] Starting optimization for state: {current_state}")
        
        # Run Optimization
        optimal_sequence = self.optimizer.solve(
            cost_function=self.cost_func,
            predictor=self.predictor,
            current_state=current_state,
            horizon=self.horizon,
            bounds=self.control_bounds
        )
        
        # Extract first action (Receding Horizon)
        u_star_0 = optimal_sequence[0]
        
        return u_star_0


# --- Requirement 4: Mock Integration ---

class MockDeepLearningPredictor(EnvironmentPredictor):
    """
    Mock implementation of a deep learning predictor.
    Model: T_next = T_curr + 0.2 * u_heat - 0.1 * u_vent + noise
           Target T is around 25.0
    """
    def predict_next_state(self, current_state: np.ndarray, control_action: np.ndarray, external_disturbances=None) -> np.ndarray:
        # State: [Temperature]
        # Action: [u_heat (0-1), u_vent (0-1)]
        
        temp = current_state[0]
        u_heat = control_action[0]
        u_vent = control_action[1]
        
        # Simple physics-like logic
        # Heat increases temp, Vent decreases temp
        delta_temp = 0.5 * u_heat - 0.3 * u_vent
        
        # Natural loss (cooling down towards 15 degrees)
        natural_loss = -0.05 * (temp - 15.0)
        
        next_temp = temp + delta_temp + natural_loss
        
        return np.array([next_temp])

class SimpleGreenhouseCost(CostFunction):
    def evaluate(self, predicted_states: List[np.ndarray], control_sequence: List[np.ndarray]) -> float:
        total_cost = 0
        target_temp = 25.0
        
        for i, state in enumerate(predicted_states):
            temp = state[0]
            action = control_sequence[i]
            
            # 1. Tracking Error Cost (Crop Discomfort)
            error_cost = (temp - target_temp) ** 2
            
            # 2. Economic Cost (Energy)
            # Heating is expensive, venting is cheap
            energy_cost = 0.5 * action[0] + 0.1 * action[1] 
            
            total_cost += error_cost + energy_cost
            
        return total_cost

# --- Main Simulation Loop ---

if __name__ == "__main__":
    # Setup
    predictor = MockDeepLearningPredictor()
    optimizer = ISSAOptimizer(pop_size=20, max_iter=30)
    cost_func = SimpleGreenhouseCost()
    
    # 2 Control Inputs: [Heater (0-1), Vent (0-1)]
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    horizon = 5 # Look 5 steps ahead
    
    mpc = MPCController(predictor, optimizer, cost_func, horizon, bounds)
    
    # Initialize Environment State
    current_sim_state = np.array([18.0]) # Start cold
    
    print("--- Starting MPC Simulation ---\n")
    print(f"Initial State: {current_sim_state}")
    
    # Simulation Loop
    for t in range(10):
        # 1. Get controls from MPC
        action = mpc.get_next_action(current_sim_state)
        
        print(f"Time {t}: Action Applied [Heat={action[0]:.2f}, Vent={action[1]:.2f}]")
        
        # 2. Apply action to 'Real' Environment
        # (Here we use the same predictor as the environment for simplicity, but in reality this is the physical system)
        current_sim_state = predictor.predict_next_state(current_sim_state, action)
        
        print(f"      -> New State: {current_sim_state}")

    print("\n--- Simulation Complete ---")
