# src/adaptive_quantum_cvrp/rl/environment.py

"""
By accepting a generic SubproblemSolver, it can be used for both the classical and quantum versions of the ALM without any code changes. 
"""


import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..common.cvrp_instance import CVRPInstance
from ..common.cvrp_solution import CVRPSolution
from ..common.evaluator import CVRPEvaluator
from ..alm.classical_solver import SubproblemSolver

class ALMPenaltyEnv(gym.Env):
    """
    A Gymnasium environment for learning the ALM penalty parameter.

    The agent's goal is to select the penalty parameter `mu` at each step
    of the ALM optimization to minimize the final solution cost.
    """
    def __init__(self, instance: CVRPInstance, solver: SubproblemSolver, max_alm_steps: int):
        super().__init__()
        self.instance = instance
        self.solver = solver
        self.max_alm_steps = max_alm_steps
        
        
        # Action: A single continuous value for the penalty `mu`
        self.action_space = spaces.Box(low=0.1, high=10.0, shape=(1,), dtype=np.float32)

        # Observation: [num_customers, capacity, avg_demand, constraint_violations]
        self.observation_space = spaces.Box(
            low=0, high=1.0, shape=(5,), dtype=np.float32 # Shape is now 5
        )
        
        self.current_step = 0
        self.lagrange_multipliers = np.zeros(self.instance.num_customers)
        self.best_feasible_cost = float('inf')

    def _get_obs(self, solution: CVRPSolution = None) -> np.ndarray:
        """Constructs the observation from the current state."""
        violations = 0
        if solution:
            visits = np.zeros(self.instance.num_customers)
            for route in solution.routes:
                for customer in route:
                    visits[customer - 1] += 1
            # Sum of absolute violations
            violations = np.sum(np.abs(visits - 1))
        
        avg_demand = np.mean(self.instance.demands[1:])

        # --- NORMALIZATION ---
        # Scale values to a smaller range (mostly [0, 1]) to ensure stability.
        # These denominators are estimates; they don't have to be perfect.
        norm_num_vehicles = self.instance.num_vehicles / 20.0 # Normalize by a reasonable max

        norm_num_customers = self.instance.num_customers / 100.0
        norm_capacity = self.instance.capacity / 10000.0
        norm_avg_demand = avg_demand / self.instance.capacity if self.instance.capacity > 0 else 0
        norm_violations = violations / self.instance.num_customers if self.instance.num_customers > 0 else 0
        


        return np.array([
            norm_num_customers,
            norm_capacity,
            norm_avg_demand,
            norm_violations,
            norm_num_vehicles 
        ], dtype=np.float32)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.lagrange_multipliers = np.zeros(self.instance.num_customers)
        self.best_feasible_cost = float('inf')
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Executes one time step within the environment."""
        penalty_mu = action[0]
        self.current_step += 1
        
        # 1. Solve the subproblem with the chosen penalty
        solution = self.solver.solve(
            self.instance, self.lagrange_multipliers, penalty_mu
        )
        
        # 2. Calculate cost and feasibility
        cost = CVRPEvaluator.get_solution_cost(self.instance, solution)
        is_feasible, _ = CVRPEvaluator.check_feasibility(self.instance, solution)

        # --- Report back the best solution found ---
        info = {}
        if is_feasible and cost < self.best_feasible_cost:
            self.best_feasible_cost = cost
            info["solution"] = solution
            info["cost"] = cost
        

        # 3. Calculate reward
        # Reward is negative cost, with a large penalty for infeasibility
        reward = -cost
        if not is_feasible:
            reward -= 1000  # Penalty for infeasibility

        # 4. Update Lagrange multipliers
        visits = np.zeros(self.instance.num_customers)
        for route in solution.routes:
            for customer in route:
                visits[customer - 1] += 1
        violations = visits - 1
        self.lagrange_multipliers += (1 / penalty_mu) * violations

        # 5. Check for termination
        terminated = self.current_step >= self.max_alm_steps
        
        obs = self._get_obs(solution)
        return obs, reward, terminated, False, info