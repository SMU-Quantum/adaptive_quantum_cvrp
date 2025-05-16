# src/rl_classical/alm_penalty_env_qubo_vqe.py

import gymnasium as gym # Or `import gym` if you are using the older Gym API
from gymnasium import spaces # Or `gym.spaces`
import time
from typing import Tuple
import numpy as np
import random
from typing import List, Dict, Optional, Any

from src.common.cvrp_instance import CVRPInstance, load_cvrp_instance
# Import your quantum ALM optimizer
from src.alm.alm_optimizer_quantum import AlmOptimizerQuantum 

# --- Constants (these should align with your penalty_networks.py and SAC agent) ---
# Define the ranges for the penalty parameters your RL agent will output
# These might need to be tuned based on experimentation.
MIN_PENALTY_RHO = 0.1  # Min for customer visit ALM penalty (rho)
MAX_PENALTY_RHO = 1000.0
MIN_PENALTY_SIGMA = 0.1 # Min for capacity ALM penalty (sigma)
MAX_PENALTY_SIGMA = 1000.0

# Number of features describing a CVRP instance (must match your network input)
# Example features: num_customers, capacity_to_total_demand_ratio, avg_demand,
# std_dev_demand, avg_dist_to_depot, density.
# KEEP THIS CONSISTENT WITH YOUR CLASSICAL AlmPenaltyEnv.py
N_INSTANCE_FEATURES = 6 # Example, adjust to your actual feature count

class AlmPenaltyEnvQuBoVqe(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, 
                 instance_files: List[str], 
                 alm_config: Dict[str, Any],      # For fixed ALM params (max_iter, factors, etc.)
                 quantum_solver_config: Dict[str, Any], # For quantum solver params
                 max_instance_dimension: Optional[int] = None, # Max dimension (nodes) for instances
                 max_steps_per_episode: int = 1 # Each episode is one ALM solve
                ):
        super(AlmPenaltyEnvQuBoVqe, self).__init__()

        self.instance_files = [f for f in instance_files if f.endswith(".vrp")]
        if not self.instance_files:
            raise ValueError("No .vrp instance files provided or found.")
            
        self.alm_config = alm_config
        self.quantum_solver_config = quantum_solver_config # Store this
        self.max_instance_dimension = max_instance_dimension
        
        self.current_instance: Optional[CVRPInstance] = None
        self.current_instance_path: Optional[str] = None
        self._load_random_instance() # Load an initial instance

        # Action space: 2 continuous values for penalty parameters (rho_visit, sigma_capacity)
        # These are scaled actions; the agent outputs values typically in [-1, 1]
        # which are then mapped to [MIN_PENALTY, MAX_PENALTY]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation space: Features of the CVRP instance
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_INSTANCE_FEATURES,), dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps_per_episode = max_steps_per_episode

    def _scale_action(self, action: np.ndarray) -> Tuple[float, float]:
        """Scales normalized actions [-1, 1] to actual penalty ranges."""
        # Action 0: rho (customer visit penalty)
        # Action 1: sigma (capacity penalty)
        
        # Denormalize from [-1, 1] to [0, 1]
        action_zero_to_one = (action + 1.0) / 2.0
        
        # Logarithmic scaling can be useful for parameters spanning orders of magnitude
        # Or linear scaling if preferred. Let's use linear for simplicity first.
        scaled_rho = MIN_PENALTY_RHO + action_zero_to_one[0] * (MAX_PENALTY_RHO - MIN_PENALTY_RHO)
        scaled_sigma = MIN_PENALTY_SIGMA + action_zero_to_one[1] * (MAX_PENALTY_SIGMA - MIN_PENALTY_SIGMA)
        
        # Ensure bounds are strictly met after scaling
        scaled_rho = np.clip(scaled_rho, MIN_PENALTY_RHO, MAX_PENALTY_RHO)
        scaled_sigma = np.clip(scaled_sigma, MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA)
        
        return float(scaled_rho), float(scaled_sigma)

    def _get_instance_features(self, instance: CVRPInstance) -> np.ndarray:
        """
        Extracts features from the CVRP instance.
        Ensure this is IDENTICAL to the feature extraction in your classical AlmPenaltyEnv.py.
        The quantum nature of the solver doesn't change what the instance *is*.
        """
        if instance is None:
            return np.zeros(N_INSTANCE_FEATURES, dtype=np.float32)

        num_nodes = instance.dimension
        num_customers = num_nodes -1 # Assuming 1 depot
        
        if num_customers <= 0: # Handle empty or depot-only instance
            return np.array([0,0,0,0,0,0], dtype=np.float32) # Example: all zeros

        demands_array = np.array([instance.demands[i] for i in range(num_nodes) if i != instance.depot])
        total_demand = np.sum(demands_array)
        
        feature1 = float(num_customers) / 50.0 # Normalize num_customers (e.g., by typical max)
        feature2 = float(instance.capacity) / (total_demand + 1e-6) # Capacity to total demand ratio
        feature3 = np.mean(demands_array) / float(instance.capacity + 1e-6) # Avg demand to capacity
        feature4 = np.std(demands_array) / float(instance.capacity + 1e-6) # Std dev demand to capacity

        # Example distance-based features (can be computationally intensive if not precalculated)
        # For simplicity, let's use placeholders or simpler aggregate distance stats if available
        # If instance.coords exist:
        # depot_coords = instance.coords[instance.depot]
        # customer_coords = np.array([instance.coords[i] for i in range(num_nodes) if i != instance.depot])
        # avg_dist_to_depot = np.mean(np.linalg.norm(customer_coords - depot_coords, axis=1)) if num_customers >0 else 0
        # feature5 = avg_dist_to_depot / 100.0 # Normalize (e.g. by typical max distance)
        # feature6 = placeholder for density or other structural feature
        # For now, using placeholders:
        feature5 = 0.0 # Placeholder for avg_dist_to_depot
        feature6 = 0.0 # Placeholder for density

        features = np.array([feature1, feature2, feature3, feature4, feature5, feature6], dtype=np.float32)
        # Ensure features are clipped or scaled to prevent extreme values if necessary
        return np.nan_to_num(features, nan=0.0, posinf=1e5, neginf=-1e5)


    def _load_random_instance(self):
        """Loads a random instance from the list and filters by dimension if needed."""
        instance_path = random.choice(self.instance_files)
        instance = load_cvrp_instance(instance_path)
        
        if self.max_instance_dimension is not None:
            # Keep trying until a suitable instance is found (can be slow if many are too large)
            # Or, pre-filter instance_files in __init__
            attempts = 0
            while instance.dimension > self.max_instance_dimension and attempts < len(self.instance_files) * 2:
                instance_path = random.choice(self.instance_files)
                instance = load_cvrp_instance(instance_path)
                attempts +=1
            if instance.dimension > self.max_instance_dimension:
                print(f"[Warning] Could not find instance within dim {self.max_instance_dimension} after {attempts} tries. Using last loaded: {instance.name} (dim {instance.dimension})")

        self.current_instance = instance
        self.current_instance_path = instance_path
        return self._get_instance_features(self.current_instance)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # super().reset(seed=seed) # For newer gym versions
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            # self.action_space.seed(seed) # For older gym
            # self.observation_space.seed(seed)

        self.current_step = 0
        observation = self._load_random_instance()
        info = {"instance_path": self.current_instance_path, "instance_name": self.current_instance.name}
        return observation, info

    def _run_alm_optimizer_quantum(self, initial_rho, initial_sigma) -> Tuple[float, int, bool, float, List[List[int]]]:
        """
        Runs the AlmOptimizerQuantum with the given penalties.
        Returns: (solution_cost, alm_iterations, is_feasible, time_taken, routes)
        """
        if self.current_instance is None:
            raise ValueError("Current instance is not loaded.")

        # Use fixed ALM parameters from config, but initial penalties from RL agent
        alm_q_optimizer = AlmOptimizerQuantum(
            instance=self.current_instance,
            initial_penalty_rate=initial_rho,             # From RL Agent
            initial_capacity_penalty_rate=initial_sigma,  # From RL Agent
            penalty_increase_factor=self.alm_config.get('penalty_increase_factor', 1.0), # Fixed, so RL controls penalties
            capacity_penalty_increase_factor=self.alm_config.get('capacity_penalty_increase_factor', 1.0), # Fixed
            max_penalty_rate=self.alm_config.get('max_penalty_rate', MAX_PENALTY_RHO), # Use consistent max
            max_capacity_penalty_rate=self.alm_config.get('max_capacity_penalty_rate', MAX_PENALTY_SIGMA),
            convergence_tolerance=self.alm_config.get('convergence_tolerance', 1e-3),
            capacity_convergence_tolerance=self.alm_config.get('capacity_convergence_tolerance', 1.0),
            max_alm_iterations=self.alm_config.get('max_alm_iterations', 50), # Crucial for speed
            subproblem_max_vehicles=self.alm_config.get('subproblem_max_vehicles'),
            verbose=self.alm_config.get('verbose_alm', 0), # Keep ALM quiet during RL
            quantum_solver_config=self.quantum_solver_config # Pass the full quantum config
        )

        start_time = time.time()
        best_solution_obj = alm_q_optimizer.solve()
        time_taken = time.time() - start_time

        if best_solution_obj and best_solution_obj.is_feasible:
            return best_solution_obj.total_cost, len(alm_q_optimizer.iteration_log), True, time_taken, best_solution_obj.routes
        elif best_solution_obj: # Infeasible solution found
            # Penalize based on cost of infeasible or a fixed large penalty
            # For simplicity, use its cost + a large penalty, or just a large fixed penalty
            return best_solution_obj.total_cost + 10000.0, len(alm_q_optimizer.iteration_log), False, time_taken, best_solution_obj.routes
        else: # No solution found at all
            return 20000.0, len(alm_q_optimizer.iteration_log), False, time_taken, []


    def step(self, action: np.ndarray):
        self.current_step += 1

        initial_rho, initial_sigma = self._scale_action(action)
        
        solution_cost, alm_iters, is_feasible, time_taken, routes = self._run_alm_optimizer_quantum(initial_rho, initial_sigma)

        # --- Reward calculation (CRITICAL - this needs careful tuning) ---
        reward = 0.0
        if is_feasible:
            # Higher reward for lower cost (e.g., inverse of cost)
            # Ensure solution_cost is positive; add small epsilon if it can be zero
            reward += 1000.0 / (solution_cost + 1e-5) 
            # Bonus for converging faster (fewer ALM iterations)
            reward += 10.0 / (alm_iters + 1e-5)
        else:
            # Penalty for infeasibility
            reward -= 100.0 
            # Penalize based on how far the cost is, or a fixed penalty
            reward -= solution_cost / 100.0 # If solution_cost for infeasible is high

        # Check if episode is done
        terminated = self.current_step >= self.max_steps_per_episode
        truncated = False # Not typically used if episode length is fixed by max_steps

        # Get observation for the next step (new random instance)
        # If terminated, the next state is usually not used by agent for update if it's a terminal state.
        # But the env should reset and provide a new starting state.
        # For this setup, one step IS an episode. So, the next_observation is from reset().
        # However, gym step must return an observation. We can return the current one,
        # or features of next instance if we load it here.
        # Simpler: let the training loop call reset(). Here, just return current obs.
        observation = self._get_instance_features(self.current_instance)
        if terminated:
            next_observation = self._load_random_instance() # Load next for next call to reset
            observation = next_observation # Return features of the *next* instance


        info = {
            "instance_path": self.current_instance_path,
            "instance_name": self.current_instance.name,
            "chosen_rho": initial_rho,
            "chosen_sigma": initial_sigma,
            "solution_cost": solution_cost if is_feasible else None, # Only log cost if feasible
            "raw_solution_cost": solution_cost, # Log raw cost always
            "alm_iterations": alm_iters,
            "is_feasible": is_feasible,
            "time_taken_alm_solve": time_taken,
            "routes": routes
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        # For now, simple text rendering. Could be expanded.
        if self.current_instance:
            print(f"Current Instance: {self.current_instance.name} (Dim: {self.current_instance.dimension})")
            # Potentially print solution if available after a step
        else:
            print("No current instance loaded.")

    def close(self):
        pass # Any cleanup if needed

# Example usage (for testing the environment itself)
if __name__ == '__main__':
    # Create dummy instance files for testing
    # In a real scenario, these would be paths to actual .vrp files.
    # IMPORTANT: These instances MUST be very small for quantum ALM.
    dummy_instance_dir = "dummy_tiny_instances"
    import os
    if not os.path.exists(dummy_instance_dir): os.makedirs(dummy_instance_dir)
    
    # Create a truly tiny instance file (e.g., 3 nodes: D, C1, C2)
    # Consistent with the test in AlmOptimizerQuantum
    instance_content_3node = """
NAME : tiny_3node.vrp
COMMENT : Dummy 3-node instance for quantum ALM env test
TYPE : CVRP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EXPLICIT
EDGE_WEIGHT_FORMAT : FULL_MATRIX
CAPACITY : 50
NODE_COORD_SECTION
1 0 0 
2 10 10
3 10 0
EDGE_WEIGHT_SECTION
0 10 15
10 0 5
15 5 0
DEMAND_SECTION
1 0
2 10
3 15
DEPOT_SECTION
1
-1
EOF
"""
    with open(os.path.join(dummy_instance_dir, "tiny_3node.vrp"), "w") as f:
        f.write(instance_content_3node)

    instance_files_test = [os.path.join(dummy_instance_dir, "tiny_3node.vrp")]

    # Fixed ALM parameters (some will be overridden by RL agent's actions)
    alm_config_test = {
        'max_alm_iterations': 20, # Keep low for env testing
        'penalty_increase_factor': 1.0, # Fixed, as RL sets initial penalties
        'capacity_penalty_increase_factor': 1.0, # Fixed
        'verbose_alm': 0 # Keep ALM quiet
    }
    # Quantum solver configuration (ensure this makes VQE fast enough for env steps)
    quantum_config_test = {
        'max_customers_in_quantum_subproblem': 2,
        'constraint_penalty_factor': 500.0,
        'vqe_reps': 1,
        'vqe_max_iter': 30, # VERY IMPORTANT: Keep low for reasonable step time
        'vqe_optimizer_method': "Powell",
        'plot_folder_prefix': "output_plots_env_test_"
    }

    env = AlmPenaltyEnvQuBoVqe(
        instance_files=instance_files_test, 
        alm_config=alm_config_test,
        quantum_solver_config=quantum_config_test,
        max_instance_dimension=5 # Only allow very small instances
    )

    print("Environment Initialized.")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    # Test reset
    obs, info = env.reset()
    print(f"Reset Observation: {obs}")
    print(f"Reset Info: {info}")

    # Test step
    # Episodes are very short (1 step per episode by default)
    for episode in range(3):
        print(f"\n--- Env Episode {episode + 1} ---")
        obs, info = env.reset() # New instance for each episode
        action = env.action_space.sample() # Random action
        print(f"Sampled Action: {action}")
        
        # Since one step is one episode here:
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        print(f"  Action taken (scaled rho, sigma): {step_info['chosen_rho']:.2f}, {step_info['chosen_sigma']:.2f}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Solution Cost (if feasible): {step_info['solution_cost']}")
        print(f"  ALM Iterations: {step_info['alm_iterations']}")
        print(f"  Is Feasible: {step_info['is_feasible']}")
        print(f"  Time for ALM solve: {step_info['time_taken_alm_solve']:.2f}s")
        print(f"  Routes: {step_info['routes']}")
        print(f"  Terminated: {terminated}, Truncated: {truncated}")
        print(f"  Next Observation (features of next instance): {next_obs}")
        if terminated:
            print("  Episode finished.")