# # cvrp_tripartite_solver/src/rl_classical/alm_penalty_env.py

# import random
# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import math # <--- ADDED THIS LINE
# from typing import List, Dict, Tuple, Optional, Any

# # Project-specific imports
# from common.cvrp_instance import CVRPInstance, load_cvrp_instance
# from alm.alm_optimizer import AlmOptimizer
# # AlmOptimizer might need this directly or indirectly, ensure it's available if so.
# # from alm.subproblem_solvers import solve_esp_with_dominance 

# # Default config for AlmOptimizer when controlled by RL
# DEFAULT_ALM_FIXED_CONFIG = {
#     "convergence_tolerance": 1e-3,
#     "capacity_convergence_tolerance": 1.0,
#     "max_alm_iterations": 100,
#     "subproblem_max_vehicles": None,
#     "verbose": 0
# }

# MIN_PENALTY_RHO = 0.1
# MAX_PENALTY_RHO = 1000.0
# MIN_PENALTY_SIGMA = 0.1
# MAX_PENALTY_SIGMA = 1000.0
# N_INSTANCE_FEATURES = 6


# class AlmPenaltyEnv(gym.Env):
#     metadata = {'render_modes': [], 'render_fps': 4}

#     def __init__(self,
#                  instance_paths: List[str],
#                  alm_fixed_config: Optional[Dict] = None,
#                  max_alm_iterations_override: Optional[int] = None,
#                  env_config: Optional[Dict] = None):
#         super().__init__()

#         if not instance_paths:
#             raise ValueError("instance_paths list cannot be empty.")
#         self.instance_paths = instance_paths
#         self.num_instances = len(instance_paths)
#         self.current_instance_idx = 0
#         self.current_instance: Optional[CVRPInstance] = None

#         self.alm_config = DEFAULT_ALM_FIXED_CONFIG.copy()
#         if alm_fixed_config:
#             self.alm_config.update(alm_fixed_config)
        
#         if max_alm_iterations_override is not None:
#             self.alm_config["max_alm_iterations"] = max_alm_iterations_override
        
#         self.env_config = env_config if env_config is not None else {}

#         self.action_space = spaces.Box(
#             low=np.array([MIN_PENALTY_RHO, MIN_PENALTY_SIGMA], dtype=np.float32),
#             high=np.array([MAX_PENALTY_RHO, MAX_PENALTY_SIGMA], dtype=np.float32),
#             dtype=np.float32
#         )

#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(N_INSTANCE_FEATURES,), dtype=np.float32
#         )

#     def _get_instance_features(self, instance: CVRPInstance) -> np.ndarray:
#         if instance is None:
#             return np.zeros(N_INSTANCE_FEATURES, dtype=np.float32)

#         num_customers = instance.dimension - 1 if instance.dimension > 0 else 0
#         vehicle_capacity = float(instance.capacity)
        
#         customer_demands = [
#             instance.demands[i] for i in range(instance.dimension) if i != instance.depot
#         ]
#         if not customer_demands:
#             mean_demand = 0.0
#             std_demand = 0.0
#             total_demand = 0.0
#         else:
#             mean_demand = np.mean(customer_demands)
#             std_demand = np.std(customer_demands)
#             total_demand = np.sum(customer_demands)

#         num_vehicles_est = instance.num_vehicles_comment if instance.num_vehicles_comment else \
#                            (math.ceil(total_demand / vehicle_capacity) if vehicle_capacity > 0 else 1)
#         num_vehicles_est = max(1, num_vehicles_est)

#         ratio_demand_capacity = total_demand / (num_vehicles_est * vehicle_capacity) if vehicle_capacity > 0 and num_vehicles_est > 0 else 0.0 # Added num_vehicles_est > 0 check
        
#         features = np.array([
#             float(num_customers),
#             vehicle_capacity,
#             float(mean_demand),
#             float(std_demand),
#             float(total_demand),
#             float(ratio_demand_capacity)
#         ], dtype=np.float32)
#         return features

#     def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
#         super().reset(seed=seed)
#         instance_path = self.instance_paths[self.current_instance_idx]
#         self.current_instance_idx = (self.current_instance_idx + 1) % self.num_instances
        
#         try:
#             self.current_instance = load_cvrp_instance(instance_path)
#         except Exception as e:
#             print(f"Error loading instance {instance_path}: {e}")
#             return np.zeros(self.observation_space.shape, dtype=np.float32), {"error": str(e), "instance_name": instance_path}

#         observation = self._get_instance_features(self.current_instance)
#         info = {"instance_name": self.current_instance.name}
        
#         return observation, info

#     def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
#         if self.current_instance is None:
#             dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
#             return dummy_obs, -1000.0, True, False, {"error": "No current instance loaded before step."}

#         rl_rho = float(action[0])
#         rl_sigma = float(action[1])

#         optimizer = AlmOptimizer(
#             instance=self.current_instance,
#             initial_penalty_rate=rl_rho,
#             penalty_increase_factor=1.0,
#             max_penalty_rate=MAX_PENALTY_RHO,
#             initial_capacity_penalty_rate=rl_sigma,
#             capacity_penalty_increase_factor=1.0,
#             max_capacity_penalty_rate=MAX_PENALTY_SIGMA,
#             convergence_tolerance=self.alm_config["convergence_tolerance"],
#             capacity_convergence_tolerance=self.alm_config["capacity_convergence_tolerance"],
#             max_alm_iterations=self.alm_config["max_alm_iterations"],
#             subproblem_max_vehicles=self.alm_config["subproblem_max_vehicles"],
#             verbose=self.alm_config["verbose"]
#         )

#         alm_solution = optimizer.solve()

#         reward = 0.0
#         info: Dict[str, Any] = {
#             "instance_name": self.current_instance.name,
#             "chosen_rho": rl_rho,
#             "chosen_sigma": rl_sigma,
#             "alm_iterations": len(optimizer.iteration_log) if optimizer.iteration_log else 0 # check if iteration_log is None or empty
#         }

#         if alm_solution and alm_solution.is_feasible:
#             reward = 100.0 / (alm_solution.total_cost + 1e-6)
#             c_iter_penalty = 0.01
#             reward -= c_iter_penalty * info["alm_iterations"] # use info value
#             info["solution_cost"] = alm_solution.total_cost
#             info["solution_feasible"] = True
#         else:
#             reward = -200.0
#             info["solution_cost"] = float('inf')
#             info["solution_feasible"] = False
#             if optimizer.iteration_log:
#                  last_log = optimizer.iteration_log[-1]
#                  info["final_max_visit_violation"] = last_log.get("max_abs_g_i")
#                  info["final_max_cap_violation"] = last_log.get("max_cap_violation")

#         terminated = True
#         truncated = False
#         next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
#         return next_observation, reward, terminated, truncated, info

#     def render(self):
#         pass

#     def close(self):
#         pass

# if __name__ == '__main__':
#     import os
#     # ensure math is imported for main example too (already done by adding at top)

#     print("Testing AlmPenaltyEnv...")
#     dummy_instances_dir = "dummy_cvrp_instances_for_env_test"
#     os.makedirs(dummy_instances_dir, exist_ok=True)
#     dummy_instance_paths = []

#     for i in range(2):
#         file_path = os.path.join(dummy_instances_dir, f"dummy_instance_{i}.vrp")
#         with open(file_path, "w") as df:
#             num_nodes = random.randint(3,5)
#             capacity = random.randint(50,150)
#             df.write(f"NAME : dummy_instance_{i}\n")
#             df.write(f"COMMENT : Test instance for AlmPenaltyEnv {i}\n")
#             df.write(f"TYPE : CVRP\n")
#             df.write(f"DIMENSION : {num_nodes}\n")
#             df.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n")
#             df.write(f"CAPACITY : {capacity}\n")
#             df.write(f"NODE_COORD_SECTION\n")
#             for N_idx in range(1, num_nodes + 1):
#                 df.write(f"{N_idx} {random.randint(0,100)} {random.randint(0,100)}\n")
#             df.write(f"DEMAND_SECTION\n")
#             df.write(f"1 0\n")
#             for N_idx in range(2, num_nodes + 1):
#                  df.write(f"{N_idx} {random.randint(5, max(1, capacity//3))}\n") # Ensure demand is reasonable
#             df.write(f"DEPOT_SECTION\n")
#             df.write(f"1\n")
#             df.write(f"-1\n")
#             df.write(f"EOF\n")
#         dummy_instance_paths.append(file_path)

#     if not dummy_instance_paths:
#         print("No dummy instances created. Exiting test.")
#         exit()

#     env = AlmPenaltyEnv(instance_paths=dummy_instance_paths, max_alm_iterations_override=30)

#     print("\n--- Testing reset() ---")
#     obs, info = env.reset()
#     print(f"Initial observation shape: {obs.shape}")
#     print(f"Initial observation (sample): {obs[:min(N_INSTANCE_FEATURES, 5)]}")
#     print(f"Info: {info}")
#     assert obs.shape == (N_INSTANCE_FEATURES,), "Observation shape mismatch"

#     print("\n--- Testing step() ---")
#     action = env.action_space.sample() 
#     print(f"Sample action: rho={action[0]:.2f}, sigma={action[1]:.2f}")
    
#     next_obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Next observation shape: {next_obs.shape}")
#     print(f"Reward: {reward:.4f}")
#     print(f"Terminated: {terminated}")
#     print(f"Truncated: {truncated}")
#     print(f"Info: {info}")
#     assert terminated, "Episode should terminate after one ALM run"

#     print("\n--- Testing reset() again and step() for second instance ---")
#     obs, info = env.reset()
#     print(f"New observation (sample): {obs[:min(N_INSTANCE_FEATURES, 5)]}")
#     print(f"Info: {info}")
#     action = env.action_space.sample()
#     print(f"Sample action: rho={action[0]:.2f}, sigma={action[1]:.2f}")
#     next_obs, reward, terminated, truncated, info = env.step(action)
#     print(f"Reward: {reward:.4f}")
#     print(f"Info: {info}")

#     # Clean up dummy files
#     for p in dummy_instance_paths:
#         if os.path.exists(p):
#             os.remove(p)
#     if os.path.exists(dummy_instances_dir) and not os.listdir(dummy_instances_dir):
#         os.rmdir(dummy_instances_dir)
#     print("\nEnvironment test finished.")


# cvrp_tripartite_solver/src/rl_classical/alm_penalty_env.py

import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import time # Added for timing
from typing import List, Dict, Tuple, Optional, Any

# Project-specific imports
from common.cvrp_instance import CVRPInstance, load_cvrp_instance
from alm.alm_optimizer import AlmOptimizer
# from alm.subproblem_solvers import solve_esp_with_dominance # Not directly used here

# Default config for AlmOptimizer when controlled by RL
DEFAULT_ALM_FIXED_CONFIG = {
    "convergence_tolerance": 1e-3,
    "capacity_convergence_tolerance": 1.0,
    "max_alm_iterations": 100,
    "subproblem_max_vehicles": None,
    "verbose": 0
}

MIN_PENALTY_RHO = 0.1
MAX_PENALTY_RHO = 1000.0
MIN_PENALTY_SIGMA = 0.1
MAX_PENALTY_SIGMA = 1000.0
N_INSTANCE_FEATURES = 6


class AlmPenaltyEnv(gym.Env):
    metadata = {'render_modes': [], 'render_fps': 4}

    def __init__(self,
                 instance_paths: List[str],
                 alm_fixed_config: Optional[Dict] = None,
                 max_alm_iterations_override: Optional[int] = None,
                 env_config: Optional[Dict] = None):
        super().__init__()

        if not instance_paths:
            raise ValueError("instance_paths list cannot be empty.")
        self.instance_paths = instance_paths
        self.num_instances = len(instance_paths)
        self.current_instance_idx = 0
        self.current_instance: Optional[CVRPInstance] = None

        self.alm_config = DEFAULT_ALM_FIXED_CONFIG.copy()
        if alm_fixed_config:
            self.alm_config.update(alm_fixed_config)
        
        if max_alm_iterations_override is not None:
            self.alm_config["max_alm_iterations"] = max_alm_iterations_override
        
        self.env_config = env_config if env_config is not None else {}

        self.action_space = spaces.Box(
            low=np.array([MIN_PENALTY_RHO, MIN_PENALTY_SIGMA], dtype=np.float32),
            high=np.array([MAX_PENALTY_RHO, MAX_PENALTY_SIGMA], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(N_INSTANCE_FEATURES,), dtype=np.float32
        )

    def _get_instance_features(self, instance: CVRPInstance) -> np.ndarray:
        if instance is None:
            return np.zeros(N_INSTANCE_FEATURES, dtype=np.float32)

        num_customers = instance.dimension - 1 if instance.dimension > 0 else 0
        vehicle_capacity = float(instance.capacity)
        
        customer_demands = [
            instance.demands[i] for i in range(instance.dimension) if i != instance.depot
        ]
        if not customer_demands:
            mean_demand = 0.0
            std_demand = 0.0
            total_demand = 0.0
        else:
            mean_demand = np.mean(customer_demands)
            std_demand = np.std(customer_demands)
            total_demand = np.sum(customer_demands)

        num_vehicles_est = instance.num_vehicles_comment if instance.num_vehicles_comment else \
                           (math.ceil(total_demand / vehicle_capacity) if vehicle_capacity > 0 else 1)
        num_vehicles_est = max(1, num_vehicles_est)

        ratio_demand_capacity = total_demand / (num_vehicles_est * vehicle_capacity) if vehicle_capacity > 0 and num_vehicles_est > 0 else 0.0
        
        features = np.array([
            float(num_customers),
            vehicle_capacity,
            float(mean_demand),
            float(std_demand),
            float(total_demand),
            float(ratio_demand_capacity)
        ], dtype=np.float32)
        return features

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        instance_path = self.instance_paths[self.current_instance_idx]
        self.current_instance_idx = (self.current_instance_idx + 1) % self.num_instances
        
        try:
            self.current_instance = load_cvrp_instance(instance_path)
        except Exception as e:
            print(f"Error loading instance {instance_path}: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32), {"error": str(e), "instance_name": instance_path, "instance_path": instance_path}

        observation = self._get_instance_features(self.current_instance)
        info = {
            "instance_name": self.current_instance.name,
            "instance_path": instance_path # Pass along the path for filtering/identification
            }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_instance is None:
            dummy_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return dummy_obs, -1000.0, True, False, {"error": "No current instance loaded before step."}

        rl_rho = float(action[0])
        rl_sigma = float(action[1])

        optimizer = AlmOptimizer(
            instance=self.current_instance,
            initial_penalty_rate=rl_rho,
            penalty_increase_factor=1.0,
            max_penalty_rate=MAX_PENALTY_RHO,
            initial_capacity_penalty_rate=rl_sigma,
            capacity_penalty_increase_factor=1.0,
            max_capacity_penalty_rate=MAX_PENALTY_SIGMA,
            convergence_tolerance=self.alm_config["convergence_tolerance"],
            capacity_convergence_tolerance=self.alm_config["capacity_convergence_tolerance"],
            max_alm_iterations=self.alm_config["max_alm_iterations"],
            subproblem_max_vehicles=self.alm_config["subproblem_max_vehicles"],
            verbose=self.alm_config["verbose"] # Use verbosity from config
        )
        
        start_time_alm = time.time()
        alm_solution_obj = optimizer.solve() # This runs the full ALM optimization
        time_taken_alm = time.time() - start_time_alm

        reward = 0.0
        info: Dict[str, Any] = {
            "instance_name": self.current_instance.name,
            "instance_path": self.current_instance_idx -1 if self.current_instance_idx > 0 else self.num_instances -1, # Path of current instance
            "chosen_rho": rl_rho,
            "chosen_sigma": rl_sigma,
            "alm_iterations": len(optimizer.iteration_log) if optimizer.iteration_log else 0,
            "time_taken_alm_solve": time_taken_alm,
            "alm_solution_object": alm_solution_obj # Return the full solution object
        }

        if alm_solution_obj and alm_solution_obj.is_feasible:
            reward = 100.0 / (alm_solution_obj.total_cost + 1e-6)
            c_iter_penalty = 0.01
            reward -= c_iter_penalty * info["alm_iterations"]
            info["solution_cost"] = alm_solution_obj.total_cost
            info["solution_feasible"] = True
            # Routes are in alm_solution_obj.routes
        else:
            reward = -200.0
            info["solution_cost"] = float('inf')
            info["solution_feasible"] = False
            if optimizer.iteration_log:
                 last_log = optimizer.iteration_log[-1]
                 info["final_max_visit_violation"] = last_log.get("max_abs_g_i")
                 info["final_max_cap_violation"] = last_log.get("max_cap_violation")
            # Routes will be None or empty if no feasible solution

        terminated = True
        truncated = False
        next_observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        return next_observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':
    import os
    print("Testing AlmPenaltyEnv with updated info...")
    dummy_instances_dir = "dummy_cvrp_instances_for_env_test"
    os.makedirs(dummy_instances_dir, exist_ok=True)
    dummy_instance_paths = []

    for i in range(1): # Create just one for simpler test
        file_path = os.path.join(dummy_instances_dir, f"dummy_instance_main_{i}.vrp")
        with open(file_path, "w") as df:
            num_nodes = 3
            capacity = 100
            df.write(f"NAME : dummy_instance_main_{i}\nTYPE : CVRP\nDIMENSION : {num_nodes}\n")
            df.write(f"EDGE_WEIGHT_TYPE : EUC_2D\nCAPACITY : {capacity}\nNODE_COORD_SECTION\n")
            df.write(f"1 0 0\n2 10 0\n3 0 10\nDEMAND_SECTION\n1 0\n2 30\n3 40\n")
            df.write(f"DEPOT_SECTION\n1\n-1\nEOF\n")
        dummy_instance_paths.append(file_path)

    env = AlmPenaltyEnv(instance_paths=dummy_instance_paths, max_alm_iterations_override=10)
    obs, reset_info = env.reset()
    print(f"Reset Info: {reset_info}")
    
    action = env.action_space.sample() 
    print(f"Sample Action: rho={action[0]:.2f}, sigma={action[1]:.2f}")
    
    _, _, _, _, step_info = env.step(action)
    print("\nStep Info:")
    for k, v in step_info.items():
        if k == "alm_solution_object":
            print(f"  {k}: {'CVRPSolution object present' if v else 'None'}")
            if v:
                print(f"    Solution Cost: {v.total_cost}")
                print(f"    Solution Feasible: {v.is_feasible}")
                print(f"    Solution Routes: {v.routes}")
        else:
            print(f"  {k}: {v}")
    
    assert "time_taken_alm_solve" in step_info
    assert "alm_solution_object" in step_info
    if step_info["alm_solution_object"]:
        assert step_info["solution_cost"] == step_info["alm_solution_object"].total_cost

    for p in dummy_instance_paths:
        if os.path.exists(p): os.remove(p)
    if os.path.exists(dummy_instances_dir) and not os.listdir(dummy_instances_dir):
        os.rmdir(dummy_instances_dir)
    print("\nEnvironment test with extended info finished.")
