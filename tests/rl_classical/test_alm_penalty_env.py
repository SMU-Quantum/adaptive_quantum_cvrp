# cvrp_tripartite_solver/tests/rl_classical/test_alm_penalty_env.py

import pytest
import gymnasium as gym
import numpy as np
import os
import random
import math
from unittest.mock import patch, MagicMock

# Project-specific imports
# Assuming your project structure and pyproject.toml allow these imports
from common.cvrp_instance import CVRPInstance 
from rl_classical.alm_penalty_env import (
    AlmPenaltyEnv,
    N_INSTANCE_FEATURES,
    MIN_PENALTY_RHO, MAX_PENALTY_RHO,
    MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA
)
# We will mock AlmOptimizer, so direct import is for type hinting if needed,
# but patching targets the string path to where it's looked up by the env.
# from alm.alm_optimizer import AlmOptimizer


@pytest.fixture
def dummy_instance_file_generator(tmp_path):
    """
    A fixture to generate a list of dummy CVRP instance file paths.
    """
    dummy_files = []
    base_dir = tmp_path / "dummy_cvrp_instances_for_test_env"
    base_dir.mkdir(exist_ok=True)

    def _create_files(num_files=2):
        for i in range(num_files):
            file_path = base_dir / f"dummy_instance_env_test_{i}.vrp"
            with open(file_path, "w") as df:
                num_nodes = random.randint(3, 5)
                capacity = random.randint(50, 150)
                df.write(f"NAME : dummy_instance_env_test_{i}\n")
                df.write(f"COMMENT : Test instance for AlmPenaltyEnv fixture {i}\n")
                df.write(f"TYPE : CVRP\n")
                df.write(f"DIMENSION : {num_nodes}\n")
                df.write(f"EDGE_WEIGHT_TYPE : EUC_2D\n") # Assuming EUC_2D for simplicity
                df.write(f"CAPACITY : {capacity}\n")
                df.write(f"NODE_COORD_SECTION\n")
                for N_idx in range(1, num_nodes + 1):
                    df.write(f"{N_idx} {random.randint(0,100)} {random.randint(0,100)}\n")
                df.write(f"DEMAND_SECTION\n")
                df.write(f"1 0\n") # Depot demand
                for N_idx in range(2, num_nodes + 1):
                    df.write(f"{N_idx} {random.randint(5, max(1, capacity//3))}\n") # Ensure demand is reasonable
                df.write(f"DEPOT_SECTION\n")
                df.write(f"1\n")
                df.write(f"-1\n")
                df.write(f"EOF\n")
            dummy_files.append(str(file_path))
        return dummy_files
    return _create_files

@pytest.fixture
def sample_cvrp_instance_for_features():
    """Provides a simple CVRPInstance object for direct feature testing."""
    return CVRPInstance(
        name="feature_test_instance",
        dimension=4, # Depot + 3 customers
        capacity=100,
        distance_matrix=[[0,1,2,3],[1,0,1,2],[2,1,0,1],[3,2,1,0]], # Dummy
        demands=[0, 10, 20, 30], # Depot demand must be 0
        depot=0,
        num_vehicles_comment=2,
        coordinates=[(0,0),(1,1),(2,2),(3,3)]
    )

@pytest.fixture
def sample_cvrp_instance_no_customers():
    """Instance with only a depot."""
    return CVRPInstance(
        name="no_customer_instance",
        dimension=1,
        capacity=100,
        distance_matrix=[[0]],
        demands=[0],
        depot=0
    )

def test_env_initialization(dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)

    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (2,)
    assert np.allclose(env.action_space.low, [MIN_PENALTY_RHO, MIN_PENALTY_SIGMA])
    assert np.allclose(env.action_space.high, [MAX_PENALTY_RHO, MAX_PENALTY_SIGMA])

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (N_INSTANCE_FEATURES,)
    assert env.num_instances == 1

def test_env_initialization_empty_paths():
    with pytest.raises(ValueError, match="instance_paths list cannot be empty"):
        AlmPenaltyEnv(instance_paths=[])

def test_get_instance_features_valid_instance(sample_cvrp_instance_for_features):
    env = AlmPenaltyEnv(instance_paths=["dummy_path"]) # Path not used for this direct test
    features = env._get_instance_features(sample_cvrp_instance_for_features)
    
    assert features.shape == (N_INSTANCE_FEATURES,)
    assert features.dtype == np.float32
    
    # Expected values based on sample_cvrp_instance_for_features and N_INSTANCE_FEATURES = 6
    # 1. Num customers: 3
    # 2. Capacity: 100
    # 3. Mean demand (10,20,30): 20
    # 4. Std demand: np.std([10,20,30]) approx 8.165
    # 5. Total demand: 60
    # 6. Ratio total_demand / (num_vehicles_comment * capacity) = 60 / (2 * 100) = 0.3
    expected_features = np.array([
        3.0, 
        100.0, 
        20.0, 
        np.std([10,20,30]), 
        60.0, 
        0.3 
    ], dtype=np.float32)
    assert np.allclose(features, expected_features)

def test_get_instance_features_no_customers(sample_cvrp_instance_no_customers):
    env = AlmPenaltyEnv(instance_paths=["dummy_path"])
    features = env._get_instance_features(sample_cvrp_instance_no_customers)
    assert features.shape == (N_INSTANCE_FEATURES,)
    # Expected: num_cust=0, capacity=100, mean_demand=0, std_demand=0, total_demand=0, ratio=0
    expected_features = np.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(features, expected_features)


def test_env_reset(dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(2)
    env = AlmPenaltyEnv(instance_paths=instance_paths)

    obs, info = env.reset()
    assert obs.shape == (N_INSTANCE_FEATURES,)
    assert isinstance(info, dict)
    assert "instance_name" in info
    assert info["instance_name"] == "dummy_instance_env_test_0" # Assuming sequential load
    first_obs = obs

    obs2, info2 = env.reset()
    assert obs2.shape == (N_INSTANCE_FEATURES,)
    assert "instance_name" in info2
    assert info2["instance_name"] == "dummy_instance_env_test_1"
    # Check if observations are different if instances are different (they should be)
    assert not np.array_equal(first_obs, obs2)


# --- Tests for step method ---
# We need to mock AlmOptimizer to avoid running the actual ALM
@patch('rl_classical.alm_penalty_env.AlmOptimizer') # Path to AlmOptimizer as seen by alm_penalty_env.py
def test_env_step_successful_alm_run(mock_alm_optimizer_class, dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths, max_alm_iterations_override=50)
    obs, reset_info = env.reset()

    # Configure the mock AlmOptimizer instance that will be created
    mock_alm_instance = MagicMock()
    mock_alm_instance.solve.return_value = MagicMock(
        is_feasible=True, 
        total_cost=150.0
    )
    mock_alm_instance.iteration_log = [{} for _ in range(10)] # Mock 10 iterations
    mock_alm_optimizer_class.return_value = mock_alm_instance

    action = env.action_space.sample() # [rho, sigma]
    rho_val, sigma_val = action[0], action[1]

    next_obs, reward, terminated, truncated, info = env.step(action)

    # Check AlmOptimizer instantiation
    mock_alm_optimizer_class.assert_called_once()
    args, kwargs = mock_alm_optimizer_class.call_args
    assert kwargs['instance'] == env.current_instance
    assert kwargs['initial_penalty_rate'] == rho_val
    assert kwargs['penalty_increase_factor'] == 1.0 # Crucial: check fixed penalty
    assert kwargs['initial_capacity_penalty_rate'] == sigma_val
    assert kwargs['capacity_penalty_increase_factor'] == 1.0 # Crucial: check fixed penalty
    assert kwargs['max_alm_iterations'] == 50

    mock_alm_instance.solve.assert_called_once()

    assert terminated is True
    assert truncated is False
    assert isinstance(reward, float)
    expected_reward = 100.0 / (150.0 + 1e-6) - (0.01 * 10)
    assert reward == pytest.approx(expected_reward)
    
    assert isinstance(info, dict)
    assert info["instance_name"] == env.current_instance.name
    assert info["chosen_rho"] == rho_val
    assert info["chosen_sigma"] == sigma_val
    assert info["alm_iterations"] == 10
    assert info["solution_cost"] == 150.0
    assert info["solution_feasible"] is True
    
    assert next_obs.shape == (N_INSTANCE_FEATURES,) # Should be dummy zero obs

@patch('rl_classical.alm_penalty_env.AlmOptimizer')
def test_env_step_alm_fails_to_find_feasible(mock_alm_optimizer_class, dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)
    obs, _ = env.reset()

    mock_alm_instance = MagicMock()
    mock_alm_instance.solve.return_value = MagicMock(is_feasible=False, total_cost=float('inf'))
    mock_alm_instance.iteration_log = [{"max_abs_g_i": 5.0, "max_cap_violation": 100.0}] # Mock one log entry
    mock_alm_optimizer_class.return_value = mock_alm_instance

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == -200.0 # Default penalty for infeasible
    assert info["solution_feasible"] is False
    assert info["final_max_visit_violation"] == 5.0
    assert info["final_max_cap_violation"] == 100.0

@patch('rl_classical.alm_penalty_env.AlmOptimizer')
def test_env_step_alm_returns_none(mock_alm_optimizer_class, dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)
    obs, _ = env.reset()

    mock_alm_instance = MagicMock()
    mock_alm_instance.solve.return_value = None # ALM completely failed
    mock_alm_instance.iteration_log = [] 
    mock_alm_optimizer_class.return_value = mock_alm_instance
    
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert terminated is True
    assert reward == -200.0
    assert info["solution_feasible"] is False

def test_env_action_space_sample_and_contains(dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)
    for _ in range(10):
        action = env.action_space.sample()
        assert env.action_space.contains(action)
        assert MIN_PENALTY_RHO <= action[0] <= MAX_PENALTY_RHO
        assert MIN_PENALTY_SIGMA <= action[1] <= MAX_PENALTY_SIGMA

def test_env_observation_space_sample_and_contains(dummy_instance_file_generator):
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)
    # Note: sampling observation space directly doesn't load an instance here,
    # it just gives a random sample from the defined Box space.
    for _ in range(10):
        obs_sample = env.observation_space.sample()
        assert env.observation_space.contains(obs_sample)

# It's good practice to check conformity with Gymnasium API
def test_gymnasium_api_conformity(dummy_instance_file_generator):
    from gymnasium.utils.env_checker import check_env
    instance_paths = dummy_instance_file_generator(1)
    env = AlmPenaltyEnv(instance_paths=instance_paths)
    # This will raise an error if the environment doesn't follow the API
    # Note: This check can be slow if your reset/step are complex.
    # Given our step runs a full ALM, this might be slow.
    # For now, we'll trust the manual checks.
    # For a more thorough check, you'd ensure no exceptions are raised by:
    try:
        check_env(env, skip_render_check=True) # AlmPenaltyEnv has no render
    except Exception as e:
        pytest.fail(f"Gymnasium check_env failed: {e}")


if __name__ == "__main__":
    # Example of how to run tests with pytest programmatically,
    # though usually you'd run `pytest` from the command line.
    # This also helps ensure the fixtures work.
    
    # Create a temporary directory for dummy files for this specific run
    temp_dir_for_main = "temp_pytest_dummy_files"
    os.makedirs(temp_dir_for_main, exist_ok=True)
    
    class TmpPath: # Basic mock for tmp_path fixture for direct script run
        def __init__(self, path):
            self.path = path
        def __truediv__(self, other):
            return TmpPath(os.path.join(self.path, other))
        def mkdir(self, exist_ok=False):
            os.makedirs(self.path, exist_ok=exist_ok)
        def __str__(self):
            return self.path

    print("Simulating test runs (output may differ from actual pytest due to mocking scope):")
    
    # Simulate dummy_instance_file_generator
    main_tmp_path = TmpPath(temp_dir_for_main)
    
    def _main_dummy_generator(num_files=1):
        paths = []
        base_dir = main_tmp_path / "dummy_cvrp_instances_for_main_test"
        base_dir.mkdir(exist_ok=True)
        for i in range(num_files):
            file_path_obj = base_dir / f"main_dummy_instance_env_test_{i}.vrp"
            with open(str(file_path_obj), "w") as df: # Simplified content for brevity
                df.write("NAME : main_dummy\nDIMENSION : 3\nCAPACITY : 100\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 10 0\n3 0 10\nDEMAND_SECTION\n1 0\n2 10\n3 10\nDEPOT_SECTION\n1\n-1\nEOF\n")
            paths.append(str(file_path_obj))
        return paths

    # Test initialization
    print("\n--- Main Test: Initialization ---")
    paths = _main_dummy_generator(1)
    test_env_main = AlmPenaltyEnv(instance_paths=paths)
    print(f"Action space: {test_env_main.action_space}")
    print(f"Observation space: {test_env_main.observation_space}")

    # Test reset
    print("\n--- Main Test: Reset ---")
    obs_main, info_main = test_env_main.reset()
    print(f"Reset Obs: {obs_main}, Info: {info_main}")
    
    # Test step (mocking would be more involved here, so just a conceptual call)
    print("\n--- Main Test: Step (conceptual without deep mock) ---")
    action_main = test_env_main.action_space.sample()
    print(f"Sample Action: {action_main}")
    # In a real test, AlmOptimizer would be mocked as above.
    # We'll just call step and expect it to run with a real AlmOptimizer if one is found.
    # This might be slow or error if AlmOptimizer dependencies aren't perfect.
    try:
        _, reward_main, _, _, info_step_main = test_env_main.step(action_main)
        print(f"Step Reward: {reward_main}, Info: {info_step_main}")
    except Exception as e:
        print(f"Error during main step test (possibly unmocked AlmOptimizer): {e}")
        print("This is expected if run directly without full pytest mocking setup for AlmOptimizer.")

    # Clean up
    for p in paths:
        if os.path.exists(p): os.remove(p)
    if os.path.exists(str(main_tmp_path / "dummy_cvrp_instances_for_main_test")):
        if not os.listdir(str(main_tmp_path / "dummy_cvrp_instances_for_main_test")):
            os.rmdir(str(main_tmp_path / "dummy_cvrp_instances_for_main_test"))
    if os.path.exists(temp_dir_for_main):
        if not os.listdir(temp_dir_for_main):
            os.rmdir(temp_dir_for_main)