# cvrp_tripartite_solver/tests/alm/test_alm_optimizer.py

import pytest
from unittest.mock import patch, call 

from common.cvrp_instance import CVRPInstance
from common.cvrp_solution import CVRPSolution
from alm.alm_optimizer import AlmOptimizer 


@pytest.fixture
def simple_instance_alm_option_b() -> CVRPInstance:
    return CVRPInstance(
        name="alm_esp_cap_relaxed_test",
        dimension=3, 
        capacity=30, 
        distance_matrix=[[0, 10, 12], [10, 0, 5], [12, 5, 0]],
        demands=[0, 20, 25], 
        depot=0,
        num_vehicles_comment=2 
    )

def test_alm_optimizer_initialization_option_b(simple_instance_alm_option_b):
    optimizer = AlmOptimizer(
        instance=simple_instance_alm_option_b,
        initial_penalty_rate=1.0, 
        initial_capacity_penalty_rate=0.5, 
        subproblem_max_vehicles=2 
    )
    assert optimizer.instance == simple_instance_alm_option_b
    assert len(optimizer.lambdas) == 2 
    assert all(l == 0.0 for l in optimizer.lambdas)
    assert len(optimizer.rhos) == 2
    assert all(r == 1.0 for r in optimizer.rhos)
    assert hasattr(optimizer, 'mus_capacity')
    assert hasattr(optimizer, 'sigmas_capacity')
    assert len(optimizer.mus_capacity) == 2 
    assert all(mu == 0.0 for mu in optimizer.mus_capacity)
    assert len(optimizer.sigmas_capacity) == 2
    assert all(sig == 0.5 for sig in optimizer.sigmas_capacity)

@patch('alm.alm_optimizer.solve_esp_with_dominance') 
def test_alm_option_b_iteration_lambda_mu_update(mock_solve_esp, simple_instance_alm_option_b):
    mock_solve_esp.return_value = [0, 1, 2, 0] 
    optimizer = AlmOptimizer(
        instance=simple_instance_alm_option_b,
        initial_penalty_rate=1.0,          
        initial_capacity_penalty_rate=0.5, 
        capacity_penalty_increase_factor=1.2, 
        max_alm_iterations=1, 
        subproblem_max_vehicles=1,
        verbose=0 
    )
    optimizer.solve()
    idx_c1 = optimizer._get_customer_lambda_rho_idx(1)
    idx_c2 = optimizer._get_customer_lambda_rho_idx(2)
    assert optimizer.lambdas[idx_c1] == pytest.approx(0.0)
    assert optimizer.lambdas[idx_c2] == pytest.approx(0.0)
    assert optimizer.rhos[idx_c1] == pytest.approx(1.0)
    assert optimizer.rhos[idx_c2] == pytest.approx(1.0)
    assert optimizer.mus_capacity[0] == pytest.approx(7.5)
    assert optimizer.sigmas_capacity[0] == pytest.approx(0.5 * 1.2)
    
    expected_rewards_cust_visit = {1: -0.0, 2: -0.0} 
    # Based on Pytest Actual Output: tabu_customers was {1}
    # This implies the mock was called after customer 1 was processed by the mock,
    # which is not what happens with subproblem_max_vehicles=1.
    # The correct expectation for the *first and only call* is tabu_customers=set().
    expected_call_obj = call(
        simple_instance_alm_option_b, 
        modified_node_rewards=expected_rewards_cust_visit,
        tabu_customers=set(), # Corrected expectation for the first call
        debug_esp=False,
        capacity_multiplier=0.0 # new kwarg 
    )
    assert mock_solve_esp.call_args_list == [expected_call_obj]
    assert mock_solve_esp.call_count == 1
    assert len(optimizer.iteration_log) == 1
    assert optimizer.iteration_log[0]['max_cap_violation'] == 15.0

@patch('alm.alm_optimizer.solve_esp_with_dominance') 
def test_alm_option_b_convergence_no_violations(mock_solve_esp, simple_instance_alm_option_b):
    mock_solve_esp.side_effect = [ [0, 1, 0], [0, 2, 0] ]
    optimizer = AlmOptimizer(
        instance=simple_instance_alm_option_b,
        initial_penalty_rate=1.0,
        initial_capacity_penalty_rate=0.5,
        max_alm_iterations=5,
        convergence_tolerance=0.01, 
        capacity_convergence_tolerance=0.1, 
        subproblem_max_vehicles=2,
        verbose=0 
    )
    best_solution = optimizer.solve()
    assert best_solution is not None
    assert best_solution.is_feasible is True
    assert best_solution.total_cost == pytest.approx(44.0) 
    assert len(optimizer.iteration_log) == 1 
    assert optimizer.iteration_log[0]["max_abs_g_i"] == 0.0 
    assert optimizer.iteration_log[0]['max_cap_violation'] == 0.0 
    assert mock_solve_esp.call_count == 2
    
    expected_rewards = {1: -0.0, 2: -0.0}
    # Corrected expectations for tabu_customers based on AlmOptimizer logic
    expected_calls = [
        call(simple_instance_alm_option_b, modified_node_rewards=expected_rewards, tabu_customers=set(), debug_esp=False, capacity_multiplier=0.0), 
        call(simple_instance_alm_option_b, modified_node_rewards=expected_rewards, tabu_customers={1}, debug_esp=False, capacity_multiplier=0.0)  
    ]
    assert mock_solve_esp.call_args_list == expected_calls

@patch('alm.alm_optimizer.solve_esp_with_dominance') 
def test_alm_penalty_update(mock_solve_esp, simple_instance_alm_option_b): 
    mock_solve_esp.side_effect = [
        [0, 1, 0],  
        [0, 1, 0]   
    ]
    initial_rho = 1.0
    increase_factor = 1.5
    initial_sigma_cap = 0.8 
    cap_increase_factor = 1.3

    optimizer = AlmOptimizer(
        instance=simple_instance_alm_option_b,
        initial_penalty_rate=initial_rho,
        penalty_increase_factor=increase_factor,
        initial_capacity_penalty_rate=initial_sigma_cap,
        capacity_penalty_increase_factor=cap_increase_factor,
        max_alm_iterations=2,
        subproblem_max_vehicles=1,
        verbose=0 
    )
    optimizer.solve()
    assert len(optimizer.iteration_log) == 2
    idx_cust1 = optimizer._get_customer_lambda_rho_idx(1) 
    idx_cust2 = optimizer._get_customer_lambda_rho_idx(2) 
    
    assert optimizer.rhos[idx_cust1] == pytest.approx(initial_rho) 
    assert optimizer.rhos[idx_cust2] == pytest.approx(initial_rho * increase_factor * increase_factor)
    assert optimizer.sigmas_capacity[0] == pytest.approx(initial_sigma_cap)


def test_alm_optimizer_no_customers_instance_option_b(simple_instance_alm_option_b): 
    instance_depot_only = CVRPInstance(
        name="depot_only_esp_cap_relaxed",
        dimension=1, capacity=simple_instance_alm_option_b.capacity, 
        distance_matrix=[[0]], demands=[0], depot=0
    )
    optimizer = AlmOptimizer(instance=instance_depot_only, max_alm_iterations=1)
    solution = optimizer.solve() 
    
    assert optimizer.num_customers == 0
    assert len(optimizer.lambdas) == 0
    assert len(optimizer.rhos) == 0
    assert len(optimizer.mus_capacity) == 0 
    assert len(optimizer.sigmas_capacity) == 0
    assert solution is not None 
    assert solution.is_feasible is True
    assert not solution.routes 
    assert solution.total_cost == 0.0 
    assert len(optimizer.iteration_log) == 1 
    assert optimizer.iteration_log[0]["max_abs_g_i"] == 0.0
    assert optimizer.iteration_log[0].get('max_cap_violation', 0.0) == 0.0