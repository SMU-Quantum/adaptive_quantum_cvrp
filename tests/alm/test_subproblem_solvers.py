# cvrp_tripartite_solver/tests/alm/test_subproblem_solvers.py

import pytest
import random
import heapq 
from common.cvrp_instance import CVRPInstance
from alm.subproblem_solvers import solve_esp_with_dominance, ESPathLabel 

@pytest.fixture
def esp_instance_fixture() -> CVRPInstance:
    return CVRPInstance(
        name="esp_test_inst",
        dimension=4, 
        capacity=50, 
        distance_matrix=[
            [0, 10, 100, 12], 
            [10, 0,   8, 100], 
            [100, 8,  0,   15], 
            [12, 100, 15, 0]
        ],
        demands=[0, 20, 10, 25], 
        depot=0
    )

@pytest.fixture
def esp_instance_for_tabu() -> CVRPInstance: 
    return CVRPInstance(
        name="esp_tabu_test",
        dimension=5, 
        capacity=100,
        distance_matrix=[
            [0, 10, 15, 20, 25], 
            [10, 0,  5, 30, 35], 
            [15, 5,  0,  8, 40],   
            [20, 30, 8,  0, 12],  
            [25, 35, 40, 12, 0]   
        ],
        demands=[0, 10, 20, 30, 15], 
        depot=0,
    )

# --- Tests for ESPathLabel dataclass ---
def test_esplabel_creation_and_properties():
    label = ESPathLabel(cost=10.5, current_node=1, path=(0, 1))
    assert label.cost == 10.5
    assert label.current_node == 1
    assert label.path == (0, 1)

def test_esplabel_comparison_for_sorting_and_heapq():
    label1 = ESPathLabel(cost=10.0, current_node=1, path=(0,1))
    label2 = ESPathLabel(cost=5.0, current_node=2, path=(0,2))
    label3 = ESPathLabel(cost=10.0, current_node=3, path=(0,3))
    
    pq = []
    heapq.heappush(pq, (label1.cost, 1, label1)) 
    heapq.heappush(pq, (label2.cost, 2, label2))
    heapq.heappush(pq, (label3.cost, 3, label3))

    cost_l2, _, l2_obj = heapq.heappop(pq)
    assert l2_obj == label2
    cost_next, _, next_obj = heapq.heappop(pq)
    assert cost_next == 10.0
    assert next_obj in [label1, label3]

def test_esplabel_hashable_and_equality():
    label1 = ESPathLabel(cost=10.0, current_node=1, path=(0,1))
    label2 = ESPathLabel(cost=10.0, current_node=1, path=(0,1)) 
    label3 = ESPathLabel(cost=10.0, current_node=2, path=(0,1,2)) 
    label4 = ESPathLabel(cost=11.0, current_node=1, path=(0,1)) 
    
    assert label1 == label2 
    assert label1 != label3 
    assert label1 != label4 
    
    s = {label1, label3, label4}
    assert label1 in s
    assert label2 in s 
    assert label3 in s
    assert label4 in s

def test_esplabel_dominance_refined(esp_instance_fixture):
    node = 1
    path1 = (0, node)
    
    l_base = ESPathLabel(cost=10, current_node=node, path=path1)
    l_better_cost_same_path = ESPathLabel(cost=8, current_node=node, path=path1)
    
    assert l_better_cost_same_path.dominates(l_base) is True 
    assert l_base.dominates(l_better_cost_same_path) is False

    path2_different_history_same_end = (0, 2, node) 
    l_different_path_hist = ESPathLabel(cost=7, current_node=node, path=path2_different_history_same_end)
    assert l_different_path_hist.dominates(l_base) is False 
    assert l_base.dominates(l_different_path_hist) is False
    
    l_equal_cost_same_path = ESPathLabel(cost=10, current_node=node, path=path1)
    assert not l_base.dominates(l_equal_cost_same_path) 
    assert not l_equal_cost_same_path.dominates(l_base)


# --- Tests for solve_esp_with_dominance ---
def test_esp_simple_direct_route(esp_instance_fixture):
    instance = esp_instance_fixture
    rewards = {1: 50.0, 2: -100.0, 3: -100.0} 
    route = solve_esp_with_dominance(instance, rewards)
    assert route == [0, 1, 0]

def test_esp_elementarity(esp_instance_fixture):
    instance = esp_instance_fixture
    rewards = { 1: 30.0, 2: 30.0, 3: -100.0 }
    route = solve_esp_with_dominance(instance, rewards)
    assert route is not None
    if len(route) > 2:
        customers_in_route = route[1:-1]
        assert len(customers_in_route) == len(set(customers_in_route))
    assert route == [0,1,0]

def test_esp_selects_best_modified_cost_route(esp_instance_fixture):
    instance = esp_instance_fixture
    rewards1 = {1: 5.0, 2: -100.0, 3: 0.0} 
    route1 = solve_esp_with_dominance(instance, rewards1)
    assert route1 == [0, 1, 0]

    rewards2 = {1: 0.0, 2: -100.0, 3: 5.0} 
    route2 = solve_esp_with_dominance(instance, rewards2)
    assert route2 == [0, 3, 0]

def test_esp_handles_no_customers_instance():
    instance_no_cust = CVRPInstance(
        name="no_cust", dimension=1, capacity=100,
        distance_matrix=[[0]], demands=[0], depot=0
    )
    rewards = {}
    route = solve_esp_with_dominance(instance_no_cust, rewards)
    assert route == [0,0]

def test_esp_with_tabu_customers(esp_instance_for_tabu): # Corrected fixture name
    instance = esp_instance_for_tabu
    rewards = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0}

    route_no_tabu = solve_esp_with_dominance(instance, rewards, tabu_customers=set())
    assert route_no_tabu is not None
    assert 1 in route_no_tabu 

    route_tabu_1 = solve_esp_with_dominance(instance, rewards, tabu_customers={1})
    assert route_tabu_1 is not None
    assert 1 not in route_tabu_1[1:-1]
    assert 2 in route_tabu_1 

    route_tabu_1_2 = solve_esp_with_dominance(instance, rewards, tabu_customers={1, 2})
    assert route_tabu_1_2 is not None
    assert 1 not in route_tabu_1_2[1:-1]
    assert 2 not in route_tabu_1_2[1:-1]
    assert 3 in route_tabu_1_2 
    
    all_customers_tabu = {1,2,3,4} 
    route_all_tabu = solve_esp_with_dominance(instance, rewards, tabu_customers=all_customers_tabu)
    assert route_all_tabu is None

def test_esp_returns_none_if_no_route_possible(esp_instance_fixture):
    instance = esp_instance_fixture
    all_cust_tabu = {1, 2, 3} 
    rewards = {1:0, 2:0, 3:0}
    route = solve_esp_with_dominance(instance, rewards, tabu_customers=all_cust_tabu)
    assert route is None

def test_esp_ignores_capacity(esp_instance_fixture): 
    instance = esp_instance_fixture 
    rewards = {1: 10.0, 2: 10.0, 3: 10.0} 
    route = solve_esp_with_dominance(instance, rewards)
    assert route is not None 
    assert route == [0,1,0]