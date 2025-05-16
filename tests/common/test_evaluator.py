# cvrp_tripartite_solver/tests/common/test_evaluator.py

import pytest
from common.cvrp_instance import CVRPInstance
from common.cvrp_solution import CVRPSolution
from common.cvrp_evaluator import calculate_solution_cost, check_solution_feasibility

@pytest.fixture
def sample_instance() -> CVRPInstance:
    """Returns a simple CVRPInstance for testing."""
    return CVRPInstance(
        name="sample_instance",
        dimension=5, # Depot 0, Customers 1, 2, 3, 4
        capacity=100,
        distance_matrix=[ # Symmetric
            [0, 10, 20, 15, 25], # Depot 0
            [10, 0,  8, 12, 18], # Cust 1
            [20,  8, 0, 14, 22], # Cust 2
            [15, 12, 14, 0, 10], # Cust 3
            [25, 18, 22, 10, 0]  # Cust 4
        ],
        demands=[0, 30, 40, 20, 35], # d0, d1, d2, d3, d4
        depot=0,
        coordinates=None # Not needed for cost calculation if matrix is explicit
    )

@pytest.fixture
def sample_instance_depot_not_0() -> CVRPInstance:
    """Instance where depot is not node 0."""
    return CVRPInstance(
        name="depot_not_0_instance",
        dimension=4, # Nodes 0,1,2,3. Depot is 3. Cust are 0,1,2
        capacity=50,
        distance_matrix=[
            [0, 5, 6, 2], # Cust 0
            [5, 0, 7, 3], # Cust 1
            [6, 7, 0, 4], # Cust 2
            [2, 3, 4, 0]  # Depot 3
        ],
        demands=[10, 15, 20, 0], # d0, d1, d2, d3 (depot)
        depot=3
    )


# --- Tests for calculate_solution_cost ---
def test_calculate_cost_empty_solution(sample_instance):
    assert calculate_solution_cost([], sample_instance) == 0.0

def test_calculate_cost_single_route(sample_instance):
    routes = [[0, 1, 3, 0]] # 0->1 (10) + 1->3 (12) + 3->0 (15) = 37
    assert calculate_solution_cost(routes, sample_instance) == 37.0

def test_calculate_cost_multiple_routes(sample_instance):
    routes = [
        [0, 1, 0],      # 0->1 (10) + 1->0 (10) = 20
        [0, 2, 4, 0]    # 0->2 (20) + 2->4 (22) + 4->0 (25) = 67
    ]                   # Total = 20 + 67 = 87
    assert calculate_solution_cost(routes, sample_instance) == 87.0

def test_calculate_cost_route_with_one_customer(sample_instance):
    routes = [[0, 4, 0]] # 0->4 (25) + 4->0 (25) = 50
    assert calculate_solution_cost(routes, sample_instance) == 50.0

def test_calculate_cost_invalid_node_index(sample_instance):
    routes = [[0, 1, 99, 0]] # Node 99 is invalid
    with pytest.raises(ValueError, match="Invalid node index in route"):
        calculate_solution_cost(routes, sample_instance)

# --- Tests for check_solution_feasibility ---
def test_feasible_solution(sample_instance):
    # Route 1: 0-1(30)-3(20)-0. Demand = 50. Cost = 10+12+15 = 37
    # Route 2: 0-2(40)-4(35)-0. Demand = 75. Cost = 20+22+25 = 67
    # Total cost = 104
    routes = [[0, 1, 3, 0], [0, 2, 4, 0]]
    cost = calculate_solution_cost(routes, sample_instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is True
    assert not violations
    assert solution.is_feasible is True # Check if solution object is updated

def test_infeasible_capacity_exceeded(sample_instance):
    # Route 1: 0-1(30)-2(40)-0. Demand = 70.
    # Route 2: 0-3(20)-4(35)-0. Demand = 55.
    # Let's make Route 1 exceed capacity: 0-1(30)-2(40)-4(35)-0. Demand = 30+40+35 = 105 > 100
    routes = [[0, 1, 2, 4, 0]]
    cost = calculate_solution_cost(routes, sample_instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "route_0_capacity" in violations

def test_infeasible_unvisited_customer(sample_instance):
    routes = [[0, 1, 0], [0, 2, 0]] # Customers 3 and 4 are unvisited
    cost = calculate_solution_cost(routes, sample_instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "unvisited_customers" in violations
    assert "3" in violations["unvisited_customers"] and "4" in violations["unvisited_customers"]

def test_infeasible_customer_visited_twice(sample_instance):
    routes = [[0, 1, 2, 0], [0, 1, 3, 0]] # Customer 1 visited in two routes
    cost = calculate_solution_cost(routes, sample_instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "customer_1_multiple_visits" in violations

def test_infeasible_route_not_start_at_depot(sample_instance):
    routes = [[1, 2, 0]] # Starts at customer 1
    cost = calculate_solution_cost(routes, sample_instance) # Cost calculation might still work
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "route_0_start" in violations

def test_infeasible_route_not_end_at_depot(sample_instance):
    routes = [[0, 2, 1]] # Ends at customer 1
    cost = calculate_solution_cost(routes, sample_instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "route_0_end" in violations

def test_empty_solution_with_customers(sample_instance):
    solution = CVRPSolution(routes=[], total_cost=0, instance_name=sample_instance.name)
    is_feasible, violations = check_solution_feasibility(solution, sample_instance)
    assert is_feasible is False
    assert "general" in violations
    assert "No routes provided" in violations["general"]

def test_empty_solution_no_customers():
    instance_no_cust = CVRPInstance(
        name="no_cust", dimension=1, capacity=100,
        distance_matrix=[[0]], demands=[0], depot=0
    )
    solution = CVRPSolution(routes=[], total_cost=0, instance_name=instance_no_cust.name)
    is_feasible, violations = check_solution_feasibility(solution, instance_no_cust)
    assert is_feasible is True
    assert not violations

def test_feasible_solution_depot_not_0(sample_instance_depot_not_0):
    # Instance: dim=4, depot=3. Cust: 0,1,2. Demands: d0=10, d1=15, d2=20. Cap=50
    # Route 1: 3-0(10)-1(15)-3. Demand = 25. Cost = 2+5+3 = 10
    # Route 2: 3-2(20)-3. Demand = 20. Cost = 4+4 = 8
    # Total cost = 18
    instance = sample_instance_depot_not_0
    routes = [[3, 0, 1, 3], [3, 2, 3]]
    cost = calculate_solution_cost(routes, instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=instance.name)
    is_feasible, violations = check_solution_feasibility(solution, instance)
    
    # print("\nViolations (depot_not_0):", violations) # For debugging
    assert is_feasible is True
    assert not violations

def test_infeasible_unvisited_customer_depot_not_0(sample_instance_depot_not_0):
    instance = sample_instance_depot_not_0
    routes = [[3, 0, 3]] # Cust 1 and 2 unvisited
    cost = calculate_solution_cost(routes, instance)
    solution = CVRPSolution(routes=routes, total_cost=cost, instance_name=instance.name)
    is_feasible, violations = check_solution_feasibility(solution, instance)
    assert is_feasible is False
    assert "unvisited_customers" in violations
    assert "1" in violations["unvisited_customers"] and "2" in violations["unvisited_customers"]

if __name__ == '__main__':
    pytest.main()
