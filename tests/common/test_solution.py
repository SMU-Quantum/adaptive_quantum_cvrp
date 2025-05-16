# cvrp_tripartite_solver/tests/common/test_solution.py

import pytest
from common.cvrp_solution import CVRPSolution

def test_cvrp_solution_creation_basic():
    """Test basic creation of a CVRPSolution object."""
    routes = [[0, 1, 2, 0], [0, 3, 0]]
    cost = 150.0
    instance_name = "test_instance_1"
    
    solution = CVRPSolution(
        routes=routes,
        total_cost=cost,
        instance_name=instance_name,
        is_feasible=True
    )
    
    assert solution.routes == routes
    assert solution.total_cost == cost
    assert solution.instance_name == instance_name
    assert solution.is_feasible is True
    assert solution.num_vehicles_used == 2
    assert len(solution.feasibility_details) == 0

def test_cvrp_solution_num_vehicles():
    """Test the num_vehicles_used property."""
    solution1 = CVRPSolution(routes=[[0,1,0]], total_cost=10)
    assert solution1.num_vehicles_used == 1
    
    solution2 = CVRPSolution(routes=[[0,1,0], [0,2,0], [0,3,0]], total_cost=30)
    assert solution2.num_vehicles_used == 3
    
    solution_empty = CVRPSolution(routes=[], total_cost=0) # An empty solution
    assert solution_empty.num_vehicles_used == 0

def test_cvrp_solution_feasibility_details():
    """Test storing feasibility details."""
    details = {"capacity_violation_route_0": "Exceeded by 10 units"}
    solution = CVRPSolution(
        routes=[[0, 1, 0]],
        total_cost=20.0,
        is_feasible=False,
        feasibility_details=details.copy() # Pass a copy
    )
    assert solution.is_feasible is False
    assert solution.feasibility_details == details

def test_cvrp_solution_str_representation():
    """Test the string representation of the solution."""
    solution = CVRPSolution(
        routes=[[0, 1, 0], [0, 2, 3, 0]],
        total_cost=55.5,
        instance_name="P-n16-k8",
        is_feasible=True
    )
    s = str(solution)
    assert "CVRPSolution for 'P-n16-k8'" in s
    assert "Total Cost: 55.5" in s
    assert "Vehicles Used: 2" in s
    assert "Is Feasible: True" in s
    assert "[0, 1, 0]" in s
    assert "[0, 2, 3, 0]" in s

if __name__ == '__main__':
    # This block is for manual execution if needed, but tests are run with pytest
    pytest.main() # You can run pytest from the command line
