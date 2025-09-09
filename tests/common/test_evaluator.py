# tests/common/test_evaluator.py

import numpy as np
import pytest

from src.adaptive_quantum_cvrp.common.cvrp_instance import CVRPInstance
from src.adaptive_quantum_cvrp.common.cvrp_solution import CVRPSolution
from src.adaptive_quantum_cvrp.common.evaluator import CVRPEvaluator

@pytest.fixture
def simple_instance():
    """Creates a simple, predictable CVRP instance for testing."""
    # Depot at (0,0), Customer 1 at (3,0), Customer 2 at (0,4)
    # Distances: D-1=3, D-2=4, 1-2=5
    nodes = np.array([[0, 0], [3, 0], [0, 4]])
    demands = np.array([0, 10, 10])
    return CVRPInstance(
        name="test_instance",
        num_customers=2,
        capacity=15,
        nodes=nodes,
        demands=demands
    )

def test_cost_calculation(simple_instance):
    """Tests that the cost of a simple solution is calculated correctly."""
    # Route: Depot -> 1 -> 2 -> Depot
    # Cost: 3 (D->1) + 5 (1->2) + 4 (2->D) = 12
    solution = CVRPSolution(routes=[[1, 2]])
    cost = CVRPEvaluator.get_solution_cost(simple_instance, solution)
    assert cost == pytest.approx(12.0)

def test_feasible_solution(simple_instance):
    """Tests that a valid solution passes the feasibility check."""
    # Route: Depot -> 1 -> Depot, Depot -> 2 -> Depot
    # Each route demand is 10, which is <= capacity (15)
    solution = CVRPSolution(routes=[[1], [2]])
    is_feasible, violations = CVRPEvaluator.check_feasibility(simple_instance, solution)
    assert is_feasible
    assert not violations

def test_infeasible_capacity_solution(simple_instance):
    """Tests that a solution violating capacity constraints is caught."""
    # Route: Depot -> 1 -> 2 -> Depot
    # Total demand is 20, which is > capacity (15)
    solution = CVRPSolution(routes=[[1, 2]])
    is_feasible, violations = CVRPEvaluator.check_feasibility(simple_instance, solution)
    assert not is_feasible
    assert "exceeds capacity" in violations[0]

def test_infeasible_unvisited_customer(simple_instance):
    """Tests that a solution with unvisited customers is caught."""
    solution = CVRPSolution(routes=[[1]])
    is_feasible, violations = CVRPEvaluator.check_feasibility(simple_instance, solution)
    assert not is_feasible
    assert "Unvisited customers: [2]" in violations[0]