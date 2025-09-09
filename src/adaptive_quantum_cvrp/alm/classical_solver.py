# src/adaptive_quantum_cvrp/alm/classical_solver.py

"""
contains the implementation of the classical heuristic subproblem solver,
It's designed to be a standalone component that solves the Lagrangian subproblem for a given set of penalties.
So we can throw in classical or quantum solvers as needed.
"""

import numpy as np
from typing import Protocol, List

from ..common.cvrp_instance import CVRPInstance
from ..common.cvrp_solution import CVRPSolution

class SubproblemSolver(Protocol):
    """
    A protocol defining the interface for a CVRP subproblem solver.

    This ensures that any solver (classical, quantum, etc.) can be used
    interchangeably with the ALMOptimizer.
    """
    def solve(self, instance: CVRPInstance, lagrange_multipliers: np.ndarray,
              penalty_mu: float) -> CVRPSolution:
        """
        Solves the Lagrangian-penalized subproblem.

        Args:
            instance: The CVRP problem instance.
            lagrange_multipliers: The current Lagrange multipliers (lambda).
            penalty_mu: The current penalty parameter (mu).

        Returns:
            The best found solution for the subproblem.
        """
        ...

class ClassicalSolver:
    """
    A classical heuristic solver for the ALM subproblem.

    This solver iteratively constructs routes using a greedy approach,
    penalized by the Lagrangian terms. It implements the SubproblemSolver protocol.
    """

    def solve(self, instance: CVRPInstance, lagrange_multipliers: np.ndarray,
              penalty_mu: float) -> CVRPSolution:
        """
        Constructs a solution using a penalized greedy heuristic.
        """
        routes: List[List[int]] = []
        customers_to_visit = set(range(1, instance.num_customers + 1))
        dist_matrix = instance.dist_matrix

        while customers_to_visit:
            current_route: List[int] = []
            current_capacity = instance.capacity
            last_node = instance.depot_id

            while customers_to_visit:
                best_customer = -1
                min_cost = float('inf')

                for customer in customers_to_visit:
                    if instance.demands[customer] <= current_capacity:
                        # Cost calculation includes original travel cost plus penalties
                        # for the 'visit exactly once' constraint.
                        lagrangian_cost = (
                            dist_matrix[last_node, customer]
                            - lagrange_multipliers[customer - 1]
                            - (1 / (2 * penalty_mu)) * (lagrange_multipliers[customer - 1] ** 2)
                        )
                        
                        if lagrangian_cost < min_cost:
                            min_cost = lagrangian_cost
                            best_customer = customer
                
                if best_customer != -1:
                    current_route.append(best_customer)
                    current_capacity -= instance.demands[best_customer]
                    customers_to_visit.remove(best_customer)
                    last_node = best_customer
                else:
                    # No more customers can be added to this route
                    break
            
            if current_route:
                routes.append(current_route)
        
        return CVRPSolution(routes=routes)