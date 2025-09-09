# src/adaptive_quantum_cvrp/alm/optimizer.py


"""
The ALMOptimizer class manages the main loop, updates parameters, and uses the provided subproblem_solver to find solutions.

It is decoupled from how the subproblems are solved.
"""

import numpy as np
from typing import Dict, Any, List, Optional

from ..common.cvrp_instance import CVRPInstance
from ..common.cvrp_solution import CVRPSolution
from ..common.evaluator import CVRPEvaluator
from .classical_solver import SubproblemSolver, ClassicalSolver

class ALMOptimizer:
    """
    Manages the Augmented Lagrangian Method (ALM) optimization loop for CVRP.

    This class orchestrates the ALM process by iteratively calling a
    subproblem solver and updating the Lagrange multipliers and penalty
    parameters.

    Attributes:
        instance (CVRPInstance): The CVRP problem to solve.
        max_iterations (int): The maximum number of ALM iterations.
        subproblem_solver (SubproblemSolver): The solver to use for the
                                              Lagrangian subproblems.
    """
    def __init__(self, instance: CVRPInstance, max_iterations: int,
                 subproblem_solver: SubproblemSolver):
        """
        Initializes the ALMOptimizer.

        Args:
            instance: The CVRP problem instance.
            max_iterations: Max number of ALM iterations.
            subproblem_solver: An object that implements the SubproblemSolver
                               protocol (e.g., ClassicalSolver).
        """
        self.instance = instance
        self.max_iterations = max_iterations
        self.subproblem_solver = subproblem_solver
        
        # Initialize Lagrange multipliers (lambda) for the 'visit once' constraint
        self.lagrange_multipliers = np.zeros(self.instance.num_customers)
        
        self.best_feasible_solution: Optional[CVRPSolution] = None
        self.best_feasible_cost = float('inf')
        self.iteration_log: List[Dict[str, Any]] = []

    def _update_lagrange_multipliers(self, solution: CVRPSolution, penalty_mu: float) -> None:
        """
        Updates the Lagrange multipliers based on constraint violations.
        
        Update rule from the paper:
        lambda_{j}^{t+1} = lambda_{j}^{t} + (1/mu) * g_{j}(S^{(t)})
        where g_j is the violation for customer j.
        """
        # Get customer visit counts
        visits = np.zeros(self.instance.num_customers)
        for route in solution.routes:
            for customer_id in route:
                visits[customer_id - 1] += 1
        
        # g_j(S) = visits_j - 1
        violations = visits - 1
        
        self.lagrange_multipliers += (1 / penalty_mu) * violations

    def solve(self, initial_penalty_mu: float = 1.0) -> Dict[str, Any]:
        """
        Executes the main ALM optimization loop.

        Args:
            initial_penalty_mu: The starting value for the penalty parameter mu.

        Returns:
            A dictionary containing the best found feasible solution, its cost,
            and the detailed iteration log.
        """
        penalty_mu = initial_penalty_mu
        
        for i in range(self.max_iterations):
            # 1. Solve the Lagrangian subproblem
            current_solution = self.subproblem_solver.solve(
                self.instance, self.lagrange_multipliers, penalty_mu
            )
            
            # 2. Evaluate the solution
            cost = CVRPEvaluator.get_solution_cost(self.instance, current_solution)
            is_feasible, violations = CVRPEvaluator.check_feasibility(self.instance, current_solution)

            log_entry = {
                "iteration": i + 1,
                "cost": cost,
                "is_feasible": is_feasible,
                "violations": violations,
                "penalty_mu": penalty_mu
            }
            self.iteration_log.append(log_entry)
            
            # 3. Update best feasible solution found so far
            if is_feasible and cost < self.best_feasible_cost:
                self.best_feasible_solution = current_solution
                self.best_feasible_cost = cost
                print(f"Iter {i+1}: New best feasible solution found with cost {cost:.2f}")

            # 4. Update Lagrange multipliers
            self._update_lagrange_multipliers(current_solution, penalty_mu)

            # Note: The penalty parameter `mu` is constant here. In the RL-enhanced
            # version, the RL agent will be responsible for updating it each step.

        return {
            "best_solution": self.best_feasible_solution,
            "best_cost": self.best_feasible_cost,
            "log": self.iteration_log
        }