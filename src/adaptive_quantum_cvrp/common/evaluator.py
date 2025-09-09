# src/adaptive_quantum_cvrp/common/evaluator.py

"""
It takes a CVRPInstance and a CVRPSolution and returns evaluation metrics. 
"""


from .cvrp_instance import CVRPInstance
from .cvrp_solution import CVRPSolution
from typing import List, Tuple

class CVRPEvaluator:
    """
    A stateless evaluator for CVRP solutions.

    Provides static methods to calculate the total cost of a solution and
    check its feasibility against the problem constraints.
    """

    @staticmethod
    def get_solution_cost(instance: CVRPInstance, solution: CVRPSolution) -> float:
        """
        Calculates the total travel cost of a given solution.

        The cost is the sum of the Euclidean distances of all routes.

        Args:
            instance (CVRPInstance): The CVRP problem instance.
            solution (CVRPSolution): The solution to evaluate.

        Returns:
            float: The total cost of the solution.
        """
        total_cost = 0.0
        dist_matrix = instance.dist_matrix
        depot = instance.depot_id

        for route in solution.routes:
            if not route:
                continue
            
            # Distance from depot to the first customer
            total_cost += dist_matrix[depot, route[0]]
            
            # Distance between customers in the route
            for i in range(len(route) - 1):
                total_cost += dist_matrix[route[i], route[i+1]]
            
            # Distance from the last customer back to the depot
            total_cost += dist_matrix[route[-1], depot]
            
        return total_cost

    @staticmethod
    def check_feasibility(instance: CVRPInstance, solution: CVRPSolution) -> Tuple[bool, List[str]]:
        """
        Checks if a solution is feasible.

        A solution is feasible if:
        1. All customers are visited exactly once.
        2. The total demand of each route does not exceed the vehicle capacity.

        Args:
            instance (CVRPInstance): The CVRP problem instance.
            solution (CVRPSolution): The solution to check.

        Returns:
            Tuple[bool, List[str]]: A tuple containing a boolean indicating
            feasibility and a list of strings describing any violations.
        """
        violations = []
        
        # 1. Check capacity constraints
        for i, route in enumerate(solution.routes):
            route_demand = sum(instance.demands[customer_id] for customer_id in route)
            if route_demand > instance.capacity:
                violations.append(f"Route {i} exceeds capacity: {route_demand} > {instance.capacity}")

        # 2. Check customer visit constraints
        all_visited_customers = [customer for route in solution.routes for customer in route]
        
        # Check for duplicates
        if len(all_visited_customers) != len(set(all_visited_customers)):
            violations.append("Some customers are visited more than once.")
        
        # Check for unvisited customers
        required_customers = set(range(1, instance.num_customers + 1))
        visited_customers_set = set(all_visited_customers)
        
        unvisited = required_customers - visited_customers_set
        if unvisited:
            violations.append(f"Unvisited customers: {sorted(list(unvisited))}")

        return not violations, violations