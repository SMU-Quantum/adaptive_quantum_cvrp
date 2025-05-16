# cvrp_tripartite_solver/src/common/solution.py

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class CVRPSolution:
    """
    Represents a solution to the Capacitated Vehicle Routing Problem.

    Attributes:
        routes: A list of routes. Each route is a list of 0-indexed node IDs,
                starting and ending with the depot.
                Example: [[0, 1, 2, 0], [0, 3, 0]]
        total_cost: The total cost (e.g., distance) of all routes in the solution.
        num_vehicles_used: The number of vehicles used, which is simply len(routes).
        instance_name: Optional name of the CVRP instance this solution is for.
        is_feasible: Optional boolean indicating if the solution is known to be feasible
                     according to all hard CVRP constraints. This might be set by an evaluator.
        feasibility_details: Optional dictionary or string providing details about
                             constraint violations if not feasible.
    """
    routes: List[List[int]]
    total_cost: float # Using float for cost to be general, can be int if distances are always int
    instance_name: Optional[str] = None
    is_feasible: Optional[bool] = None
    feasibility_details: Optional[dict] = field(default_factory=dict) # Stores details of violations

    @property
    def num_vehicles_used(self) -> int:
        """Returns the number of vehicles used in this solution."""
        return len(self.routes)

    def __str__(self) -> str:
        route_str = "\n  ".join(map(str, self.routes))
        return (
            f"CVRPSolution for '{self.instance_name}':\n"
            f"  Total Cost: {self.total_cost}\n"
            f"  Vehicles Used: {self.num_vehicles_used}\n"
            f"  Is Feasible: {self.is_feasible}\n"
            f"  Routes:\n  {route_str}"
        )

if __name__ == '__main__':
    # Example Usage:
    sol_feasible = CVRPSolution(
        routes=[[0, 1, 3, 0], [0, 2, 4, 0]],
        total_cost=125.5,
        instance_name="Example1",
        is_feasible=True
    )
    print(sol_feasible)

    sol_infeasible = CVRPSolution(
        routes=[[0, 1, 0, 2, 0]], # Invalid route
        total_cost=90.0,
        instance_name="Example2",
        is_feasible=False,
        feasibility_details={"route_0": "Invalid structure"}
    )
    print(f"\n{sol_infeasible}")
    print(f"Feasibility details for Example2: {sol_infeasible.feasibility_details}")

