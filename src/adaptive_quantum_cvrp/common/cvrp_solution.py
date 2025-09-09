# src/adaptive_quantum_cvrp/common/cvrp_solution.py

"""
This class provides a clear and simple structure for representing a solution. 
It also includes type hints and a helpful __repr__ for easy debugging.
"""

from typing import List

class CVRPSolution:
    """
    Represents a solution to a CVRP instance.

    A solution consists of a set of routes, where each route is a list of
    customer IDs, starting and ending at the depot (implicitly).

    Attributes:
        routes (List[List[int]]): A list of routes. Each route is a list of
                                  customer node IDs in the order they are visited.
                                  The depot (node 0) is not explicitly included in the
                                  customer lists. E.g., [[1, 2], [3, 4]].
    """
    def __init__(self, routes: List[List[int]]):
        """Initializes the CVRPSolution."""
        # Ensure routes are valid (no empty routes unless it's the only one)
        self.routes = [route for route in routes if route] or [[]]

    def __repr__(self) -> str:
        return f"CVRPSolution(routes={self.routes})"

    @property
    def num_vehicles(self) -> int:
        """Returns the number of vehicles used in the solution."""
        return len(self.routes)