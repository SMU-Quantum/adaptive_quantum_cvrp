# # # cvrp_tripartite_solver/src/common/evaluator.py

# # from typing import Dict, Tuple, List, Set
# # from common.cvrp_instance import CVRPInstance
# # from common.cvrp_solution import CVRPSolution

# # def calculate_solution_cost(
# #     solution_routes: List[List[int]],
# #     instance: CVRPInstance
# # ) -> float:
# #     """
# #     Calculates the total cost of a given set of routes for a CVRP instance.
# #     Assumes routes are lists of 0-indexed node IDs.
# #     """
# #     total_cost = 0.0
# #     if not instance.distance_matrix:
# #         raise ValueError("Instance distance matrix is not available.")

# #     for route in solution_routes:
# #         if not route or len(route) < 2: # An empty or single-node route has no cost
# #             continue 
# #         current_route_cost = 0.0
# #         for i in range(len(route) - 1):
# #             u = route[i]
# #             v = route[i+1]
# #             if not (0 <= u < instance.dimension and 0 <= v < instance.dimension):
# #                 raise ValueError(f"Invalid node index in route {route}. Max index: {instance.dimension - 1}")
# #             current_route_cost += instance.distance_matrix[u][v]
# #         total_cost += current_route_cost
# #     return total_cost


# # def check_solution_feasibility(
# #     solution: CVRPSolution,
# #     instance: CVRPInstance
# # ) -> Tuple[bool, Dict[str, str]]:
# #     """
# #     Checks the feasibility of a CVRPSolution against the CVRP instance constraints.

# #     Returns:
# #         A tuple (is_feasible: bool, violations: Dict[str, str])
# #         violations dictionary contains descriptions of any constraint violations found.
# #     """
# #     violations: Dict[str, str] = {}
# #     all_customers = set(range(1, instance.dimension)) # Customers are 1 to n-1 (if depot is 0)
# #                                                       # Or more generally, all non-depot nodes.
# #                                                       # Assuming depot is instance.depot
    
# #     if instance.depot < 0 or instance.depot >= instance.dimension:
# #         violations["instance_error"] = f"Invalid depot index {instance.depot} in instance."
# #         return False, violations # Cannot proceed if depot is invalid

# #     all_customers = set(i for i in range(instance.dimension) if i != instance.depot)
# #     visited_customers: Set[int] = set()

# #     if not solution.routes and all_customers: # No routes but customers exist
# #         violations["general"] = "No routes provided, but customers exist."
# #         # If there are no customers, an empty solution can be considered feasible.
# #         if not all_customers:
# #              return True, violations # No customers, no routes = feasible
# #         return False, violations


# #     for i, route in enumerate(solution.routes):
# #         route_idx_str = f"route_{i}"

# #         if not route:
# #             violations[f"{route_idx_str}_structure"] = "Route is empty."
# #             continue # Move to next route

# #         # 1. Check if route starts and ends at the depot
# #         if route[0] != instance.depot:
# #             violations[f"{route_idx_str}_start"] = f"Route does not start at depot {instance.depot}. Starts at {route[0]}."
# #         if route[-1] != instance.depot:
# #             violations[f"{route_idx_str}_end"] = f"Route does not end at depot {instance.depot}. Ends at {route[-1]}."

# #         # 2. Check capacity and collect visited customers for this route
# #         current_route_demand = 0
# #         customers_in_route: Set[int] = set()

# #         for node_idx in range(len(route)):
# #             node = route[node_idx]
# #             if not (0 <= node < instance.dimension):
# #                 violations[f"{route_idx_str}_node_invalid"] = f"Node {node} is out of bounds."
# #                 continue # Skip to next node in this route if invalid

# #             if node != instance.depot: # It's a customer
# #                 if node in visited_customers: # Check if already visited by a *previous* route
# #                     violations[f"customer_{node}_multiple_visits"] = f"Customer {node} visited in multiple routes (e.g., {route_idx_str} and a previous one)."
# #                 if node in customers_in_route and node_idx != 0 and node_idx != len(route)-1 : # Visited twice within the *same* route (excluding depot at start/end)
# #                      # This check is for simple subtours or re-visits.
# #                      # More complex subtour checks (e.g. 0-1-2-1-3-0) are harder here.
# #                      # This simple check catches 0-1-2-1-0
# #                      pass # This is more of a solution quality issue or sub-tour, not strictly a base CVRP violation if demands are met.
# #                           # True subtour (excluding depot) check is more complex.
                
# #                 current_route_demand += instance.demands[node]
# #                 visited_customers.add(node)
# #                 customers_in_route.add(node)
        
# #         if current_route_demand > instance.capacity:
# #             violations[f"{route_idx_str}_capacity"] = f"Exceeds capacity. Demand: {current_route_demand}, Capacity: {instance.capacity}."

# #     # 3. Check if all customers were visited exactly once
# #     unvisited_customers = all_customers - visited_customers
# #     if unvisited_customers:
# #         violations["unvisited_customers"] = f"Customers not visited: {sorted(list(unvisited_customers))}."

# #     # Customers visited more than once would have been caught by the `customer_{node}_multiple_visits` check.
# #     # The `visited_customers` set naturally handles the "exactly once" part if no `_multiple_visits` error occurred.

# #     is_feasible = not bool(violations)
# #     solution.is_feasible = is_feasible # Update the solution object
# #     solution.feasibility_details = violations.copy()

# #     return is_feasible, violations

# # if __name__ == '__main__':
# #     # Example Usage (requires CVRPInstance to be defined and importable)
# #     # Create dummy instance for testing
# #     from common.cvrp_instance import CVRPInstance # For example usage

# #     # A simple instance
# #     dummy_instance = CVRPInstance(
# #         name="dummy_eval_instance",
# #         dimension=5, # Depot + 4 customers
# #         capacity=100,
# #         distance_matrix=[
# #             [0, 10, 15, 20, 25], # Depot 0
# #             [10, 0, 5, 30, 35],  # Cust 1
# #             [15, 5, 0, 8, 40],   # Cust 2
# #             [20, 30, 8, 0, 12],  # Cust 3
# #             [25, 35, 40, 12, 0]  # Cust 4
# #         ],
# #         demands=[0, 20, 30, 40, 25], # d0, d1, d2, d3, d4
# #         depot=0
# #     )

# #     # --- Test Case 1: Feasible Solution ---
# #     routes1 = [[0, 1, 2, 0], [0, 3, 4, 0]]
# #     cost1 = calculate_solution_cost(routes1, dummy_instance) # (10+5+15) + (20+12+25) = 30 + 57 = 87
# #     sol1 = CVRPSolution(routes=routes1, total_cost=cost1, instance_name=dummy_instance.name)
# #     is_feasible1, violations1 = check_solution_feasibility(sol1, dummy_instance)
    
# #     print(f"--- Solution 1 (Cost: {sol1.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible1}")
# #     print(f"Violations: {violations1}")
# #     assert is_feasible1 is True
# #     assert not violations1
# #     assert sol1.total_cost == 87

# #     # --- Test Case 2: Infeasible - Capacity Exceeded ---
# #     # Route 0: 1 (20) + 3 (40) = 60. Route 1: 2 (30) + 4 (25) = 55. Both < 100.
# #     # Let's make route 0 exceed capacity: 0-1-3-0. Demands: d1=20, d3=40. Total = 60. OK.
# #     # Let's make route 0: 0-1-2-3-0. Demands: d1=20, d2=30, d3=40. Total = 90. OK.
# #     # Let's make route 0: 0-1-2-3-4-0. Demands: d1=20, d2=30, d3=40, d4=25. Total = 115. Exceeds 100.
# #     routes2 = [[0, 1, 2, 3, 4, 0]]
# #     cost2 = calculate_solution_cost(routes2, dummy_instance)
# #     sol2 = CVRPSolution(routes=routes2, total_cost=cost2, instance_name=dummy_instance.name)
# #     is_feasible2, violations2 = check_solution_feasibility(sol2, dummy_instance)

# #     print(f"\n--- Solution 2 (Cost: {sol2.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible2}")
# #     print(f"Violations: {violations2}")
# #     assert is_feasible2 is False
# #     assert "route_0_capacity" in violations2

# #     # --- Test Case 3: Infeasible - Customer Unvisited ---
# #     routes3 = [[0, 1, 0], [0, 2, 0]] # Customers 3 and 4 unvisited
# #     cost3 = calculate_solution_cost(routes3, dummy_instance)
# #     sol3 = CVRPSolution(routes=routes3, total_cost=cost3, instance_name=dummy_instance.name)
# #     is_feasible3, violations3 = check_solution_feasibility(sol3, dummy_instance)
    
# #     print(f"\n--- Solution 3 (Cost: {sol3.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible3}")
# #     print(f"Violations: {violations3}")
# #     assert is_feasible3 is False
# #     assert "unvisited_customers" in violations3
# #     assert "3" in violations3["unvisited_customers"] and "4" in violations3["unvisited_customers"]


# #     # --- Test Case 4: Infeasible - Customer Visited Twice ---
# #     routes4 = [[0, 1, 2, 0], [0, 1, 3, 0]] # Customer 1 visited twice
# #     cost4 = calculate_solution_cost(routes4, dummy_instance)
# #     sol4 = CVRPSolution(routes=routes4, total_cost=cost4, instance_name=dummy_instance.name)
# #     is_feasible4, violations4 = check_solution_feasibility(sol4, dummy_instance)

# #     print(f"\n--- Solution 4 (Cost: {sol4.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible4}")
# #     print(f"Violations: {violations4}")
# #     assert is_feasible4 is False
# #     assert "customer_1_multiple_visits" in violations4

# #     # --- Test Case 5: Infeasible - Route not starting/ending at depot ---
# #     routes5_start = [[1, 2, 0]] # Starts at 1 instead of 0
# #     cost5_start = calculate_solution_cost(routes5_start, dummy_instance)
# #     sol5_start = CVRPSolution(routes=routes5_start, total_cost=cost5_start, instance_name=dummy_instance.name)
# #     is_feasible5_start, violations5_start = check_solution_feasibility(sol5_start, dummy_instance)
# #     print(f"\n--- Solution 5 Start (Cost: {sol5_start.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible5_start}")
# #     print(f"Violations: {violations5_start}")
# #     assert is_feasible5_start is False
# #     assert "route_0_start" in violations5_start
# #     assert "unvisited_customers" in violations5_start # Other customers will be unvisited

# #     routes5_end = [[0, 2, 1]] # Ends at 1 instead of 0
# #     cost5_end = calculate_solution_cost(routes5_end, dummy_instance)
# #     sol5_end = CVRPSolution(routes=routes5_end, total_cost=cost5_end, instance_name=dummy_instance.name)
# #     is_feasible5_end, violations5_end = check_solution_feasibility(sol5_end, dummy_instance)
# #     print(f"\n--- Solution 5 End (Cost: {sol5_end.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible5_end}")
# #     print(f"Violations: {violations5_end}")
# #     assert is_feasible5_end is False
# #     assert "route_0_end" in violations5_end

# #     # --- Test Case 6: Empty solution with no customers ---
# #     dummy_instance_no_cust = CVRPInstance(
# #         name="no_cust_instance", dimension=1, capacity=100,
# #         distance_matrix=[[0]], demands=[0], depot=0
# #     )
# #     sol6 = CVRPSolution(routes=[], total_cost=0, instance_name=dummy_instance_no_cust.name)
# #     is_feasible6, violations6 = check_solution_feasibility(sol6, dummy_instance_no_cust)
# #     print(f"\n--- Solution 6 (No customers, no routes) (Cost: {sol6.total_cost}) ---")
# #     print(f"Is Feasible: {is_feasible6}")
# #     print(f"Violations: {violations6}")
# #     assert is_feasible6 is True
# #     assert not violations6


# # src/common/cvrp_evaluator.py

# import numpy as np
# from .cvrp_instance import CVRPInstance # Assuming CVRPInstance is in the same 'common' package
# from .cvrp_solution import CVRPSolution

# def calculate_solution_cost(routes: list[list[int]], instance: CVRPInstance) -> float:
#     """
#     Calculates the total cost of a list of routes based on the instance's distance matrix.
#     """
#     total_cost = 0.0
    
#     # --- This is the line to fix ---
#     # Original: if not instance.distance_matrix:
#     # Corrected check:
#     if instance.distance_matrix is None: # Or more specific check if it could be an empty array
#         raise ValueError("Distance matrix is not initialized in the CVRPInstance.")
#     if not isinstance(instance.distance_matrix, np.ndarray) or instance.distance_matrix.size == 0:
#         # Handle cases where it might be an empty list that became an array, or other invalid states
#         # Or simply rely on the None check if that's the only expected "invalid" state.
#         # For now, let's assume if it's not None, it should be a usable NumPy array.
#         # A more robust check could be `if not hasattr(instance, 'distance_matrix') or instance.distance_matrix is None:`
#         # but given the error, it *is* an array, so `is None` is the most direct fix for the ambiguous truth value.
#         pass # If it's an array, it will proceed. The error was about the `if not ...` syntax.

#     for route in routes:
#         if not route or len(route) < 2: # Empty or single-node route has no travel cost
#             continue
#         current_route_cost = 0.0
#         for i in range(len(route) - 1):
#             u, v = route[i], route[i+1]
#             # Check if indices are within bounds
#             if not (0 <= u < instance.distance_matrix.shape[0] and \
#                     0 <= v < instance.distance_matrix.shape[0]):
#                 raise ValueError(f"Node index out of bounds for distance matrix. Route: {route}, Indices: ({u},{v}), Matrix shape: {instance.distance_matrix.shape}")
#             current_route_cost += instance.distance_matrix[u, v]
#         total_cost += current_route_cost
#     return total_cost

# def check_solution_feasibility(
#     solution: CVRPSolution, 
#     instance: CVRPInstance,
#     verbose: bool = False
# ) -> tuple[bool, dict]:
#     """
#     Checks the feasibility of a CVRPSolution against capacity and visit constraints.
#     Returns a boolean (is_feasible) and a dictionary of violations.
#     """
#     if not isinstance(solution, CVRPSolution) or not isinstance(instance, CVRPInstance):
#         raise TypeError("Invalid input types for solution or instance.")

#     violations = {
#         "capacity": [], # List of tuples: (route_idx, violated_load, capacity)
#         "unvisited_customers": [], # List of customer_idx
#         "overvisited_customers": [], # List of tuples: (customer_idx, num_visits)
#         "depot_not_in_route_ends": [] # List of route_idx where depot handling is wrong
#     }
#     is_feasible = True

#     # 1. Check capacity constraints for each route
#     for i, route in enumerate(solution.routes):
#         if not route: # Skip empty routes if any
#             continue
            
#         # Basic structural check: does route start and end at depot?
#         if route[0] != instance.depot or route[-1] != instance.depot:
#             violations["depot_not_in_route_ends"].append(i)
#             is_feasible = False
#             if verbose: print(f"Route {i} does not start/end at depot {instance.depot}: {route}")


#         current_load = 0
#         for node_idx in route:
#             if node_idx != instance.depot: # Depot has no demand / demand is 0
#                 if 0 <= node_idx < len(instance.demands):
#                     current_load += instance.demands[node_idx]
#                 else:
#                     # This indicates a serious issue with the route or instance data
#                     if verbose: print(f"Warning: Node index {node_idx} in route {i} is out of bounds for demands array (len {len(instance.demands)}).")
#                     # Depending on strictness, this could be an infeasibility. For now, we'll skip its demand.

#         if current_load > instance.capacity:
#             violations["capacity"].append({"route_idx": i, "load": current_load, "capacity": instance.capacity})
#             is_feasible = False
#             if verbose: print(f"Route {i} violates capacity: load {current_load} > capacity {instance.capacity}")

#     # 2. Check customer visit constraints (each customer visited exactly once by some route)
#     all_customer_nodes = [idx for idx in range(instance.dimension) if idx != instance.depot]
#     visited_counts = {customer: 0 for customer in all_customer_nodes}

#     for route in solution.routes:
#         for node_idx in route:
#             if node_idx != instance.depot and node_idx in visited_counts:
#                 visited_counts[node_idx] += 1
    
#     for customer, count in visited_counts.items():
#         if count == 0:
#             violations["unvisited_customers"].append(customer)
#             is_feasible = False
#             if verbose: print(f"Customer {customer} unvisited.")
#         elif count > 1:
#             violations["overvisited_customers"].append({"customer_idx": customer, "visits": count})
#             is_feasible = False
#             if verbose: print(f"Customer {customer} visited {count} times (overvisited).")
            
#     if violations["depot_not_in_route_ends"]: # If any route has bad depot handling
#         is_feasible = False
        
#     return is_feasible, violations


# src/common/cvrp_evaluator.py

import numpy as np
from .cvrp_instance import CVRPInstance # Assuming CVRPInstance is in the same 'common' package
from .cvrp_solution import CVRPSolution
from typing import List, Dict, Tuple, Optional # Added for type hinting


def calculate_solution_cost(routes: List[List[int]], instance: CVRPInstance) -> float:
    """
    Calculates the total cost of a list of routes based on the instance's distance matrix.
    """
    total_cost = 0.0

    # --- Local Fix for distance_matrix type ---
    # Ideally, instance.distance_matrix should always be a NumPy array from load_cvrp_instance.
    # This conversion is a safeguard if it's loaded as a list of lists.
    if instance.distance_matrix is None:
        raise ValueError("Distance matrix is not initialized in the CVRPInstance.")
    
    if not isinstance(instance.distance_matrix, np.ndarray):
        try:
            # Attempt to convert to NumPy array if it's a list of lists
            distance_matrix_np = np.array(instance.distance_matrix, dtype=float)
            if distance_matrix_np.ndim != 2:
                raise ValueError("Distance matrix, if a list, must be convertible to a 2D array.")
        except Exception as e:
            raise ValueError(f"Could not convert instance.distance_matrix to a usable NumPy array: {e}")
    else:
        distance_matrix_np = instance.distance_matrix # It's already a NumPy array
    # --- End of Local Fix ---
    # BEST FIX: Ensure load_cvrp_instance in src/common/cvrp_instance.py
    #           always creates instance.distance_matrix as np.ndarray.

    if distance_matrix_np.size == 0 and instance.dimension > 1 : # Check for empty matrix if not a single depot instance
        raise ValueError("Distance matrix is empty or invalid for the given dimension.")


    for route in routes:
        if not route or len(route) < 2:
            continue
        current_route_cost = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            
            # Use distance_matrix_np which is guaranteed to be a NumPy array here
            if not (0 <= u < distance_matrix_np.shape[0] and \
                    0 <= v < distance_matrix_np.shape[0]):
                raise ValueError(
                    f"Node index out of bounds for distance matrix. Route: {route}, "
                    f"Indices: ({u},{v}), Matrix shape: {distance_matrix_np.shape}"
                )
            current_route_cost += distance_matrix_np[u, v]
        total_cost += current_route_cost
    return total_cost

def check_solution_feasibility(
    solution: CVRPSolution,
    instance: CVRPInstance,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Checks the feasibility of a CVRPSolution against capacity and visit constraints.
    Returns a boolean (is_feasible) and a dictionary of violations.
    """
    if not isinstance(solution, CVRPSolution) or not isinstance(instance, CVRPInstance):
        raise TypeError("Invalid input types for solution or instance.")

    violations: Dict[str, List[Any]] = { # More specific type hint if possible
        "capacity": [], 
        "unvisited_customers": [], 
        "overvisited_customers": [], 
        "depot_not_in_route_ends": [] 
    }
    is_feasible = True

    # 1. Check capacity constraints for each route
    for i, route in enumerate(solution.routes):
        if not route: 
            continue
            
        if route[0] != instance.depot or route[-1] != instance.depot:
            violations["depot_not_in_route_ends"].append(i)
            is_feasible = False
            if verbose: print(f"Route {i} does not start/end at depot {instance.depot}: {route}")

        current_load = 0
        # Ensure demands is accessible and node indices are valid
        if instance.demands is None:
             raise ValueError("Instance demands are not initialized.")

        for node_idx in route:
            if node_idx != instance.depot: 
                if 0 <= node_idx < len(instance.demands):
                    current_load += instance.demands[node_idx]
                else:
                    if verbose: print(f"Warning: Node index {node_idx} in route {i} is out of bounds for demands array (len {len(instance.demands)}).")

        if instance.capacity is None:
            raise ValueError("Instance capacity is not initialized.")
        if current_load > instance.capacity:
            violations["capacity"].append({"route_idx": i, "load": current_load, "capacity": instance.capacity}) # Changed to dict
            is_feasible = False
            if verbose: print(f"Route {i} violates capacity: load {current_load} > capacity {instance.capacity}")

    # 2. Check customer visit constraints
    if instance.dimension is None or instance.depot is None:
        raise ValueError("Instance dimension or depot is not properly initialized for visit checks.")
        
    all_customer_nodes = [idx for idx in range(instance.dimension) if idx != instance.depot]
    visited_counts = {customer: 0 for customer in all_customer_nodes}

    for route in solution.routes:
        for node_idx in route:
            if node_idx != instance.depot and node_idx in visited_counts:
                visited_counts[node_idx] += 1
    
    for customer, count in visited_counts.items():
        if count == 0:
            violations["unvisited_customers"].append(customer)
            is_feasible = False
            if verbose: print(f"Customer {customer} unvisited.")
        elif count > 1:
            violations["overvisited_customers"].append({"customer_idx": customer, "visits": count}) # Changed to dict
            is_feasible = False
            if verbose: print(f"Customer {customer} visited {count} times (overvisited).")
            
    if violations["depot_not_in_route_ends"]:
        is_feasible = False
        
    return is_feasible, violations