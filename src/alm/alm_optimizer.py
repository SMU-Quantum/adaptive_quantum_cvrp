# cvrp_tripartite_solver/src/alm/alm_optimizer.py

import math
import copy 
from typing import List, Dict, Optional, Tuple, Set

from common.cvrp_instance import CVRPInstance
from common.cvrp_solution import CVRPSolution
from common.cvrp_evaluator import check_solution_feasibility, calculate_solution_cost
# Ensure this imports the ESP solver (capacity-agnostic)
from alm.subproblem_solvers import solve_esp_with_dominance 

class AlmOptimizer:
    """
    Implements the Augmented Lagrangian Method for CVRP,
    relaxing both customer visit and vehicle capacity constraints.
    """
    def __init__(self,
                 instance: CVRPInstance,
                 # Parameters for customer visit constraints
                 initial_penalty_rate: float = 1.0, # rho
                 penalty_increase_factor: float = 1.1, 
                 max_penalty_rate: float = 1000.0,
                 initial_lagrange_multipliers: Optional[List[float]] = None, # lambdas
                 convergence_tolerance: float = 1e-3, 
                 # Parameters for capacity constraints
                 initial_capacity_penalty_rate: float = 1.0, # sigma
                 capacity_penalty_increase_factor: float = 1.1,
                 max_capacity_penalty_rate: float = 1000.0,
                 initial_capacity_multipliers: Optional[List[float]] = None, # mus
                 capacity_convergence_tolerance: float = 1.0, # Tolerance for capacity violation (e.g., in units of demand)
                 # General ALM parameters
                 max_alm_iterations: int = 100,
                 subproblem_max_vehicles: Optional[int] = None,
                 verbose: int = 0):
        
        self.instance = instance
        self.num_customers = 0
        self.customer_indices: List[int] = []
        self.verbose = verbose

        if instance.dimension > 0 : 
            self.customer_indices = sorted([i for i in range(instance.dimension) if i != instance.depot])
            self.num_customers = len(self.customer_indices)
        
        if instance.dimension > 1 and self.num_customers == 0 and instance.depot is not None:
             if self.verbose > 0:
                print(f"Warning: Instance '{instance.name}' has dimension {instance.dimension} "
                      f"but no identified customer nodes (non-depot nodes).")
        elif instance.dimension == 1 and instance.depot == 0 :
             pass 
        elif self.num_customers <= 0 and instance.dimension > 1: 
             raise ValueError("Instance must have at least one customer (dimension > 1 and non-depot nodes exist).")

        # --- Customer Visit Multipliers & Penalties ---
        if initial_lagrange_multipliers:
            if len(initial_lagrange_multipliers) != self.num_customers:
                raise ValueError(f"Expected {self.num_customers} initial customer visit Lagrange multipliers, "
                                 f"got {len(initial_lagrange_multipliers)}")
            self.lambdas: List[float] = list(initial_lagrange_multipliers)
        else:
            self.lambdas: List[float] = [0.0] * self.num_customers
        self.rhos: List[float] = [initial_penalty_rate] * self.num_customers

        # --- ALM Parameters for Customer Visits ---
        self.penalty_increase_factor = penalty_increase_factor
        self.max_penalty_rate = max_penalty_rate
        self.convergence_tolerance = convergence_tolerance
        
        # --- Subproblem Vehicle Count ---
        # This determines how many routes we attempt to generate and thus how many capacity multipliers we need.
        default_max_vehicles = self.num_customers if self.num_customers > 0 else 1
        if instance.num_vehicles_comment is not None and instance.num_vehicles_comment > 0:
            default_max_vehicles = instance.num_vehicles_comment
        self.subproblem_max_vehicles = subproblem_max_vehicles if subproblem_max_vehicles is not None \
                                       else default_max_vehicles
        if self.num_customers == 0: 
            self.subproblem_max_vehicles = 0

        # --- Capacity Multipliers & Penalties (one per potential vehicle/route) ---
        if initial_capacity_multipliers:
            if len(initial_capacity_multipliers) != self.subproblem_max_vehicles:
                raise ValueError(f"Expected {self.subproblem_max_vehicles} initial capacity multipliers, "
                                 f"got {len(initial_capacity_multipliers)}")
            self.mus_capacity: List[float] = list(initial_capacity_multipliers)
        else:
            self.mus_capacity: List[float] = [0.0] * self.subproblem_max_vehicles
        self.sigmas_capacity: List[float] = [initial_capacity_penalty_rate] * self.subproblem_max_vehicles
        
        # --- ALM Parameters for Capacity ---
        self.capacity_penalty_increase_factor = capacity_penalty_increase_factor
        self.max_capacity_penalty_rate = max_capacity_penalty_rate
        self.capacity_convergence_tolerance = capacity_convergence_tolerance

        # --- General ---
        self.max_alm_iterations = max_alm_iterations
        self.best_feasible_solution: Optional[CVRPSolution] = None
        self.iteration_log: List[Dict] = []


    def _get_customer_lambda_rho_idx(self, customer_node_idx: int) -> Optional[int]:
        try:
            return self.customer_indices.index(customer_node_idx)
        except ValueError:
            return None 

    def solve(self) -> Optional[CVRPSolution]:
        print(f"Starting ALM (Capacity Relaxed) for instance: {self.instance.name}")
        print(f"Parameters: initial_rho_visit={self.rhos[0] if self.rhos else 'N/A'}, "
              f"initial_sigma_cap={self.sigmas_capacity[0] if self.sigmas_capacity else 'N/A'}, "
              f"max_iter={self.max_alm_iterations}, "
              f"subproblem_max_vehicles_to_generate={self.subproblem_max_vehicles}, verbose={self.verbose}")

        if self.num_customers == 0: 
            print("Instance has no customers. Returning empty feasible solution.")
            # ... (return empty solution logic as before) ...
            empty_solution = CVRPSolution(routes=[], total_cost=0.0, instance_name=self.instance.name, is_feasible=True)
            self.best_feasible_solution = empty_solution
            self.iteration_log.append({
                "iteration": 1, "subproblem_cost": 0.0, "max_abs_g_i": 0.0, "max_cap_violation": 0.0,
                "is_feasible_overall": True, "best_feasible_cost": 0.0,
                "avg_lambda": 0, "avg_rho": 0, "avg_mu_cap":0, "avg_sigma_cap":0, "num_routes_generated":0
            })
            return empty_solution


        for alm_iter in range(self.max_alm_iterations):
            # --- Iteration Start Logging ---
            if self.verbose > 0:
                print(f"\n--- ALM Iteration {alm_iter + 1} ---")
            else: 
                if (alm_iter + 1) % 10 == 0 or alm_iter == 0 : 
                     print(f"\n--- ALM Iteration {alm_iter + 1} ---")

            # --- 1. Solve Subproblem (Generate Routes with ESP) ---
            current_iteration_routes: List[List[int]] = []
            customers_covered_this_alm_iter: Set[int] = set()
            route_loads_this_iter: List[int] = [] # To store loads of generated routes
            
            base_customer_rewards: Dict[int, float] = {
                cust_idx: -self.lambdas[self._get_customer_lambda_rho_idx(cust_idx)]
                for cust_idx in self.customer_indices
                if self._get_customer_lambda_rho_idx(cust_idx) is not None
            }
            if self.verbose > 1:
                print(f"  Base ESP Rewards (=-lambdas_visit) (first 5 of {len(base_customer_rewards)}): "
                      f"{ {k:round(v,2) for i,(k,v) in enumerate(base_customer_rewards.items()) if i<5} }")

            for vehicle_idx in range(self.subproblem_max_vehicles): # vehicle_idx is 0 to K-1
                if self.verbose > 1:
                    print(f"  Attempt {vehicle_idx+1}: Calling ESP. Tabu customers: {customers_covered_this_alm_iter}")
                
                # The ESP solver is capacity-agnostic. Rewards are only for customer visits.
                # Capacity multipliers (mus) will affect the ALM objective, not directly arc costs here.
                route = solve_esp_with_dominance(
                    self.instance,
                    modified_node_rewards=base_customer_rewards,
                    # pass a copy so that later mutations of the ALM set
                    # don’t retroactively change the ‘recorded’ call argument
                    tabu_customers=customers_covered_this_alm_iter.copy(),
                    debug_esp=(self.verbose > 2),
                    capacity_multiplier=self.mus_capacity[vehicle_idx]
                )

                
                if route and len(route) > 2 : 
                    current_iteration_routes.append(route)
                    current_route_load = sum(self.instance.demands[node] for node in route if node != self.instance.depot)
                    route_loads_this_iter.append(current_route_load)

                    newly_covered_in_route: Set[int] = set()
                    for node in route:
                        if node != self.instance.depot:
                            newly_covered_in_route.add(node)
                    
                    if not newly_covered_in_route.issubset(customers_covered_this_alm_iter) or not current_iteration_routes[:-1]:
                        customers_covered_this_alm_iter.update(newly_covered_in_route)
                        if self.verbose > 1:
                            print(f"    ESP returned route: {route} (load: {current_route_load}), newly_covered: {newly_covered_in_route}")
                    else: 
                        current_iteration_routes.pop() 
                        route_loads_this_iter.pop()
                        if self.verbose > 1:
                            print(f"    ESP returned redundant route (already covered): {route}. Discarding.")
                        if len(customers_covered_this_alm_iter) == self.num_customers:
                            if self.verbose > 1: print("    All customers covered by current set of routes in this ALM iteration.")
                            break 
                else: 
                    if self.verbose > 1:
                        print(f"    ESP returned no further useful route (attempt {vehicle_idx+1}).")
                    break 
            
            current_iteration_cost = calculate_solution_cost(current_iteration_routes, self.instance)
            current_alm_solution = CVRPSolution(
                routes=current_iteration_routes,
                total_cost=current_iteration_cost,
                instance_name=self.instance.name
            )
            # --- Subproblem Logging ---
            if self.verbose == 0 and (alm_iter + 1) % 10 != 0 and alm_iter != 0 : pass
            else: print(f"Subproblem generated {len(current_iteration_routes)} routes with original cost {current_iteration_cost:.2f}")

            # --- 2. Check Overall Feasibility & Calculate Violations ---
            is_overall_feasible, overall_violations_dict = check_solution_feasibility(current_alm_solution, self.instance)
            current_alm_solution.is_feasible = is_overall_feasible 
            current_alm_solution.feasibility_details = overall_violations_dict

            # --- 2a. Customer Visit Violations (g_i) ---
            customer_visit_counts = {cust_idx: 0 for cust_idx in self.customer_indices}
            for route_item in current_iteration_routes: 
                for node in route_item:
                    if node != self.instance.depot and node in self.customer_indices: 
                        customer_visit_counts[node] = customer_visit_counts.get(node, 0) + 1
            
            g_i_values: List[float] = [0.0] * self.num_customers
            max_abs_g_i_violation = 0.0
            sum_sq_g_i_violations = 0.0
            for i, cust_idx in enumerate(self.customer_indices):
                visits = customer_visit_counts.get(cust_idx, 0)
                g_i_values[i] = float(visits - 1)
                max_abs_g_i_violation = max(max_abs_g_i_violation, abs(g_i_values[i]))
                sum_sq_g_i_violations += g_i_values[i]**2
            avg_abs_g_i_violation = math.sqrt(sum_sq_g_i_violations / self.num_customers) if self.num_customers > 0 else 0.0
            
            # --- 2b. Capacity Violations (h_k) ---
            capacity_violations: List[float] = [0.0] * self.subproblem_max_vehicles # For routes actually generated
            max_cap_violation = 0.0
            sum_sq_cap_violations = 0.0
            for k in range(len(current_iteration_routes)): # Only for routes that were formed
                load_k = route_loads_this_iter[k]
                violation_k = max(0, load_k - self.instance.capacity)
                capacity_violations[k] = violation_k # Store violation for this specific generated route
                max_cap_violation = max(max_cap_violation, violation_k)
                sum_sq_cap_violations += violation_k**2
            avg_cap_violation = math.sqrt(sum_sq_cap_violations / len(current_iteration_routes)) if current_iteration_routes else 0.0

            # --- Violation Logging ---
            if self.verbose == 0 and (alm_iter + 1) % 10 != 0 and alm_iter != 0: pass
            else:
                print(f"Max customer visit violation |g_i|: {max_abs_g_i_violation:.2f}, Avg: {avg_abs_g_i_violation:.2f}")
                print(f"Max capacity violation per route: {max_cap_violation:.2f}, Avg: {avg_cap_violation:.2f}")


            if is_overall_feasible:
                # ... (Best solution update logic as before) ...
                if self.verbose > 0 or not self.best_feasible_solution or current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    print("Current ALM solution is FEASIBLE overall.")
                if self.best_feasible_solution is None or \
                   current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    self.best_feasible_solution = current_alm_solution
                    print(f"NEW BEST FEASIBLE solution found with cost: {self.best_feasible_solution.total_cost:.2f}")
            elif self.verbose > 0: 
                 print(f"Current ALM solution is INFEASIBLE. Violations (sample): {dict(list(overall_violations_dict.items())[:3])}")

            # --- 3. Update Lagrange Multipliers (lambdas for visits, mus for capacity) ---
            # Customer visit lambdas
            for i in range(self.num_customers):
                self.lambdas[i] = self.lambdas[i] + self.rhos[i] * g_i_values[i]
            # Capacity mus (for each generated route up to subproblem_max_vehicles)
            for k in range(len(current_iteration_routes)): # Only update for routes that exist
                if k < len(self.mus_capacity): # Safety check
                     self.mus_capacity[k] = max(0, self.mus_capacity[k] + self.sigmas_capacity[k] * capacity_violations[k])
            # For routes not generated in this iteration, their mus could be reset or decayed (advanced)
            # For now, if fewer routes than subproblem_max_vehicles are generated, later mus are untouched.

            # --- 4. Update Penalty Parameters (rhos for visits, sigmas for capacity) ---
            # Customer visit rhos
            if max_abs_g_i_violation > self.convergence_tolerance: 
                for i in range(self.num_customers):
                    if abs(g_i_values[i]) > self.convergence_tolerance * 0.1: 
                        self.rhos[i] = min(self.rhos[i] * self.penalty_increase_factor, self.max_penalty_rate)
            # Capacity sigmas
            if max_cap_violation > self.capacity_convergence_tolerance:
                for k in range(len(current_iteration_routes)): # Only for routes that exist
                    if k < len(self.sigmas_capacity): # Safety check
                        if capacity_violations[k] > self.capacity_convergence_tolerance * 0.1:
                            self.sigmas_capacity[k] = min(self.sigmas_capacity[k] * self.capacity_penalty_increase_factor, 
                                                          self.max_capacity_penalty_rate)
            
            # --- Parameter Logging (verbose) ---
            if self.verbose > 1:
                cust_details_log = []
                for i, cust_idx in enumerate(self.customer_indices[:min(5, self.num_customers)]): 
                    lambda_idx = self._get_customer_lambda_rho_idx(cust_idx)
                    if lambda_idx is not None:
                        cust_details_log.append(
                            f"  Cust {cust_idx}: visits={customer_visit_counts.get(cust_idx,0)}, "
                            f"g_i={g_i_values[lambda_idx]:.2f}, "
                            f"lambda={self.lambdas[lambda_idx]:.2f} (reward={-self.lambdas[lambda_idx]:.2f}), "
                            f"rho={self.rhos[lambda_idx]:.2f}"
                        )
                print("\n".join(cust_details_log))
                if self.mus_capacity:
                    print(f"  Capacity Mus (first 5): {[round(m,2) for m in self.mus_capacity[:5]]}")
                    print(f"  Capacity Sigmas (first 5): {[round(s,2) for s in self.sigmas_capacity[:5]]}")


            # --- Iteration Log Entry ---
            log_entry = {
                "iteration": alm_iter + 1,
                "subproblem_original_cost": current_iteration_cost,
                "num_routes_generated": len(current_iteration_routes),
                "max_abs_g_i": max_abs_g_i_violation,
                "avg_abs_g_i": avg_abs_g_i_violation,
                "max_cap_violation": max_cap_violation,
                "avg_cap_violation": avg_cap_violation,
                "is_feasible_overall": is_overall_feasible,
                "best_feasible_cost": self.best_feasible_solution.total_cost if self.best_feasible_solution else float('inf'),
                "avg_lambda_visit": sum(self.lambdas)/len(self.lambdas) if self.lambdas else 0,
                "avg_rho_visit": sum(self.rhos)/len(self.rhos) if self.rhos else 0,
                "avg_mu_capacity": sum(self.mus_capacity)/len(self.mus_capacity) if self.mus_capacity else 0,
                "avg_sigma_capacity": sum(self.sigmas_capacity)/len(self.sigmas_capacity) if self.sigmas_capacity else 0,
            }
            self.iteration_log.append(log_entry)

            # --- 5. Check Convergence ---
            customer_visit_converged = max_abs_g_i_violation <= self.convergence_tolerance
            capacity_converged = max_cap_violation <= self.capacity_convergence_tolerance
            
            if customer_visit_converged and capacity_converged and is_overall_feasible: 
                print(f"ALM converged at iteration {alm_iter + 1} with feasible solution and all violations within tolerance.")
                break
        
        # --- Loop End ---
        if alm_iter == self.max_alm_iterations - 1:
            print("ALM reached max iterations.")

        if self.best_feasible_solution:
            print(f"\nBest feasible solution found with cost: {self.best_feasible_solution.total_cost:.2f}")
            # Final check is good practice
            final_check_feasible, final_violations = check_solution_feasibility(self.best_feasible_solution, self.instance)
            self.best_feasible_solution.is_feasible = final_check_feasible
            self.best_feasible_solution.feasibility_details = final_violations
        else:
            print("\nNo feasible solution found by ALM.")
        return self.best_feasible_solution

if __name__ == '__main__':
    print("Running ALM Optimizer (Capacity Relaxed) basic example...")
    from common.cvrp_instance import CVRPInstance
    
    # Instance where capacity might be an issue for simple routes
    dummy_instance_alm_cap = CVRPInstance(
        name="alm_cap_relaxed_main",
        dimension=4, # Depot 0, Cust 1, 2, 3
        capacity=40, 
        distance_matrix=[ [0,10,15,20], [10,0,5,100], [15,5,0,8], [20,100,8,0] ],
        demands=[0, 20, 25, 15], # d0, d1=20, d2=25, d3=15
        depot=0,
        num_vehicles_comment=3 
    )
    optimizer = AlmOptimizer(
        instance=dummy_instance_alm_cap,
        initial_penalty_rate=1.0, 
        penalty_increase_factor=1.1,
        initial_capacity_penalty_rate=1.0,
        capacity_penalty_increase_factor=1.2,
        max_alm_iterations=50, 
        convergence_tolerance=0.01,
        capacity_convergence_tolerance=0.1,
        subproblem_max_vehicles=3, # Try with 3 vehicles
        verbose=1 
    )
    best_solution = optimizer.solve()
    if best_solution:
        print("\n--- Final Best Solution from ALM (Capacity Relaxed main example) ---")
        print(best_solution)
        if best_solution.feasibility_details:
            print(f"Details of feasibility: {best_solution.feasibility_details}")
    else:
        print("\n--- No feasible solution found by ALM in this example run (Capacity Relaxed main example) ---")
    
    if optimizer.iteration_log:
        print("\nALM Iteration Log (Capacity Relaxed main example - last 10 entries):")
        for log_entry in optimizer.iteration_log[-10:]:
            print(log_entry)
