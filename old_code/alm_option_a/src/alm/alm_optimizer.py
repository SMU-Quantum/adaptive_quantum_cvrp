# cvrp_tripartite_solver/src/alm/alm_optimizer.py

import math
import copy 
from typing import List, Dict, Optional, Tuple, Set

from common.cvrp_instance import CVRPInstance
from common.cvrp_solution import CVRPSolution
from common.cvrp_evaluator import check_solution_feasibility, calculate_solution_cost
from alm.subproblem_solvers import solve_espprc_with_dominance 

class AlmOptimizer:
    def __init__(self,
                 instance: CVRPInstance,
                 initial_penalty_rate: float = 1.0,
                 penalty_increase_factor: float = 1.1, 
                 max_penalty_rate: float = 1000.0,
                 initial_lagrange_multipliers: Optional[List[float]] = None,
                 max_alm_iterations: int = 100,
                 convergence_tolerance: float = 1e-3, 
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

        if initial_lagrange_multipliers:
            if len(initial_lagrange_multipliers) != self.num_customers:
                raise ValueError(f"Expected {self.num_customers} initial Lagrange multipliers, "
                                 f"got {len(initial_lagrange_multipliers)}")
            self.lambdas: List[float] = list(initial_lagrange_multipliers)
        else:
            self.lambdas: List[float] = [0.0] * self.num_customers

        self.rhos: List[float] = [initial_penalty_rate] * self.num_customers

        self.penalty_increase_factor = penalty_increase_factor
        self.max_penalty_rate = max_penalty_rate
        self.max_alm_iterations = max_alm_iterations
        self.convergence_tolerance = convergence_tolerance
        
        self.best_feasible_solution: Optional[CVRPSolution] = None
        self.iteration_log: List[Dict] = []

        default_max_vehicles = self.num_customers if self.num_customers > 0 else 1
        if instance.num_vehicles_comment is not None and instance.num_vehicles_comment > 0:
            default_max_vehicles = instance.num_vehicles_comment
        
        self.subproblem_max_vehicles = subproblem_max_vehicles if subproblem_max_vehicles is not None \
                                       else default_max_vehicles
        if self.num_customers == 0: 
            self.subproblem_max_vehicles = 0

    def _get_customer_lambda_rho_idx(self, customer_node_idx: int) -> Optional[int]:
        try:
            return self.customer_indices.index(customer_node_idx)
        except ValueError:
            return None 

    def solve(self) -> Optional[CVRPSolution]:
        print(f"Starting ALM for instance: {self.instance.name}")
        print(f"Parameters: initial_rho={self.rhos[0] if self.rhos else 'N/A'}, "
              f"increase_factor={self.penalty_increase_factor}, max_iter={self.max_alm_iterations}, "
              f"subproblem_max_vehicles_to_generate={self.subproblem_max_vehicles}, verbose={self.verbose}")

        if self.num_customers == 0: 
            print("Instance has no customers. Returning empty feasible solution.")
            empty_solution = CVRPSolution(routes=[], total_cost=0.0, instance_name=self.instance.name, is_feasible=True)
            self.best_feasible_solution = empty_solution
            self.iteration_log.append({
                "iteration": 1, "subproblem_cost": 0.0, "max_abs_g_i": 0.0,
                "is_feasible_overall": True, "best_feasible_cost": 0.0,
                "avg_lambda": 0, "avg_rho": 0, "num_routes_generated":0
            })
            return empty_solution

        for alm_iter in range(self.max_alm_iterations):
            if self.verbose > 0:
                print(f"\n--- ALM Iteration {alm_iter + 1} ---")
            else: 
                if (alm_iter + 1) % 10 == 0 or alm_iter == 0 : 
                     print(f"\n--- ALM Iteration {alm_iter + 1} ---")

            current_iteration_routes: List[List[int]] = []
            customers_covered_this_alm_iter: Set[int] = set()
            
            # *** CORRECTED REWARD INTERPRETATION ***
            # ESPPRC minimizes: sum(original_arc_costs) - sum(rewards_for_nodes_in_path)
            # We want to pass rewards such that unvisited customers (negative lambda) are attractive.
            # So, reward_j = -lambda_j
            base_customer_rewards: Dict[int, float] = {
                cust_idx: -self.lambdas[self._get_customer_lambda_rho_idx(cust_idx)]
                for cust_idx in self.customer_indices
                if self._get_customer_lambda_rho_idx(cust_idx) is not None
            }
            if self.verbose > 1:
                print(f"  Base ESPPRC Rewards (=-lambdas) (first 5 of {len(base_customer_rewards)}): "
                      f"{ {k:round(v,2) for i,(k,v) in enumerate(base_customer_rewards.items()) if i<5} }")

            for vehicle_num_attempt in range(self.subproblem_max_vehicles):
                if self.verbose > 1:
                    print(f"  Attempt {vehicle_num_attempt+1}: Calling ESPPRC. Tabu customers: {customers_covered_this_alm_iter}")
                
                # Pass the base_customer_rewards (which are -lambdas)
                # The ESPPRC will try to maximize these rewards (or minimize cost - reward)
                route = solve_espprc_with_dominance(
                    self.instance,
                    modified_node_rewards=base_customer_rewards, 
                    tabu_customers=customers_covered_this_alm_iter 
                )
                
                if route and len(route) > 2 : 
                    current_iteration_routes.append(route)
                    newly_covered_in_route: Set[int] = set()
                    for node in route:
                        if node != self.instance.depot:
                            newly_covered_in_route.add(node)
                    
                    if not newly_covered_in_route.issubset(customers_covered_this_alm_iter) or not current_iteration_routes[:-1]:
                        customers_covered_this_alm_iter.update(newly_covered_in_route)
                        if self.verbose > 1:
                            print(f"    ESPPRC returned route: {route}, newly covered now: {newly_covered_in_route}, all covered this iter: {customers_covered_this_alm_iter}")
                    else: 
                        current_iteration_routes.pop() 
                        if self.verbose > 1:
                            print(f"    ESPPRC returned redundant route (already covered): {route}. Discarding.")
                        if len(customers_covered_this_alm_iter) == self.num_customers:
                            if self.verbose > 1: print("    All customers covered by current set of routes in this ALM iteration.")
                            break 
                else: 
                    if self.verbose > 1:
                        print(f"    ESPPRC returned no further useful route (attempt {vehicle_num_attempt+1}).")
                    break 
            
            current_iteration_cost = calculate_solution_cost(current_iteration_routes, self.instance)
            current_alm_solution = CVRPSolution(
                routes=current_iteration_routes,
                total_cost=current_iteration_cost,
                instance_name=self.instance.name
            )
            if self.verbose == 0 and (alm_iter + 1) % 10 != 0 and alm_iter != 0 : 
                pass
            else:
                print(f"Subproblem generated {len(current_iteration_routes)} routes with original cost {current_iteration_cost:.2f}")

            is_overall_feasible, overall_violations_dict = check_solution_feasibility(current_alm_solution, self.instance)
            current_alm_solution.is_feasible = is_overall_feasible 
            current_alm_solution.feasibility_details = overall_violations_dict

            customer_visit_counts = {cust_idx: 0 for cust_idx in self.customer_indices}
            for route_item in current_iteration_routes: 
                for node in route_item:
                    if node != self.instance.depot:
                        if node in self.customer_indices: 
                             customer_visit_counts[node] = customer_visit_counts.get(node, 0) + 1
            
            g_i_values: List[float] = [0.0] * self.num_customers
            max_abs_violation = 0.0
            sum_sq_violations = 0.0

            for i, cust_idx in enumerate(self.customer_indices):
                visits = customer_visit_counts.get(cust_idx, 0)
                g_i_values[i] = float(visits - 1)
                max_abs_violation = max(max_abs_violation, abs(g_i_values[i]))
                sum_sq_violations += g_i_values[i]**2
            
            avg_abs_violation = math.sqrt(sum_sq_violations / self.num_customers) if self.num_customers > 0 else 0.0
            
            if self.verbose == 0 and (alm_iter + 1) % 10 != 0 and alm_iter != 0:
                pass
            else:
                print(f"Max customer visit violation |g_i|: {max_abs_violation:.2f}, Avg abs violation: {avg_abs_violation:.2f}")

            if is_overall_feasible:
                if self.verbose > 0 or not self.best_feasible_solution or current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    print("Current ALM solution is FEASIBLE overall.")
                if self.best_feasible_solution is None or \
                   current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    self.best_feasible_solution = current_alm_solution
                    print(f"NEW BEST FEASIBLE solution found with cost: {self.best_feasible_solution.total_cost:.2f}")
            elif self.verbose > 0: 
                 print(f"Current ALM solution is INFEASIBLE. Violations (sample): {dict(list(overall_violations_dict.items())[:3])}")

            for i in range(self.num_customers):
                self.lambdas[i] = self.lambdas[i] + self.rhos[i] * g_i_values[i]

            if max_abs_violation > self.convergence_tolerance: 
                for i in range(self.num_customers):
                    if abs(g_i_values[i]) > self.convergence_tolerance * 0.1: 
                        self.rhos[i] = min(self.rhos[i] * self.penalty_increase_factor, self.max_penalty_rate)
            
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

            log_entry = {
                "iteration": alm_iter + 1,
                "subproblem_original_cost": current_iteration_cost,
                "num_routes_generated": len(current_iteration_routes),
                "max_abs_g_i": max_abs_violation,
                "avg_abs_g_i": avg_abs_violation,
                "is_feasible_overall": is_overall_feasible,
                "best_feasible_cost": self.best_feasible_solution.total_cost if self.best_feasible_solution else float('inf'),
                "avg_lambda": sum(self.lambdas)/len(self.lambdas) if self.lambdas else 0,
                "avg_rho": sum(self.rhos)/len(self.rhos) if self.rhos else 0,
            }
            self.iteration_log.append(log_entry)

            if max_abs_violation <= self.convergence_tolerance and is_overall_feasible: 
                print(f"ALM converged at iteration {alm_iter + 1} with feasible solution and small violations.")
                break
        
        if alm_iter == self.max_alm_iterations - 1:
            print("ALM reached max iterations.")

        if self.best_feasible_solution:
            print(f"\nBest feasible solution found with cost: {self.best_feasible_solution.total_cost:.2f}")
            final_check_feasible, final_violations = check_solution_feasibility(self.best_feasible_solution, self.instance)
            self.best_feasible_solution.is_feasible = final_check_feasible
            self.best_feasible_solution.feasibility_details = final_violations
        else:
            print("\nNo feasible solution found by ALM.")
        return self.best_feasible_solution

if __name__ == '__main__':
    print("Running ALM Optimizer basic example...")
    from common.cvrp_instance import CVRPInstance
    
    dummy_instance_alm = CVRPInstance(
        name="alm_test_instance_main",
        dimension=3, 
        capacity=50,
        distance_matrix=[ [0, 10, 12], [10, 0, 5], [12, 5, 0] ],
        demands=[0, 20, 30], 
        depot=0,
        num_vehicles_comment=2 
    )
    optimizer = AlmOptimizer(
        instance=dummy_instance_alm,
        initial_penalty_rate=0.5,
        penalty_increase_factor=1.05, 
        max_alm_iterations=20, 
        convergence_tolerance=0.01,
        subproblem_max_vehicles=2,
        verbose=2 
    )
    best_solution = optimizer.solve()
    if best_solution:
        print("\n--- Final Best Solution from ALM (main example) ---")
        print(best_solution)
        print(f"Details of feasibility: {best_solution.feasibility_details}")
    else:
        print("\n--- No feasible solution found by ALM in this example run (main example) ---")
    
    if optimizer.iteration_log:
        print("\nALM Iteration Log (main example - last 5 entries):")
        for log_entry in optimizer.iteration_log[-5:]:
            print(log_entry)

    instance_no_cust = CVRPInstance(name="no_cust_main", dimension=1, capacity=100, distance_matrix=[[0]], demands=[0], depot=0)
    opt_no_cust = AlmOptimizer(instance=instance_no_cust, verbose=1)
    sol_no_cust = opt_no_cust.solve()
    print(f"\nSolution for no_cust_main: {sol_no_cust}")
