# src/alm/alm_optimizer_quantum.py
import numpy as np
import math
import copy
import time # For basic timing
from typing import List, Dict, Optional, Tuple, Set

from src.common.cvrp_instance import CVRPInstance
from src.common.cvrp_solution import CVRPSolution
from src.common.cvrp_evaluator import check_solution_feasibility, calculate_solution_cost

# Import the quantum TSP subproblem solver
from src.quantum_alm.qubo_vqe_route_solver import solve_tsp_subproblem_quantum

class AlmOptimizerQuantum:
    """
    Implements the Augmented Lagrangian Method for CVRP,
    where route generation subproblems are solved using a quantum
    solver (QUBO-VQE for TSP subproblems).
    Relaxes both customer visit and vehicle capacity constraints at the ALM level.
    """
    def __init__(self,
                 instance: CVRPInstance,
                 # ALM parameters for customer visit constraints
                 initial_penalty_rate: float = 1.0, # rho for customer visits
                 penalty_increase_factor: float = 1.1,
                 max_penalty_rate: float = 1000.0,
                 initial_lagrange_multipliers: Optional[List[float]] = None, # lambdas for customer visits
                 convergence_tolerance: float = 1e-3,
                 # ALM parameters for capacity constraints
                 initial_capacity_penalty_rate: float = 1.0, # sigma for capacities
                 capacity_penalty_increase_factor: float = 1.1,
                 max_capacity_penalty_rate: float = 1000.0,
                 initial_capacity_multipliers: Optional[List[float]] = None, # mus for capacities
                 capacity_convergence_tolerance: float = 1.0,
                 # General ALM parameters
                 max_alm_iterations: int = 100,
                 subproblem_max_vehicles: Optional[int] = None, # Max routes to attempt per ALM iter
                 verbose: int = 0,
                 # Quantum Solver Configuration
                 quantum_solver_config: Optional[Dict] = None):

        self.instance = instance
        self.num_customers = 0
        self.customer_indices: List[int] = [] # Node indices of customers
        self.verbose = verbose

        if instance.dimension > 0:
            self.customer_indices = sorted([i for i in range(instance.dimension) if i != instance.depot])
            self.num_customers = len(self.customer_indices)

        if self.num_customers <= 0 and instance.dimension > 1:
            if instance.depot is not None: # Allow instances with only a depot (dim=1)
                 raise ValueError("Instance must have at least one customer (non-depot node).")
            # If dim=1 and depot=0, it's a valid empty problem.

        # --- Customer Visit Multipliers & Penalties (ALM level) ---
        self._customer_to_internal_idx_map: Dict[int, int] = {
            node_idx: i for i, node_idx in enumerate(self.customer_indices)
        }
        if initial_lagrange_multipliers:
            if len(initial_lagrange_multipliers) != self.num_customers:
                raise ValueError(f"Expected {self.num_customers} initial customer visit Lagrange multipliers, "
                                 f"got {len(initial_lagrange_multipliers)}")
            self.lambdas_visit: List[float] = list(initial_lagrange_multipliers)
        else:
            self.lambdas_visit: List[float] = [0.0] * self.num_customers
        self.rhos_visit: List[float] = [initial_penalty_rate] * self.num_customers

        self.penalty_increase_factor = penalty_increase_factor
        self.max_penalty_rate = max_penalty_rate
        self.convergence_tolerance = convergence_tolerance

        default_max_vehicles = self.num_customers if self.num_customers > 0 else 1
        if instance.num_vehicles_comment is not None and instance.num_vehicles_comment > 0:
            default_max_vehicles = max(1, instance.num_vehicles_comment)

        self.subproblem_max_vehicles_to_generate = subproblem_max_vehicles if subproblem_max_vehicles is not None \
                                       else default_max_vehicles
        if self.num_customers == 0 : self.subproblem_max_vehicles_to_generate = 0

        num_capacity_multipliers = self.subproblem_max_vehicles_to_generate
        if initial_capacity_multipliers:
            if len(initial_capacity_multipliers) != num_capacity_multipliers:
                raise ValueError(f"Expected {num_capacity_multipliers} initial capacity multipliers, "
                                 f"got {len(initial_capacity_multipliers)}")
            self.mus_capacity: List[float] = list(initial_capacity_multipliers)
        else:
            self.mus_capacity: List[float] = [0.0] * num_capacity_multipliers
        self.sigmas_capacity: List[float] = [initial_capacity_penalty_rate] * num_capacity_multipliers

        self.capacity_penalty_increase_factor = capacity_penalty_increase_factor
        self.max_capacity_penalty_rate = max_capacity_penalty_rate
        self.capacity_convergence_tolerance = capacity_convergence_tolerance

        self.max_alm_iterations = max_alm_iterations
        self.best_feasible_solution: Optional[CVRPSolution] = None
        self.iteration_log: List[Dict] = []

        self.quantum_solver_config = quantum_solver_config if quantum_solver_config else {}
        self.quantum_solver_config.setdefault('edge_cost_factor', 1.0)
        self.quantum_solver_config.setdefault('reward_factor', 1.0)
        self.quantum_solver_config.setdefault('constraint_penalty_factor', 500.0)
        self.quantum_solver_config.setdefault('vqe_reps', 2)
        self.quantum_solver_config.setdefault('vqe_max_iter', 100)
        self.quantum_solver_config.setdefault('vqe_optimizer_method', "Powell") # Changed from Powell based on user's prior code
        self.quantum_solver_config.setdefault('plot_folder_prefix', "output_plots_alm_qsub_")
        self.quantum_solver_config.setdefault('max_customers_in_quantum_subproblem', 2)

        if self.verbose > 0:
            print(f"AlmOptimizerQuantum initialized. Quantum solver config: {self.quantum_solver_config}")

    def _get_internal_customer_idx(self, customer_node_idx: int) -> Optional[int]:
        return self._customer_to_internal_idx_map.get(customer_node_idx)

    def _select_customer_subset_for_quantum_route(
            self,
            remaining_unvisited_customers: Set[int],
            current_lambdas: Dict[int, float]
        ) -> List[int]:
        if not remaining_unvisited_customers:
            return []
        max_c_in_sub = self.quantum_solver_config['max_customers_in_quantum_subproblem']
        sorted_customers = sorted(
            list(remaining_unvisited_customers),
            key=lambda cid: current_lambdas.get(cid, 0),
            reverse=False
        )
        selected_customers = sorted_customers[:max_c_in_sub]
        if not selected_customers:
            return []
        subproblem_nodes = [self.instance.depot] + selected_customers
        if self.verbose > 1:
            print(f"    Selected customers for quantum subproblem: {selected_customers} "
                  f"(out of {len(remaining_unvisited_customers)} unvisited). "
                  f"Total nodes for quantum TSP: {subproblem_nodes}")
        return subproblem_nodes

    def solve(self) -> Optional[CVRPSolution]:
        if self.verbose > 0:
            print(f"Starting Quantum ALM for instance: {self.instance.name}")
            # ... (other initial prints) ...

        if self.num_customers == 0 and self.instance.dimension == 1 and self.instance.depot == 0:
            # ... (handle empty problem) ...
            empty_solution = CVRPSolution(routes=[], total_cost=0.0, instance_name=self.instance.name, is_feasible=True)
            self.best_feasible_solution = empty_solution
            self.iteration_log.append({"iteration": 1, "subproblem_cost": 0.0, "max_abs_g_i": 0.0, "max_cap_violation": 0.0,
                                       "is_feasible_overall": True, "best_feasible_cost": 0.0,
                                       "avg_lambda_visit": 0, "avg_rho_visit": 0, "avg_mu_capacity":0, "avg_sigma_capacity":0,
                                       "num_routes_generated":0, "time_subproblem_solve":0, "time_alm_iter":0})
            return empty_solution

        alm_start_time = time.time()

        for alm_iter in range(self.max_alm_iterations):
            iter_start_time = time.time()
            if self.verbose > 0:
                print(f"\n--- ALM Iteration {alm_iter + 1} ---")

            current_iteration_routes: List[List[int]] = []
            current_iteration_route_costs: List[float] = []
            customers_covered_this_alm_iter: Set[int] = set()
            route_loads_this_iter: List[int] = []
            
            alm_node_rewards: Dict[int, float] = {}
            for i, cust_node_idx in enumerate(self.customer_indices):
                alm_node_rewards[cust_node_idx] = -self.lambdas_visit[i] 
            alm_node_rewards[self.instance.depot] = 0 

            subproblem_solve_total_time = 0

            for route_idx in range(self.subproblem_max_vehicles_to_generate):
                if self.verbose > 1:
                    print(f"  Attempting to generate quantum route {route_idx + 1}/{self.subproblem_max_vehicles_to_generate}")

                remaining_unvisited_for_selection = set(self.customer_indices) - customers_covered_this_alm_iter
                if not remaining_unvisited_for_selection:
                    if self.verbose > 1: print("    All customers covered in this ALM iteration by previous quantum routes.")
                    break

                current_lambdas_map = {
                    node_idx: self.lambdas_visit[self._get_internal_customer_idx(node_idx)]
                    for node_idx in remaining_unvisited_for_selection
                    if self._get_internal_customer_idx(node_idx) is not None
                }
                
                subproblem_nodes = self._select_customer_subset_for_quantum_route(
                    remaining_unvisited_for_selection,
                    current_lambdas_map
                )

                if not subproblem_nodes or len(subproblem_nodes) <= 1:
                    if self.verbose > 1: print("    No suitable customer subset for quantum route, or all covered.")
                    break
                
                subproblem_nodes_for_quantum = [self.instance.depot] + [n for n in subproblem_nodes if n != self.instance.depot]
                subproblem_nodes_for_quantum = sorted(set(subproblem_nodes_for_quantum), key=subproblem_nodes_for_quantum.index)

                if len(subproblem_nodes_for_quantum) < 2 :
                    if self.verbose > 1: print(f"    Quantum subproblem {subproblem_nodes_for_quantum} too small, skipping.")
                    continue

                q_sub_start_time = time.time()
                current_q_config = self.quantum_solver_config.copy()
                current_q_config['plot_folder_prefix'] = (
                    f"{self.quantum_solver_config.get('plot_folder_prefix', 'q_alm_')}"
                    f"iter{alm_iter+1}_r{route_idx+1}_"
                )

                quantum_route, vqe_obj_val = solve_tsp_subproblem_quantum(
                    self.instance.distance_matrix,
                    subproblem_nodes_for_quantum,
                    alm_node_rewards,
                    current_q_config
                )
                subproblem_solve_total_time += (time.time() - q_sub_start_time)

                if quantum_route and len(quantum_route) > 2:
                    if len(subproblem_nodes_for_quantum) == 2 and quantum_route[0] == self.instance.depot and quantum_route[-1] == self.instance.depot:
                        pass 
                    elif len(subproblem_nodes_for_quantum) > 2 and (len(quantum_route)-1) != len(set(quantum_route)-{self.instance.depot}):
                        if self.verbose > 0:
                            # Prepare vqe_obj_val for printing safely
                            vqe_obj_str = 'N/A'
                            if vqe_obj_val is not None:
                                try: vqe_obj_str = f"{float(vqe_obj_val):.2f}"
                                except: vqe_obj_str = str(vqe_obj_val)
                            print(f"    Quantum route {quantum_route} might be suboptimal or incomplete for selected nodes {subproblem_nodes_for_quantum}. VQE Obj: {vqe_obj_str}")
                            
                    true_route_cost = calculate_solution_cost([quantum_route], self.instance)
                    route_load = sum(self.instance.demands[node] for node in quantum_route if node != self.instance.depot)
                    
                    current_iteration_routes.append(quantum_route)
                    current_iteration_route_costs.append(true_route_cost)
                    route_loads_this_iter.append(route_load)

                    newly_covered_in_route: Set[int] = set()
                    for node in quantum_route:
                        if node != self.instance.depot and node in self.customer_indices:
                            newly_covered_in_route.add(node)
                    
                    # ***** THIS IS THE CORRECTED PRINT BLOCK *****
                    if self.verbose > 1:
                        # Prepare strings for safe printing
                        str_tc = "ERR_TC"
                        try: str_tc = f"{float(true_route_cost):.2f}"
                        except: pass # Keep ERR_TC or handle more gracefully

                        str_vqe_obj = "N/A"
                        if vqe_obj_val is not None:
                            try: str_vqe_obj = f"{float(vqe_obj_val):.2f}"
                            except: str_vqe_obj = str(vqe_obj_val) # Fallback

                        print(f"    Quantum TSP subproblem for {subproblem_nodes_for_quantum} returned route: {quantum_route} "
                              f"(true cost: {str_tc}, load: {route_load}, VQE obj: {str_vqe_obj})")
                        print(f"    Customers newly covered by this route: {newly_covered_in_route}")
                    # ***** END OF CORRECTED PRINT BLOCK *****

                    customers_covered_this_alm_iter.update(newly_covered_in_route)
                else:
                    if self.verbose > 1:
                        print(f"    Quantum TSP for {subproblem_nodes_for_quantum} returned no valid route or a trivial one.")
            
            total_subproblem_original_cost = sum(current_iteration_route_costs)
            if self.verbose > 0:
                 print(f"  Subproblem (Quantum TSP) generated {len(current_iteration_routes)} routes "
                       f"with total original cost {total_subproblem_original_cost:.2f}. "
                       f"Total quantum solve time: {subproblem_solve_total_time:.2f}s")

            # --- Subsequent ALM logic (feasibility, violations, multiplier/penalty updates, logging, convergence) ---
            # ... (This part remains largely the same as your classical ALM, make sure it's complete) ...
            current_alm_solution = CVRPSolution(
                routes=current_iteration_routes,
                total_cost=total_subproblem_original_cost,
                instance_name=self.instance.name
            )
            is_overall_feasible, overall_violations_dict = check_solution_feasibility(current_alm_solution, self.instance)
            current_alm_solution.is_feasible = is_overall_feasible
            current_alm_solution.feasibility_details = overall_violations_dict

            g_i_values: List[float] = [0.0] * self.num_customers
            max_abs_g_i_violation = 0.0
            customer_visit_counts = {cust_idx: 0 for cust_idx in self.customer_indices}
            for r_item in current_iteration_routes: # Renamed r to r_item
                for node in r_item:
                    if node != self.instance.depot and node in self.customer_indices:
                        customer_visit_counts[node] += 1
            
            for i, cust_node_idx in enumerate(self.customer_indices):
                visits = customer_visit_counts.get(cust_node_idx, 0)
                g_i_values[i] = float(visits - 1)
                max_abs_g_i_violation = max(max_abs_g_i_violation, abs(g_i_values[i]))

            num_gen_routes = len(current_iteration_routes)
            capacity_violations_h_k: List[float] = [0.0] * self.subproblem_max_vehicles_to_generate # Initialize for all potential routes
            max_cap_violation_val = 0.0
            for k_idx in range(num_gen_routes): # Renamed k to k_idx
                load_k = route_loads_this_iter[k_idx]
                violation_k = max(0, load_k - self.instance.capacity)
                capacity_violations_h_k[k_idx] = violation_k
                max_cap_violation_val = max(max_cap_violation_val, violation_k)
            
            if self.verbose > 0:
                avg_g_i_viol = sum(abs(g) for g in g_i_values) / self.num_customers if self.num_customers else 0
                avg_h_k_viol = sum(capacity_violations_h_k[:num_gen_routes]) / num_gen_routes if num_gen_routes else 0 # Avg over generated routes
                print(f"  Max customer visit |g_i|: {max_abs_g_i_violation:.2f} (Avg: {avg_g_i_viol:.2f})")
                print(f"  Max capacity viol |h_k|: {max_cap_violation_val:.2f} (Avg: {avg_h_k_viol:.2f})")

            if is_overall_feasible:
                if self.verbose > 0 or not self.best_feasible_solution or current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    print(f"  Current ALM solution is FEASIBLE overall. Cost: {current_alm_solution.total_cost:.2f}")
                if self.best_feasible_solution is None or \
                   current_alm_solution.total_cost < self.best_feasible_solution.total_cost:
                    self.best_feasible_solution = current_alm_solution
                    print(f"  NEW BEST FEASIBLE solution found. Cost: {self.best_feasible_solution.total_cost:.2f}")
            elif self.verbose > 0:
                print(f"  Current ALM solution is INFEASIBLE. Violations (sample): {dict(list(overall_violations_dict.items())[:3])}")

            for i in range(self.num_customers):
                self.lambdas_visit[i] = self.lambdas_visit[i] + self.rhos_visit[i] * g_i_values[i]
            for k_idx in range(num_gen_routes): 
                 self.mus_capacity[k_idx] = max(0, self.mus_capacity[k_idx] + self.sigmas_capacity[k_idx] * capacity_violations_h_k[k_idx])
            
            if max_abs_g_i_violation > self.convergence_tolerance : 
                for i in range(self.num_customers):
                    if abs(g_i_values[i]) > self.convergence_tolerance * 0.1: 
                        self.rhos_visit[i] = min(self.rhos_visit[i] * self.penalty_increase_factor, self.max_penalty_rate)
            if max_cap_violation_val > self.capacity_convergence_tolerance:
                for k_idx in range(num_gen_routes):
                    if capacity_violations_h_k[k_idx] > self.capacity_convergence_tolerance * 0.1:
                        self.sigmas_capacity[k_idx] = min(self.sigmas_capacity[k_idx] * self.capacity_penalty_increase_factor,
                                                      self.max_capacity_penalty_rate)
            
            iter_time = time.time() - iter_start_time
            log_entry = {
                "iteration": alm_iter + 1, "subproblem_original_cost": total_subproblem_original_cost,
                "num_routes_generated": len(current_iteration_routes), "max_abs_g_i": max_abs_g_i_violation,
                "avg_abs_g_i": sum(abs(g) for g in g_i_values) / self.num_customers if self.num_customers else 0,
                "max_cap_violation": max_cap_violation_val,
                "avg_cap_violation": sum(h for h in capacity_violations_h_k[:num_gen_routes] if h > 0) / num_gen_routes if num_gen_routes else 0,
                "is_feasible_overall": is_overall_feasible,
                "best_feasible_cost": self.best_feasible_solution.total_cost if self.best_feasible_solution else float('inf'),
                "avg_lambda_visit": sum(self.lambdas_visit)/len(self.lambdas_visit) if self.lambdas_visit else 0,
                "avg_rho_visit": sum(self.rhos_visit)/len(self.rhos_visit) if self.rhos_visit else 0,
                "avg_mu_capacity": sum(self.mus_capacity)/len(self.mus_capacity) if self.mus_capacity else 0,
                "avg_sigma_capacity": sum(self.sigmas_capacity)/len(self.sigmas_capacity) if self.sigmas_capacity else 0,
                "time_subproblem_solve_sec": subproblem_solve_total_time, "time_alm_iter_sec": iter_time
            }
            self.iteration_log.append(log_entry)
            if self.verbose > 0: print(f"  Iteration {alm_iter+1} log: {log_entry}")

            customer_visit_converged = max_abs_g_i_violation <= self.convergence_tolerance
            capacity_converged = max_cap_violation_val <= self.capacity_convergence_tolerance
            
            if customer_visit_converged and capacity_converged and is_overall_feasible:
                if self.verbose >= 0: 
                    print(f"ALM converged at iteration {alm_iter + 1} with feasible solution and all violations within tolerance.")
                break
        
        if alm_iter == self.max_alm_iterations - 1 and self.verbose >=0 :
            print("ALM reached max iterations.")
        total_alm_time = time.time() - alm_start_time
        if self.verbose >=0: print(f"\nTotal ALM (Quantum) optimization time: {total_alm_time:.2f} seconds.")

        if self.best_feasible_solution:
            if self.verbose >=0: print(f"Best feasible solution found with cost: {self.best_feasible_solution.total_cost:.2f}")
            final_check_feasible, final_violations = check_solution_feasibility(self.best_feasible_solution, self.instance)
            self.best_feasible_solution.is_feasible = final_check_feasible
            self.best_feasible_solution.feasibility_details = final_violations
        elif self.verbose >=0: print("No feasible solution found by ALM.")
        return self.best_feasible_solution


if __name__ == '__main__':
    print("--- Testing AlmOptimizerQuantum ---")

    # Depot:0, Customers: 1, 2, 3. Capacity: 100
    # Demands: d1=10, d2=15, d3=20
    # Distances:
    # D-1:10, D-2:15, D-3:20
    # 1-2:5,  1-3:12
    # 2-3:8
    instance_3_cust = CVRPInstance(
        name="3_customer_quantum_test",
        dimension=4, # Depot + 3 customers
        capacity=100,
        distance_matrix=np.array([
            [0, 10, 15, 20], # Depot to 0,1,2,3
            [10, 0,  5, 12], # C1 to 0,1,2,3
            [15, 5,  0,  8], # C2 to 0,1,2,3
            [20, 12, 8,  0]  # C3 to 0,1,2,3
        ]),
        demands=[0, 10, 15, 20], # d_depot, d1, d2, d3
        depot=0,
        num_vehicles_comment=2 # Hint, may need 2 vehicles
    )

    q_config = {
        'max_customers_in_quantum_subproblem': 2, 
        'constraint_penalty_factor': 600.0, # Might need adjustment
        'vqe_reps': 1,                            
        'vqe_max_iter': 50, # Keep low for initial tests                      
        'plot_folder_prefix': "output_plots_alm_q_3cust_"
        # 'vqe_optimizer_method': "SPSA" # Already default in your __init__
    }

    alm_q_optimizer = AlmOptimizerQuantum(
        instance=instance_3_cust,
        initial_penalty_rate=1.0,
        max_alm_iterations=15, # Allow a few more iterations
        subproblem_max_vehicles=2, # Allow up to 2 routes
        verbose=2, 
        quantum_solver_config=q_config
    )

    print(f"\nStarting ALM solve for '{instance_3_cust.name}'...")
    solution = alm_q_optimizer.solve()

    if solution:
        print("\n--- Final Solution from AlmOptimizerQuantum (3 Cust) ---")
        # ... (rest of your solution printing logic) ...
        print(f"Instance: {solution.instance_name}, Feasible: {solution.is_feasible}, Total Cost: {solution.total_cost}")
        print("Routes:")
        for i, route in enumerate(solution.routes):
            route_load = sum(instance_3_cust.demands[node] for node in route if node != instance_3_cust.depot)
            print(f"  Route {i+1}: {route} (Load: {route_load})")
        if solution.feasibility_details: print(f"Feasibility Details: {solution.feasibility_details}")

    else:
        print("\n--- No solution found by AlmOptimizerQuantum (3 Cust) ---")

    print("\n--- Iteration Log (3 Cust) ---")
    for log_entry in alm_q_optimizer.iteration_log: print(log_entry)