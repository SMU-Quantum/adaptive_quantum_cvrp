# # src/quantum_alm/qubo_vqe_route_solver.py

# import numpy as np
# from qiskit_optimization import QuadraticProgram

# from .quantum_vqe_runner import qubo_to_qp, run_vqe 
# from .qubo_formulations import formulate_defined_path_qubo, formulate_tsp_subproblem_qubo
# from .solution_decoders import decode_path_from_bitstring, decode_tsp_solution_from_qiskit_tsp

# # --- test_solve_defined_path function remains unchanged ---
# def test_solve_defined_path():
#     print("--- Testing Quantum Solver for a Defined Path ---")
#     adj_matrix = np.array([
#         [0, 10, 15, 20], [10, 0, 5,  12], [15, 5,  0,  8], [20, 12, 8,  0]   
#     ])
#     defined_nodes_in_path = [0, 1, 2, 0]
#     node_rewards = {0:0, 1:2, 2:3, 3:0} 
#     depot_node = 0
#     path_enforcement_penalty = 200.0
#     print("\n--- Formulating QUBO for Defined Path ---")
#     qp, var_map, var_idx_to_edge_map_internal = formulate_defined_path_qubo(
#         adj_matrix, defined_nodes_in_path, node_rewards,
#         path_enforcement_penalty=path_enforcement_penalty
#     )
#     if qp.get_num_vars() == 0:
#         print("QUBO has no variables, skipping VQE.")
#         return
#     print("\n--- Running VQE ---")
#     vqe_params = { "reps": 2, "max_iter": 50, "optimizer_method": "Powell", "plot_folder": "output_plots_defined_path"}
#     vqe_result_dict = run_vqe(qp, **vqe_params)
#     print("\n--- VQE Result Dict ---")
#     print(f"  Energy (from VQE result, includes QUBO offset): {vqe_result_dict.get('energy')}")
#     print(f"  Optimal Bitstring: {vqe_result_dict.get('bitstring')}")
#     print(f"  Converged: {vqe_result_dict.get('converged')}")
#     if vqe_result_dict and vqe_result_dict.get("bitstring") is not None:
#         print("\n--- Decoding VQE Solution ---")
#         optimal_bitstring = vqe_result_dict["bitstring"]
#         objective_value = vqe_result_dict["energy"]
#         # MockSolutionSample not needed with direct bitstring passing to specialized decoder
#         decoded_path_nodes, path_fval_from_decoder = decode_path_from_bitstring(
#             bitstring_values=optimal_bitstring, fval=objective_value, qp=qp,
#             var_idx_to_edge_map=var_idx_to_edge_map_internal, expected_start_node=depot_node
#         )
#         if decoded_path_nodes:
#             print(f"\nSuccessfully decoded path: {decoded_path_nodes}")
#             print(f"Objective value for this path (from VQE via decoder): {path_fval_from_decoder}") 
#             manual_cost = 0; actual_rewards_collected = 0; visited_for_reward = set()
#             for i in range(len(decoded_path_nodes) - 1):
#                 u, v = decoded_path_nodes[i], decoded_path_nodes[i+1]
#                 manual_cost += adj_matrix[u,v]
#                 if v not in visited_for_reward and v != depot_node : 
#                     actual_rewards_collected += node_rewards.get(v,0); visited_for_reward.add(v)
#             print(f"Manually calculated path distance: {manual_cost}")
#             print(f"Manually calculated rewards collected: {actual_rewards_collected}")
#             print(f"Manually calculated (Dist - Reward): {manual_cost - actual_rewards_collected}")
#         else: print("\nFailed to decode a valid path from VQE output.")
#     else: print("\nVQE did not return a valid result or bitstring.")
#     print("\n--- Defined Path Quantum Solver Test Complete ---")


# def test_solve_tsp_subproblem():
#     print("\n\n--- Testing Quantum Solver for a TSP Subproblem ---")
#     full_adj_matrix = np.array([
#         [0, 10, 15, 20], [10, 0, 5,  12], [15, 5,  0,  8], [20, 12, 8,  0]   
#     ])
#     subproblem_nodes_to_visit = [0, 1, 2] 
#     node_rewards = {0:0, 1:2, 2:3, 3:0} 
        
#     print("\n--- Formulating QUBO for TSP Subproblem ---")
#     # Note the change in return values to include tsp_model_instance_for_decode
#     qubo_qp_tsp, sub_to_orig_map, tsp_model_instance_for_decode = formulate_tsp_subproblem_qubo(
#         full_adj_matrix,
#         subproblem_nodes_to_visit,
#         node_rewards,
#         edge_cost_factor=1.0,
#         reward_factor=1.0,
#         constraint_penalty_factor=500.0 # Adjusted penalty
#     )

#     if not qubo_qp_tsp:
#         print("Failed to formulate TSP QUBO for subproblem. Exiting test.")
#         return
        
#     print(f"Subproblem to original node map: {sub_to_orig_map}")
#     print(f"Number of QUBO variables for TSP: {qubo_qp_tsp.get_num_vars()}")

#     print("\n--- Running VQE for TSP Subproblem ---")
#     vqe_params = {
#         "reps": 2, 
#         "max_iter": 200, # Might need more for SPSA on 9 vars
#         "optimizer_method": "Powell", 
#         "plot_folder": "output_plots_tsp_subproblem"
#     }
#     vqe_result_dict = run_vqe(qubo_qp_tsp, **vqe_params)

#     print("\n--- VQE Result Dict (TSP) ---")
#     print(f"  Energy (from VQE result): {vqe_result_dict.get('energy')}")
#     print(f"  Optimal Bitstring (TSP): {vqe_result_dict.get('bitstring')}")
#     print(f"  Converged: {vqe_result_dict.get('converged')}")

#     if vqe_result_dict and vqe_result_dict.get("bitstring") is not None:
#         print("\n--- Decoding VQE TSP Solution ---")
        
#         # Pass the tsp_model_instance to the decoder
#         decoded_tsp_tour_orig_nodes, path_fval = decode_tsp_solution_from_qiskit_tsp(
#             vqe_solution_dict=vqe_result_dict,
#             tsp_model_instance=tsp_model_instance_for_decode, # Use the returned Tsp instance
#             sub_to_orig_node_map=sub_to_orig_map
#         )

#         if decoded_tsp_tour_orig_nodes:
#             print(f"\nSuccessfully decoded TSP tour (original nodes): {decoded_tsp_tour_orig_nodes}")
#             print(f"Objective value for this tour (from VQE): {path_fval}")
            
#             manual_dist_cost = 0; actual_rewards_collected = 0; visited_for_reward = set()
#             depot_node = subproblem_nodes_to_visit[0]
#             for i in range(len(decoded_tsp_tour_orig_nodes) - 1):
#                 u, v = decoded_tsp_tour_orig_nodes[i], decoded_tsp_tour_orig_nodes[i+1]
#                 manual_dist_cost += full_adj_matrix[u,v]
#                 if v != depot_node and v not in visited_for_reward:
#                     actual_rewards_collected += node_rewards.get(v,0); visited_for_reward.add(v)
#             print(f"Manually calculated tour distance (original): {manual_dist_cost}")
#             print(f"Manually calculated rewards collected (original): {actual_rewards_collected}")
#             print(f"Manually calculated (Original Dist - Original Reward): {manual_dist_cost - actual_rewards_collected}")
#             print(f"(Compare VQE energy {path_fval} to expected TSP subproblem cost ~25, now including constraint penalties from QUBO conversion)")
#         else:
#             print("\nFailed to decode a valid TSP tour from VQE output.")
#     else:
#         print("\nVQE did not return a valid result or bitstring for TSP.")
#     print("\n--- TSP Subproblem Quantum Solver Test Complete ---")

# if __name__ == '__main__':
#     # test_solve_defined_path() 
#     test_solve_tsp_subproblem()


# src/quantum_alm/qubo_vqe_route_solver.py

import numpy as np
from typing import Optional, List, Dict, Tuple # For type hinting
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Tsp as TspAppClass


from .quantum_vqe_runner import qubo_to_qp, run_vqe 
from .qubo_formulations import formulate_defined_path_qubo, formulate_tsp_subproblem_qubo
from .solution_decoders import decode_path_from_bitstring, decode_tsp_solution_from_qiskit_tsp

# --- Public function for ALM to call ---
def solve_tsp_subproblem_quantum(
    full_adj_matrix: np.ndarray,
    subproblem_nodes_to_visit: List[int], # e.g., [depot, c1, c2]
    node_rewards: Dict[int, float],       # ALM's modified_node_rewards
    config: Dict                             # For various parameters
) -> Tuple[Optional[List[int]], Optional[float]]:
    """
    Solves a TSP subproblem for a given subset of nodes using QUBO-VQE.

    Args:
        full_adj_matrix: Distance matrix for the entire CVRP problem.
        subproblem_nodes_to_visit: List of original node IDs for this subproblem.
        node_rewards: Current rewards/penalties from ALM for each node.
        config: Dictionary containing configuration parameters:
            - 'edge_cost_factor': float (default 1.0)
            - 'reward_factor': float (default 1.0)
            - 'constraint_penalty_factor': float (for QuadraticProgramToQubo, e.g., 500.0)
            - 'vqe_reps': int (e.g., 2)
            - 'vqe_max_iter': int (e.g., 200)
            - 'vqe_optimizer_method': str (e.g., "SPSA")
            - 'plot_folder_prefix': str (e.g., "output_plots_alm_subproblem_")

    Returns:
        A tuple (decoded_route, vqe_objective_value):
            - decoded_route: List of original node IDs forming the tour (e.g., [0, 2, 1, 0]), or None.
            - vqe_objective_value: The energy value from VQE, or None.
    """
    print(f"\n[Quantum Subproblem] Solving for nodes: {subproblem_nodes_to_visit}")

    edge_cost_factor = config.get('edge_cost_factor', 1.0)
    reward_factor = config.get('reward_factor', 1.0)
    constraint_penalty = config.get('constraint_penalty_factor', 500.0)
    
    qubo_qp_tsp, sub_to_orig_map, tsp_model_instance = formulate_tsp_subproblem_qubo(
        full_adj_matrix,
        subproblem_nodes_to_visit,
        node_rewards,
        edge_cost_factor=edge_cost_factor,
        reward_factor=reward_factor,
        constraint_penalty_factor=constraint_penalty # Pass this to the formulation
    )

    if not qubo_qp_tsp:
        print("[Quantum Subproblem] Failed to formulate TSP QUBO.")
        return None, None
    
    if not tsp_model_instance: # Should be caught by qubo_qp_tsp check, but good practice
        print("[Quantum Subproblem] TSP model instance not created.")
        return None, None

    # print(f"[Quantum Subproblem] QUBO vars: {qubo_qp_tsp.get_num_vars()}")

    vqe_params_dict = {
        "reps": config.get('vqe_reps', 2),
        "max_iter": config.get('vqe_max_iter', 200),
        "optimizer_method": config.get('vqe_optimizer_method', "Powell"),
        "plot_folder": f"{config.get('plot_folder_prefix', 'output_plots_subproblem_')}{'_'.join(map(str, subproblem_nodes_to_visit))}"
    }
    
    vqe_result_dict = run_vqe(qubo_qp_tsp, **vqe_params_dict)

    if not vqe_result_dict or vqe_result_dict.get("bitstring") is None:
        print("[Quantum Subproblem] VQE did not return a valid result or bitstring.")
        return None, None

    # print(f"[Quantum Subproblem] VQE Energy: {vqe_result_dict.get('energy')}, Bitstring: {vqe_result_dict.get('bitstring')}")

    decoded_tour_orig_nodes, path_fval = decode_tsp_solution_from_qiskit_tsp(
        vqe_solution_dict=vqe_result_dict,
        tsp_model_instance=tsp_model_instance,
        sub_to_orig_node_map=sub_to_orig_map
    )

    if decoded_tour_orig_nodes:
        print(f"[Quantum Subproblem] Decoded Route: {decoded_tour_orig_nodes}, VQE Obj: {path_fval:.3f}")
    else:
        print("[Quantum Subproblem] Failed to decode a valid tour.")
        
    return decoded_tour_orig_nodes, path_fval


# --- test_solve_defined_path function remains unchanged ---
def test_solve_defined_path():
    # ... (implementation as before, for brevity) ...
    print("--- Testing Quantum Solver for a Defined Path ---")
    adj_matrix = np.array([
        [0, 10, 15, 20], [10, 0, 5,  12], [15, 5,  0,  8], [20, 12, 8,  0]   
    ])
    defined_nodes_in_path = [0, 1, 2, 0]
    node_rewards = {0:0, 1:2, 2:3, 3:0} 
    depot_node = 0
    path_enforcement_penalty = 200.0
    # print("\n--- Formulating QUBO for Defined Path ---")
    qp, var_map, var_idx_to_edge_map_internal = formulate_defined_path_qubo(
        adj_matrix, defined_nodes_in_path, node_rewards,
        path_enforcement_penalty=path_enforcement_penalty
    )
    if qp.get_num_vars() == 0: return
    # print("\n--- Running VQE ---")
    vqe_params = { "reps": 2, "max_iter": 50, "optimizer_method": "Powell", "plot_folder": "output_plots_defined_path"}
    vqe_result_dict = run_vqe(qp, **vqe_params)
    # ... (rest of defined path test print/decode logic as before) ...
    if vqe_result_dict and vqe_result_dict.get("bitstring") is not None:
        optimal_bitstring = vqe_result_dict["bitstring"]; objective_value = vqe_result_dict["energy"]
        decoded_path_nodes, _ = decode_path_from_bitstring(
            bitstring_values=optimal_bitstring, fval=objective_value, qp=qp,
            var_idx_to_edge_map=var_idx_to_edge_map_internal, expected_start_node=depot_node)
        if decoded_path_nodes: print(f"Defined Path Decoded: {decoded_path_nodes}, VQE Obj: {objective_value:.3f}")
    print("\n--- Defined Path Quantum Solver Test Complete ---")


def test_solve_tsp_subproblem_via_main_function():
    """
    Tests the new solve_tsp_subproblem_quantum function.
    """
    print("\n\n--- Testing solve_tsp_subproblem_quantum Main Function ---")
    full_adj_matrix = np.array([
        [0, 10, 15, 20], 
        [10, 0, 5,  12], 
        [15, 5,  0,  8],  
        [20, 12, 8,  0]   
    ])
    subproblem_nodes = [0, 1, 2] # Depot 0, Cust 1, Cust 2
    rewards = {0:0, 1:2, 2:3, 3:0} 
    
    config = {
        'edge_cost_factor': 1.0,
        'reward_factor': 1.0,
        'constraint_penalty_factor': 500.0, # Penalty for Tsp constraints in QUBO
        'vqe_reps': 2,
        'vqe_max_iter': 200, # Iterations for VQE's classical optimizer
        'vqe_optimizer_method': "Powell",
        'plot_folder_prefix': "output_plots_tsp_test_"
    }

    route, vqe_obj = solve_tsp_subproblem_quantum(
        full_adj_matrix,
        subproblem_nodes,
        rewards,
        config
    )

    if route:
        print(f"\nTest Result - Route: {route}, VQE Objective: {vqe_obj}")
        # Manual cost check
        manual_dist_cost = 0; actual_rewards_collected = 0; visited_for_reward = set()
        depot_node = subproblem_nodes[0]
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            manual_dist_cost += full_adj_matrix[u,v]
            if v != depot_node and v not in visited_for_reward:
                actual_rewards_collected += rewards.get(v,0); visited_for_reward.add(v)
        print(f"  Manually calculated tour distance (original): {manual_dist_cost}")
        print(f"  Manually calculated rewards collected (original): {actual_rewards_collected}")
        print(f"  Manually calculated (Original Dist - Original Reward): {manual_dist_cost - actual_rewards_collected}")
    else:
        print("\nTest Result - Failed to find a route.")
    
    print("\n--- Test of solve_tsp_subproblem_quantum Main Function Complete ---")


if __name__ == '__main__':
    # test_solve_defined_path() 
    # test_solve_tsp_subproblem() # You can comment out the old direct test
    test_solve_tsp_subproblem_via_main_function() # Test the new callable function