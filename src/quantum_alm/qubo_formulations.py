# # src/quantum_alm/qubo_formulations.py

# import numpy as np
# from typing import Optional
# from qiskit_optimization import QuadraticProgram
# from qiskit_optimization.applications import Tsp # Keep this import
# from qiskit_optimization.converters import QuadraticProgramToQubo

# # --- formulate_defined_path_qubo function remains unchanged ---
# def formulate_defined_path_qubo(
#     adj_matrix: np.ndarray,
#     defined_nodes_in_path: list, 
#     node_rewards: dict, 
#     edge_cost_factor: float = 1.0,
#     reward_factor: float = 1.0, 
#     path_enforcement_penalty: float = 100.0
# ) -> tuple[QuadraticProgram, dict, dict]:
#     # ... (implementation as before) ...
#     qp = QuadraticProgram(name="DefinedPathQUBO")
#     var_map = {} 
#     var_idx_to_edge_map_internal = {}
#     if not defined_nodes_in_path or len(defined_nodes_in_path) < 2:
#         return qp, var_map, {}
#     path_edges = []
#     for i in range(len(defined_nodes_in_path) - 1):
#         u, v = defined_nodes_in_path[i], defined_nodes_in_path[i+1]
#         if u == v: continue
#         path_edges.append((u,v))
#     for u, v in path_edges:
#         var_name = f"x_{u}_{v}"
#         qp.binary_var(name=var_name)
#         var_map[(u,v)] = var_name
#     current_idx = 0
#     for qp_var in qp.variables: 
#         parts = qp_var.name.split('_')
#         u_idx, v_idx = int(parts[1]), int(parts[2])
#         var_idx_to_edge_map_internal[current_idx] = (u_idx, v_idx) 
#         current_idx += 1
#     linear_coeffs = {} 
#     quadratic_coeffs = {} 
#     constant_offset = 0.0
#     for u, v in path_edges:
#         var_name = var_map[(u,v)]
#         distance_cost = adj_matrix[u,v] * edge_cost_factor
#         linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) + distance_cost
#         reward_v = node_rewards.get(v, 0.0) * reward_factor
#         linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) - reward_v
#         constant_offset += path_enforcement_penalty
#         linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) - path_enforcement_penalty
#     qp.minimize(constant=constant_offset, linear=linear_coeffs, quadratic=quadratic_coeffs)
#     return qp, var_map, var_idx_to_edge_map_internal


# def formulate_tsp_subproblem_qubo(
#     full_adj_matrix: np.ndarray,
#     subproblem_nodes: list,
#     node_rewards: dict,
#     edge_cost_factor: float = 1.0,
#     reward_factor: float = 1.0,
#     constraint_penalty_factor: float = 200.0
# ) -> tuple[Optional[QuadraticProgram], Optional[dict], Optional[Tsp]]: # Added Tsp to return
#     """
#     Formulates a QUBO for a TSP subproblem.
#     Returns the QuadraticProgram, the node map, and the Tsp model instance.
#     """
#     num_sub_nodes = len(subproblem_nodes)
#     if num_sub_nodes < 3:
#         print("[Warning] Subproblem too small for TSP. Needs at least 3 nodes for Tsp class.")
#         return None, None, None # Return None for Tsp model as well

#     # ... (sub_adj_matrix creation as before) ...
#     orig_to_sub_node_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(subproblem_nodes)}
#     sub_to_orig_node_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(subproblem_nodes)}
#     sub_adj_matrix = np.zeros((num_sub_nodes, num_sub_nodes))
#     for i in range(num_sub_nodes):
#         for j in range(num_sub_nodes):
#             if i == j:
#                 continue
#             orig_node_u = sub_to_orig_node_map[i]
#             orig_node_v = sub_to_orig_node_map[j]
#             distance = full_adj_matrix[orig_node_u, orig_node_v]
#             reward_at_v = 0
#             if orig_node_v != subproblem_nodes[0]: 
#                  reward_at_v = node_rewards.get(orig_node_v, 0.0)
#             cost_uv = (edge_cost_factor * distance) - (reward_factor * reward_at_v)
#             sub_adj_matrix[i, j] = cost_uv
            
#     tsp_model_instance = None # Initialize
#     try:
#         tsp_model_instance = Tsp(sub_adj_matrix) # This is the instance we need
#         qp_with_constraints = tsp_model_instance.to_quadratic_program()
#     except Exception as e:
#         print(f"[Error] Failed to create TSP model or convert to QP: {e}")
#         return None, None, None
        
#     if qp_with_constraints.linear_constraints or qp_with_constraints.quadratic_constraints:
#         print("[Info] Problem has constraints. Converting to QUBO using QuadraticProgramToQubo...")
#         converter = QuadraticProgramToQubo(penalty=constraint_penalty_factor)
#         qubo_qp = converter.convert(qp_with_constraints)
#         # print(f"[Info] Conversion to QUBO complete. Original problem had {len(qp_with_constraints.linear_constraints)} linear and {len(qp_with_constraints.quadratic_constraints)} quadratic constraints.")
#         # print(f"[Info] New QUBO problem has {len(qubo_qp.linear_constraints)} linear and {len(qubo_qp.quadratic_constraints)} quadratic constraints (should be 0).")
#     else:
#         # This case is less likely for Tsp default formulation but good to have
#         print("[Info] Problem from Tsp class has no explicit constraints. Using as is.")
#         qubo_qp = qp_with_constraints

#     return qubo_qp, sub_to_orig_node_map, tsp_model_instance # Return the Tsp instance


# # --- if __name__ == '__main__': block remains the same ---
# if __name__ == '__main__':
#     # ... (previous test for formulate_defined_path_qubo can remain) ...

#     print("\n\n--- Testing formulate_tsp_subproblem_qubo ---")
#     full_adj_matrix_test = np.array([
#         [0, 10, 15, 20], [10, 0, 5,  12], [15, 5,  0,  8], [20, 12, 8,  0]   
#     ])
#     sub_nodes_test = [0, 1, 2] 
#     rewards_test = {0:0, 1:2, 2:3, 3:1} 
    
#     # Note the change in return values
#     qp_tsp, node_map_tsp, tsp_instance_for_decode = formulate_tsp_subproblem_qubo(
#         full_adj_matrix_test, sub_nodes_test, rewards_test,
#         constraint_penalty_factor=500.0 
#     )
    
#     if qp_tsp:
#         print("\n--- Quadratic Program for TSP Subproblem [0,1,2] (QUBO form) ---")
#         print(f"Number of variables in TSP QUBO: {qp_tsp.get_num_vars()}")
#         print(f"Number of linear constraints: {len(qp_tsp.linear_constraints)}")
#         print(f"Number of quadratic constraints: {len(qp_tsp.quadratic_constraints)}")
#         print("\nSubproblem node map (sub_idx -> orig_idx):")
#         print(node_map_tsp)
#     else:
#         print("Failed to formulate TSP QUBO.")

# src/quantum_alm/qubo_formulations.py

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.applications import Tsp
from qiskit_optimization.converters import QuadraticProgramToQubo
from typing import Optional, List, Dict, Tuple # Added for type hinting

# --- formulate_defined_path_qubo function remains unchanged ---
def formulate_defined_path_qubo(
    adj_matrix: np.ndarray,
    defined_nodes_in_path: list,
    node_rewards: dict,
    edge_cost_factor: float = 1.0,
    reward_factor: float = 1.0,
    path_enforcement_penalty: float = 100.0
) -> tuple[QuadraticProgram, dict, dict]:
    qp = QuadraticProgram(name="DefinedPathQUBO")
    var_map = {}
    var_idx_to_edge_map_internal = {}
    if not defined_nodes_in_path or len(defined_nodes_in_path) < 2:
        return qp, var_map, {}
    path_edges = []
    for i in range(len(defined_nodes_in_path) - 1):
        u, v = defined_nodes_in_path[i], defined_nodes_in_path[i+1]
        if u == v: continue
        path_edges.append((u,v))
    for u, v in path_edges:
        var_name = f"x_{u}_{v}"
        qp.binary_var(name=var_name)
        var_map[(u,v)] = var_name
    current_idx = 0
    # Assuming qp.variables are created in the order qp.binary_var was called.
    # This mapping is crucial for the simple defined_path decoder.
    temp_edge_list_for_idx_map = [] 
    for u_edge, v_edge in path_edges:
        temp_edge_list_for_idx_map.append((u_edge,v_edge))

    for i, qp_var in enumerate(qp.variables):
        if i < len(temp_edge_list_for_idx_map): # Safety check
            var_idx_to_edge_map_internal[i] = temp_edge_list_for_idx_map[i]
        else: # Should not happen if var creation order matches path_edges
            # Fallback or raise error if var name parsing is needed but not robust
            try:
                parts = qp_var.name.split('_')
                u_idx, v_idx = int(parts[1]), int(parts[2])
                var_idx_to_edge_map_internal[i] = (u_idx, v_idx)
            except:
                 raise ValueError(f"Could not map variable {qp_var.name} at index {i} back to an edge.")


    linear_coeffs = {}
    quadratic_coeffs = {}
    constant_offset = 0.0
    for u, v in path_edges:
        var_name = var_map[(u,v)]
        distance_cost = adj_matrix[u,v] * edge_cost_factor # This adj_matrix assumed to be np.array
        linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) + distance_cost
        reward_v = node_rewards.get(v, 0.0) * reward_factor
        linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) - reward_v
        constant_offset += path_enforcement_penalty
        linear_coeffs[var_name] = linear_coeffs.get(var_name, 0) - path_enforcement_penalty
    qp.minimize(constant=constant_offset, linear=linear_coeffs, quadratic=quadratic_coeffs)
    return qp, var_map, var_idx_to_edge_map_internal


def formulate_tsp_subproblem_qubo(
    full_adj_matrix: List[List[float]] | np.ndarray, # Accept list of lists or ndarray
    subproblem_nodes: List[int],
    node_rewards: Dict[int, float],
    edge_cost_factor: float = 1.0,
    reward_factor: float = 1.0,
    constraint_penalty_factor: float = 200.0
) -> Tuple[Optional[QuadraticProgram], Optional[Dict[int,int]], Optional[Tsp]]:
    """
    Formulates a QUBO for a TSP subproblem.
    Returns the QuadraticProgram, the node map, and the Tsp model instance.
    """
    num_sub_nodes = len(subproblem_nodes)
    if num_sub_nodes < 3:
        print("[Warning] Subproblem too small for TSP. Needs at least 3 nodes for Tsp class.")
        return None, None, None

    # ***** ADDED FIX: Ensure full_adj_matrix is a NumPy array *****
    if not isinstance(full_adj_matrix, np.ndarray):
        matrix_to_use = np.array(full_adj_matrix, dtype=float)
        if matrix_to_use.ndim != 2:
             raise ValueError(f"full_adj_matrix, if a list, must be convertible to a 2D array. Got shape {matrix_to_use.shape}")
    else:
        matrix_to_use = full_adj_matrix
    # ***** END OF FIX *****

    orig_to_sub_node_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(subproblem_nodes)}
    sub_to_orig_node_map = {new_idx: orig_idx for new_idx, orig_idx in enumerate(subproblem_nodes)}

    sub_adj_matrix = np.zeros((num_sub_nodes, num_sub_nodes))
    for i in range(num_sub_nodes):
        for j in range(num_sub_nodes):
            if i == j:
                continue
            orig_node_u = sub_to_orig_node_map[i]
            orig_node_v = sub_to_orig_node_map[j]
            
            # Use matrix_to_use (which is guaranteed to be a NumPy array) for indexing
            distance = matrix_to_use[orig_node_u, orig_node_v]
            
            reward_at_v = 0
            if orig_node_v != subproblem_nodes[0]:
                 reward_at_v = node_rewards.get(orig_node_v, 0.0)
            cost_uv = (edge_cost_factor * distance) - (reward_factor * reward_at_v)
            sub_adj_matrix[i, j] = cost_uv
            
    tsp_model_instance = None
    try:
        tsp_model_instance = Tsp(sub_adj_matrix)
        qp_with_constraints = tsp_model_instance.to_quadratic_program()
    except Exception as e:
        print(f"[Error] Failed to create TSP model or convert to QP: {e}")
        return None, None, None
        
    if qp_with_constraints.linear_constraints or qp_with_constraints.quadratic_constraints:
        print("[Info] Problem has constraints. Converting to QUBO using QuadraticProgramToQubo...")
        converter = QuadraticProgramToQubo(penalty=constraint_penalty_factor)
        qubo_qp = converter.convert(qp_with_constraints)
    else:
        print("[Info] Problem from Tsp class has no explicit constraints. Using as is.")
        qubo_qp = qp_with_constraints

    return qubo_qp, sub_to_orig_node_map, tsp_model_instance


# --- if __name__ == '__main__': block ---
if __name__ == '__main__':
    print("--- Testing formulate_defined_path_qubo ---")
    # ... (defined path test - make sure adj_matrix here is also np.array if not already)
    sample_adj_matrix_np = np.array([ # Ensuring this one is numpy array for its test
        [0, 10, 15], [10, 0, 5], [15, 5, 0]
    ])
    sample_defined_nodes = [0, 1, 2, 0]
    sample_rewards = {0: 0, 1: 2, 2: 3} 
    qp_test, v_map, v_idx_map = formulate_defined_path_qubo(
        sample_adj_matrix_np, sample_defined_nodes, sample_rewards,
        edge_cost_factor=1.0, reward_factor=1.0, path_enforcement_penalty=100.0 
    )
    # ... (rest of defined path test print logic)

    print("\n\n--- Testing formulate_tsp_subproblem_qubo ---")
    full_adj_matrix_test_np = np.array([ # Ensuring this one is numpy array for its test
        [0, 10, 15, 20], [10, 0, 5,  12], [15, 5,  0,  8], [20, 12, 8,  0]   
    ])
    sub_nodes_test = [0, 1, 2] 
    rewards_test = {0:0, 1:2, 2:3, 3:1} 
    
    qp_tsp, node_map_tsp, tsp_instance_for_decode = formulate_tsp_subproblem_qubo(
        full_adj_matrix_test_np, sub_nodes_test, rewards_test,
        constraint_penalty_factor=500.0 
    )
    
    if qp_tsp:
        print("\n--- Quadratic Program for TSP Subproblem [0,1,2] (QUBO form) ---")
        print(f"Number of variables in TSP QUBO: {qp_tsp.get_num_vars()}")
        print(f"Number of linear constraints: {len(qp_tsp.linear_constraints)}")
        print(f"Number of quadratic constraints: {len(qp_tsp.quadratic_constraints)}")
        print("\nSubproblem node map (sub_idx -> orig_idx):")
        print(node_map_tsp)
    else:
        print("Failed to formulate TSP QUBO.")