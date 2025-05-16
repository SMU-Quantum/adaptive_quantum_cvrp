# # src/quantum_alm/solution_decoders.py

# import numpy as np
# from qiskit_optimization import QuadraticProgram
# from qiskit_optimization.applications import Tsp as TspAppClass
# from typing import Optional
# # --- decode_path_from_bitstring function remains unchanged ---
# def decode_path_from_bitstring( 
#     bitstring_values: list,      
#     fval: Optional[float],       
#     qp: QuadraticProgram,        
#     var_idx_to_edge_map: dict, 
#     expected_start_node: int
# ) -> tuple[Optional[list], Optional[float]]:
#     # ... (implementation as before) ...
#     if not isinstance(bitstring_values, (list, np.ndarray)):
#         print("[Error] Invalid bitstring_values type in decoder. Expected list or numpy array.")
#         return None, None
#     chosen_edges = []
#     for i, val in enumerate(bitstring_values):
#         if np.isclose(val, 1.0): 
#             if i in var_idx_to_edge_map:
#                 chosen_edges.append(var_idx_to_edge_map[i])
#             # else: # Commented out for brevity
#                 # print(f"[Warning] Index {i} from bitstring not found in var_idx_to_edge_map.")
#     if not chosen_edges:
#         # print("[Info] No edges chosen in the bitstring.") # Can be noisy
#         return None, fval
#     path = [expected_start_node]
#     current_node = expected_start_node
#     edge_lookup = {u:v for u,v in chosen_edges if u != v} 
#     for _ in range(len(chosen_edges) + 1): 
#         if current_node in edge_lookup:
#             next_node = edge_lookup[current_node]
#             path.append(next_node)
#             del edge_lookup[current_node] 
#             current_node = next_node
#             if current_node == expected_start_node and len(path) > 1: 
#                 break 
#             if len(path) > len(var_idx_to_edge_map) + 1 : 
#                 print("[Warning] Path reconstruction seems to be looping or too long.")
#                 return None, fval 
#         else:
#             if not edge_lookup: 
#                 break
#             else: 
#                 print(f"[Warning] Path broken at node {current_node}. Chosen edges: {chosen_edges}, Path so far: {path}, Remaining lookup: {edge_lookup}")
#                 return None, fval
#     return path, fval


# def decode_tsp_solution_from_qiskit_tsp(
#     vqe_solution_dict: dict,
#     tsp_model_instance: TspAppClass,
#     sub_to_orig_node_map: dict,
# ) -> tuple[Optional[list], Optional[float]]:
#     bitstring_list = vqe_solution_dict.get("bitstring") # This is a Python list
#     energy = vqe_solution_dict.get("energy")

#     if bitstring_list is None:
#         print("[Error] 'bitstring' not found in VQE solution dictionary.")
#         return None, energy
    
#     if tsp_model_instance is None:
#         print("[Error] 'tsp_model_instance' is None, cannot interpret solution.")
#         return None, energy

#     try:
#         # Convert the Python list to a NumPy array of floats or ints
#         # Tsp.interpret usually expects numeric types that can be cast to bool for binary vars.
#         numpy_bitstring = np.array(bitstring_list, dtype=float) # Using float, as x often represents probabilities before thresholding
        
#         sub_problem_tour_indices = tsp_model_instance.interpret(numpy_bitstring)
#     except Exception as e:
#         print(f"[Error] Could not interpret TSP solution from bitstring using Tsp.interpret: {e}")
#         print(f"Bitstring was: {bitstring_list} (type: {type(bitstring_list)})")
#         print(f"Converted NumPy bitstring was: {numpy_bitstring} (type: {type(numpy_bitstring)}, dtype: {numpy_bitstring.dtype})")
#         return None, energy
        
#     if not sub_problem_tour_indices: # Catches empty list or None
#         print("[Warning] Tsp.interpret returned no tour or an empty tour.")
#         return None, energy

#     original_nodes_tour = [sub_to_orig_node_map[sub_idx] for sub_idx in sub_problem_tour_indices]
    
#     if original_nodes_tour:
#         full_original_tour = original_nodes_tour + [original_nodes_tour[0]]
#         return full_original_tour, energy
#     else:
#         return None, energy

# src/quantum_alm/solution_decoders.py

import numpy as np
from qiskit_optimization import QuadraticProgram 
from qiskit_optimization.applications import Tsp as TspAppClass # Use an alias
from typing import Optional, List, Dict, Tuple, Any # Added Any for flexibility

# --- decode_path_from_bitstring function remains unchanged ---
def decode_path_from_bitstring( 
    bitstring_values: list,      
    fval: Optional[float],       
    qp: QuadraticProgram,        
    var_idx_to_edge_map: dict, 
    expected_start_node: int
) -> tuple[Optional[list], Optional[float]]:
    if not isinstance(bitstring_values, (list, np.ndarray)):
        print("[Error] Invalid bitstring_values type in decoder. Expected list or numpy array.")
        return None, None
    chosen_edges = []
    for i, val in enumerate(bitstring_values):
        if np.isclose(val, 1.0): 
            if i in var_idx_to_edge_map:
                chosen_edges.append(var_idx_to_edge_map[i])
            # else: # Commented out for brevity
                # print(f"[Warning] Index {i} from bitstring not found in var_idx_to_edge_map.")
    if not chosen_edges:
        # print("[Info] No edges chosen in the bitstring.") # Can be noisy
        return None, fval
    path = [expected_start_node]
    current_node = expected_start_node
    edge_lookup = {u:v for u,v in chosen_edges if u != v} 
    for _ in range(len(chosen_edges) + 1): 
        if current_node in edge_lookup:
            next_node = edge_lookup[current_node]
            path.append(next_node)
            del edge_lookup[current_node] 
            current_node = next_node
            if current_node == expected_start_node and len(path) > 1: 
                break 
            if len(path) > len(var_idx_to_edge_map) + 1 : 
                print("[Warning] Path reconstruction seems to be looping or too long.")
                return None, fval 
        else:
            if not edge_lookup: 
                break
            else: 
                print(f"[Warning] Path broken at node {current_node}. Chosen edges: {chosen_edges}, Path so far: {path}, Remaining lookup: {edge_lookup}")
                return None, fval
    return path, fval


def decode_tsp_solution_from_qiskit_tsp(
    vqe_solution_dict: dict,
    tsp_model_instance: TspAppClass, 
    sub_to_orig_node_map: dict,
) -> tuple[Optional[list], Optional[float]]:
    """
    Decodes a TSP solution using the 'interpret' method of the Tsp model instance.
    """
    bitstring_list = vqe_solution_dict.get("bitstring") 
    energy = vqe_solution_dict.get("energy")

    if bitstring_list is None:
        print("[Error] 'bitstring' not found in VQE solution dictionary.")
        return None, energy
    
    if tsp_model_instance is None:
        print("[Error] 'tsp_model_instance' is None, cannot interpret solution.")
        return None, energy

    sub_problem_tour_indices = None # Initialize
    try:
        numpy_bitstring = np.array(bitstring_list, dtype=float) 
        sub_problem_tour_indices = tsp_model_instance.interpret(numpy_bitstring)
        
        # --- Add Debugging Prints ---
        print(f"    DEBUG_DECODER: sub_problem_tour_indices from Tsp.interpret: {sub_problem_tour_indices}")
        if sub_problem_tour_indices:
            print(f"    DEBUG_DECODER: type of sub_problem_tour_indices: {type(sub_problem_tour_indices)}")
            if isinstance(sub_problem_tour_indices, list) and len(sub_problem_tour_indices) > 0:
                print(f"    DEBUG_DECODER: type of first element in tour: {type(sub_problem_tour_indices[0])}")
        # --- End Debugging Prints ---

    except Exception as e:
        print(f"[Error] Could not interpret TSP solution from bitstring using Tsp.interpret: {e}")
        print(f"    Bitstring was: {bitstring_list} (type: {type(bitstring_list)})")
        if 'numpy_bitstring' in locals(): # Check if numpy_bitstring was defined
             print(f"    Converted NumPy bitstring was: {numpy_bitstring} (type: {type(numpy_bitstring)}, dtype: {numpy_bitstring.dtype})")
        return None, energy
        
    if not sub_problem_tour_indices: 
        print("[Warning] Tsp.interpret returned no tour or an empty tour.")
        return None, energy

    original_nodes_tour = []
    try:
        for sub_idx_item in sub_problem_tour_indices:
            # Potential Fix: If sub_idx_item is a list containing a single integer
            if isinstance(sub_idx_item, list) and len(sub_idx_item) == 1 and isinstance(sub_idx_item[0], (int, np.integer)):
                actual_sub_idx = sub_idx_item[0]
                print(f"    DEBUG_DECODER: Corrected sub_idx_item from {sub_idx_item} to {actual_sub_idx}")
            elif isinstance(sub_idx_item, (int, np.integer)):
                actual_sub_idx = sub_idx_item
            else:
                # If it's neither an int nor a list of a single int, this is an unexpected format
                print(f"[Error] Unexpected item type in sub_problem_tour_indices: {sub_idx_item} (type: {type(sub_idx_item)})")
                return None, energy # Cannot proceed
            
            original_nodes_tour.append(sub_to_orig_node_map[actual_sub_idx])

    except TypeError as te: # Catch the unhashable type error specifically if the fix above isn't enough
        print(f"[Error] TypeError during mapping of sub_problem_tour_indices: {te}")
        print(f"    Problematic sub_problem_tour_indices: {sub_problem_tour_indices}")
        return None, energy
    except KeyError as ke: # Catch key error if an index is not in map
        print(f"[Error] KeyError during mapping: index {ke} not in sub_to_orig_node_map.")
        print(f"    Problematic sub_problem_tour_indices: {sub_problem_tour_indices}")
        print(f"    sub_to_orig_node_map: {sub_to_orig_node_map}")
        return None, energy


    if original_nodes_tour:
        full_original_tour = original_nodes_tour + [original_nodes_tour[0]]
        return full_original_tour, energy
    else:
        # This case might be redundant if the loop doesn't produce original_nodes_tour
        print("[Warning] Failed to construct original_nodes_tour, possibly due to empty or malformed sub_problem_tour_indices.")
        return None, energy
