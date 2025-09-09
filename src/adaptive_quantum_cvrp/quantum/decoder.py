# src/adaptive_quantum_cvrp/quantum/decoder.py

"""
It samples from the final VQE circuit to find the most likely solution bitstring and 
then reconstructs the vehicle routes from the activated arcs.
"""


from qiskit.primitives import BackendSamplerV2
from qiskit_optimization import QuadraticProgram
from typing import List, Dict, Any
from qiskit_aer import AerSimulator
from ..common.cvrp_solution import CVRPSolution

def decode_solution(
    vqe_output: Dict[str, Any],
    qp: QuadraticProgram
) -> CVRPSolution:
    """
    Decodes the VQE result by sampling the optimal circuit.

    Args:
        vqe_output: The dictionary returned by the `run_vqe` function.
        qp: The QuadraticProgram defining the problem variables.

    Returns:
        The decoded CVRP solution.
    """
    ansatz = vqe_output["ansatz"]
    optimal_params = vqe_output["optimal_params"]
    
    # Bind optimal parameters to the ansatz to create the final circuit
    final_circuit = ansatz.assign_parameters(optimal_params)
    final_circuit.measure_all() # Add measurements to all qubits

    # 1. Instantiate a backend (the simulator)
    backend = AerSimulator(method="matrix_product_state")
    
    # 2. Instantiate the V2 Sampler with the backend
    sampler = BackendSamplerV2(backend=backend)
    

    # 3. Run the sampler job
    # The run method takes the circuit and optional shots
    result = sampler.run([final_circuit], shots=1024).result()
    



    # 4. Get the measurement counts
    # The result object holds the data in a structured way.
    counts = result[0].data.meas.get_counts()

    if not counts:
        return CVRPSolution(routes=[]) # Return empty solution if no counts
    
    best_bitstring = max(counts, key=counts.get)
    
    
    # Translate bitstring back to variable assignments
    solution_vars = qp.variables
    
    var_values = {var.name: int(bit) for var, bit in zip(solution_vars, best_bitstring[::-1])}
    # var_values = {var.name: int(bit) for var, bit in zip(solution_vars, best_bitstring)}

    # Reconstruct routes from the arc variables (x_i_j)
    adj = {i: [] for i in range(qp.get_num_vars())}
    for var_name, value in var_values.items():
        if value == 1 and var_name.startswith("x_"):
            try:
                _, i, j = var_name.split('_')
                adj[int(i)].append(int(j))
            except (IndexError, ValueError):
                continue

    routes: List[List[int]] = []
    
    # Find routes starting from the depot (node 0)
    if 0 not in adj: return CVRPSolution(routes=[])

    visited = set()
    for start_node in adj[0]:
        if start_node in visited: continue
        
        current_route = [start_node]
        visited.add(start_node)
        
        curr = start_node
        while curr in adj and adj[curr]:
            # Handle cases where a node might have multiple outbound arcs in a bad sample
            next_node = -1

            for node in adj[curr]:
                if node not in visited:
                    next_node = node
                    break
            
            if next_node == -1 or next_node == 0: break # Route ends or gets stuck
            
            current_route.append(next_node)
            visited.add(next_node)
            curr = next_node
        
        routes.append(current_route)
        
    return CVRPSolution(routes=routes)