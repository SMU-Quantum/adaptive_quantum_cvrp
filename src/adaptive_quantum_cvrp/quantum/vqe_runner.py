# src/adaptive_quantum_cvrp/quantum/vqe_runner.py

"""
This module abstracts the interaction with the quantum computing framework (e.g., Qiskit). It takes a QUBO and runs the VQE algorithm.
"""

# src/adaptive_quantum_cvrp/quantum/vqe_runner.py

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from qiskit.circuit.library import RealAmplitudes, efficient_su2
from qiskit.primitives import BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from typing import Dict, Any

backend = AerSimulator(method="matrix_product_state")

# This is the cost function that the classical optimizer will minimize.
def _cost_func(params, ansatz, hamiltonian, estimator, cost_history_dict):
    """Callback function for the classical optimizer."""
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["cost_history"].append(energy)
    return energy

def run_vqe(
    qubit_op: SparsePauliOp,
    offset: float,
    optimizer_options: Dict[str, Any] = {'maxiter': 100}
) -> Dict[str, Any]:
    """
    Runs the VQE algorithm using a classical optimizer from SciPy.

    Args:
        qubit_op: The Ising Hamiltonian for the problem.
        offset: The energy offset from the QUBO conversion.
        optimizer_options: Options for the SciPy COBYLA optimizer.

    Returns:
        A dictionary containing the optimization result, final energy, and history.
    """
    estimator = BackendEstimatorV2(backend=backend)
    ansatz = efficient_su2(num_qubits=qubit_op.num_qubits, reps=1).decompose()
    num_params = ansatz.num_parameters
    
    x0 = 2 * np.pi * np.random.random(num_params)
    cost_history_dict = {"iters": 0, "cost_history": []}

    print(f"Optimizing {num_params} parameters using COBYLA...")
    res: OptimizeResult = minimize(
        _cost_func,
        x0,
        args=(ansatz, qubit_op, estimator, cost_history_dict),
        method="COBYLA",
        options=optimizer_options
    )

    final_energy = res.fun + offset
    print(f"VQE finished. Final energy: {final_energy:.4f}")

    return {
        "result": res,
        "final_energy": final_energy,
        "cost_history": cost_history_dict["cost_history"],
        "optimal_params": res.x,
        "ansatz": ansatz
    }