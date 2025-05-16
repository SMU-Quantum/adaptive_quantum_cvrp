# === src/quantum_solver.py ===

import numpy as np
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp
from docplex.mp.model import Model

from qiskit.circuit.library import EfficientSU2
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2

# Initialize Qiskit backend and primitives
backend = AerSimulator(method="matrix_product_state")
estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend, options={"default_shots": 8000})


def qubo_to_qp(Q, offset=0.0) -> QuadraticProgram:
    """Convert QUBO dictionary into a Qiskit QuadraticProgram."""
    mdl = Model()
    index_set = set()
    for i, j in Q.keys():
        index_set.add(i)
        index_set.add(j)
    n = max(index_set) + 1
    x = [mdl.binary_var(name=f'x_{i}') for i in range(n)]
    objective = offset
    for (i, j), coeff in Q.items():
        objective += coeff * x[i] * x[j]
    mdl.minimize(objective)
    return from_docplex_mp(mdl)


def run_vqe(qp, reps=2, optimizer_method="COBYLA", max_iter=10, plot_folder="output_plots", optimizer_kwargs=None):
    """
    Run VQE using EfficientSU2 ansatz and Qiskit's Estimator primitive.

    Returns a dictionary with the best energy, bitstring, full distribution,
    and convergence information.
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    start_time = time.time()

    # Convert to Ising Hamiltonian
    qubitOp, offset = qp.to_ising()
    num_qubits = qubitOp.num_qubits

    # Prepare parameterized ansatz
    ansatz = EfficientSU2(num_qubits, reps=reps).decompose(reps=3)
    num_params = ansatz.num_parameters

    # Track optimization
    cost_history = {
        "iters": 0,
        "costs": [],
        "params": None,
    }

    def cost_func(params):
        pub = (ansatz, [qubitOp], [params])
        result = estimator.run(pubs=[pub]).result()
        energy = result[0].data.evs[0]
        cost_history["iters"] += 1
        cost_history["costs"].append(energy)
        cost_history["params"] = params
        print(f"[Iter {cost_history['iters']}] Energy: {energy:.6f}")
        return energy

    # Optimize parameters
    x0 = 2 * np.pi * np.random.rand(num_params)
    res = minimize(cost_func, x0, method=optimizer_method, options={"maxiter": max_iter, **optimizer_kwargs})
    best_params = res.x

    # Sample from final ansatz
    ansatz = ansatz.assign_parameters(best_params)
    ansatz.measure_all()
    job = sampler.run([(ansatz,)], shots=8000)
    result = job.result()[0].data.meas
    counts_int = result.get_int_counts()

    # Extract most probable bitstring
    shots = sum(counts_int.values())
    final_dist = {k: v / shots for k, v in counts_int.items()}
    best_int = max(final_dist, key=final_dist.get)
    best_bitstring = list(map(int, np.binary_repr(best_int, width=num_qubits)))
    best_bitstring.reverse()

    # Plot cost vs iteration
    os.makedirs(plot_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plot_folder, f"cost_vs_iterations_{timestamp}.png")

    plt.plot(cost_history["costs"])
    plt.xlabel("Iteration")
    plt.ylabel("Estimated Energy")
    plt.title("VQE Convergence")
    plt.grid(True)
    plt.savefig(plot_path)
    plt.close()

    duration = time.time() - start_time

    return {
        "energy": res.fun + offset,
        "bitstring": best_bitstring,
        "bitstring_str": " | ".join("".join(map(str, best_bitstring[i:i+8])) for i in range(0, len(best_bitstring), 8)),
        "distribution": final_dist,
        "iterations": cost_history["iters"],
        "converged": res.success,
        "params": best_params,
        "time_taken_sec": duration,
        "plot_path": plot_path
    }