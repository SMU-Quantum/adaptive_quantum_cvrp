# src/adaptive_quantum_cvrp/quantum/qubo.py

"""
This module is responsible for constructing the QUBO matrix for the ALM subproblem.
"""


import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import SparsePauliOp
from typing import Tuple

from ..common.cvrp_instance import CVRPInstance

def convert_cvrp_to_ising(
    instance: CVRPInstance,
    # lagrange_multipliers: np.ndarray,
    num_vehicles: int,
    penalty: float = 1e5
) -> Tuple[SparsePauliOp, float, QuadraticProgram]:
    """
    Constructs the CVRP as a QuadraticProgram and converts it to an Ising Hamiltonian.

    This uses an arc-based formulation where x_i,j is a binary variable that is 1
    if the path from node i to node j is taken, and 0 otherwise.

    Args:
        instance: The CVRP problem instance.
        lagrange_multipliers: Lagrange multipliers (not used in this simplified formulation
                              but kept for interface consistency).
        num_vehicles: The number of vehicles available.
        penalty: The penalty coefficient for constraint violations in the QUBO.

    Returns:
        A tuple containing:
        - The Ising Hamiltonian (qubit_op).
        - A constant energy offset.
        - The original QuadraticProgram for decoding purposes.
    """
    n_nodes = instance.num_nodes
    customers = list(range(1, n_nodes))

    # 1. Build the Quadratic Program
    qp = QuadraticProgram(name="CVRP")

    # Add binary variables x_i,j for arcs
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                qp.binary_var(f"x_{i}_{j}")

    # 2. Define the objective function (minimize travel distance)
    objective = {}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                objective[f"x_{i}_{j}"] = instance.dist_matrix[i, j]
    qp.minimize(linear=objective)

    # 3. Define constraints
    # Each customer must be visited exactly once
    for i in customers:
        in_arcs = {f"x_{j}_{i}": 1 for j in range(n_nodes) if j != i}
        qp.linear_constraint(in_arcs, sense="==", rhs=1, name=f"in_{i}")

        out_arcs = {f"x_{i}_{j}": 1 for j in range(n_nodes) if j != i}
        qp.linear_constraint(out_arcs, sense="==", rhs=1, name=f"out_{i}")
    
    # Exactly num_vehicles must leave the depot
    depot_out_arcs = {f"x_{0}_{j}": 1 for j in customers}
    qp.linear_constraint(depot_out_arcs, sense="==", rhs=num_vehicles, name="vehicle_out")

    # 4. Convert to QUBO and then to Ising Hamiltonian
    qubo_converter = QuadraticProgramToQubo(penalty=penalty)
    qubo = qubo_converter.convert(qp)
    
    qubit_op, offset = qubo.to_ising()
    return qubit_op, offset, qp