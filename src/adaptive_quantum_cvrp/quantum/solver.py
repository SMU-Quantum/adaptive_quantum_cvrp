# src/adaptive_quantum_cvrp/quantum/solver.py

import numpy as np
from typing import Any, Dict

from ..alm.classical_solver import SubproblemSolver
from ..common.cvrp_instance import CVRPInstance
from ..common.cvrp_solution import CVRPSolution
from .qubo import convert_cvrp_to_ising
from .vqe_runner import run_vqe
from .decoder import decode_solution

class QuantumSolver:
    """
    A quantum-based solver for the ALM subproblem using VQE.
    Implements the SubproblemSolver protocol.
    """
    def __init__(self, num_vehicles: int, vqe_options: Dict[str, Any]):
        self.num_vehicles = num_vehicles
        self.vqe_options = vqe_options

    def solve(self, instance: CVRPInstance, lagrange_multipliers: np.ndarray,
              penalty_mu: float) -> CVRPSolution:
        """
        Solves the subproblem using the updated VQE workflow.
        """
        print("QuantumSolver: Converting CVRP to Ising Hamiltonian...")
        qubit_op, offset, qp = convert_cvrp_to_ising(
            instance, self.num_vehicles
        )

        print("QuantumSolver: Running VQE...")
        vqe_output = run_vqe(qubit_op, offset, self.vqe_options.get("optimizer", {}))

        print("QuantumSolver: Decoding solution...")
        solution = decode_solution(vqe_output, qp)
        
        return solution