# # src/adaptive_quantum_cvrp/common/cvrp_instance.py

# """
# This class is responsible for parsing and storing the CVRP problem data from a file.
# """

# from __future__ import annotations
# import numpy as np
# from typing import List, Dict, Tuple

# class CVRPInstance:
#     """
#     Represents a Capacitated Vehicle Routing Problem (CVRP) instance.

#     This class parses a .vrp file in the CVRPLIB format and stores the problem
#     data, including node coordinates, demands, and vehicle capacity.

#     Attributes:
#         name (str): The name of the problem instance.
#         num_customers (int): The number of customers (excluding the depot).
#         capacity (int): The capacity of each vehicle.
#         depot_id (int): The ID of the depot node, typically 0.
#         nodes (np.ndarray): A NumPy array of node coordinates, where the first
#                             node is the depot. Shape: (num_nodes, 2).
#         demands (np.ndarray): A NumPy array of customer demands. The depot's
#                               demand is 0. Shape: (num_nodes,).
#         dist_matrix (np.ndarray): A square matrix of Euclidean distances
#                                   between all pairs of nodes.
#                                   Shape: (num_nodes, num_nodes).
#     """

#     def __init__(self, name: str, num_customers: int, capacity: int,
#                  nodes: np.ndarray, demands: np.ndarray):
#         """Initializes the CVRPInstance."""
#         self.name = name
#         self.num_customers = num_customers
#         self.capacity = capacity
#         self.depot_id = 0
#         self.nodes = nodes
#         self.demands = demands
#         self.dist_matrix = self._compute_distance_matrix()

#     @property
#     def num_nodes(self) -> int:
#         """Returns the total number of nodes (customers + depot)."""
#         return self.num_customers + 1

#     def _compute_distance_matrix(self) -> np.ndarray:
#         """Computes the Euclidean distance matrix between all nodes."""
#         num_nodes = self.num_nodes
#         dist_matrix = np.zeros((num_nodes, num_nodes))
#         for i in range(num_nodes):
#             for j in range(i, num_nodes):
#                 dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
#                 dist_matrix[i, j] = dist_matrix[j, i] = dist
#         return dist_matrix

#     @classmethod
#     def from_file(cls, filepath: str) -> CVRPInstance:
#         """
#         Parses a .vrp file and creates a CVRPInstance.

#         Args:
#             filepath (str): The path to the .vrp file.

#         Returns:
#             CVRPInstance: An object representing the parsed CVRP instance.
        
#         Raises:
#             ValueError: If the file format is incorrect or sections are missing.
#         """
#         with open(filepath, 'r') as f:
#             lines = f.readlines()

#         name = ""
#         num_customers = 0
#         capacity = 0
#         node_coords: Dict[int, Tuple[float, float]] = {}
#         demands: Dict[int, int] = {}
        
#         section = ""
#         for line in lines:
#             line = line.strip()
#             if not line or line.startswith("COMMENT"):
#                 continue
            
#             if line.startswith("NAME"):
#                 name = line.split(":")[1].strip()
#             elif line.startswith("DIMENSION"):
#                 num_customers = int(line.split(":")[1].strip()) - 1
#             elif line.startswith("CAPACITY"):
#                 capacity = int(line.split(":")[1].strip())
#             elif line.startswith("NODE_COORD_SECTION"):
#                 section = "COORDS"
#             elif line.startswith("DEMAND_SECTION"):
#                 section = "DEMANDS"
#             elif line.startswith("DEPOT_SECTION"):
#                 section = "DEPOT"
#             elif section:
#                 parts = line.split()
#                 if not parts: continue
                
#                 node_id = int(parts[0]) - 1 # 0-indexed

#                 if section == "COORDS":
#                     node_coords[node_id] = (float(parts[1]), float(parts[2]))
#                 elif section == "DEMANDS":
#                     demands[node_id] = int(parts[1])
#                 elif line == "-1":
#                     section = "" # End of section

#         if not all([name, num_customers > 0, capacity > 0, node_coords, demands]):
#             raise ValueError(f"Failed to parse all required sections from {filepath}")

#         num_nodes = num_customers + 1
#         nodes_arr = np.array([node_coords[i] for i in range(num_nodes)])
#         demands_arr = np.array([demands[i] for i in range(num_nodes)])

#         return cls(name, num_customers, capacity, nodes_arr, demands_arr)

#     def __repr__(self) -> str:
#         return (f"CVRPInstance(name='{self.name}', "
#                 f"num_customers={self.num_customers}, "
#                 f"capacity={self.capacity})")

# src/adaptive_quantum_cvrp/common/cvrp_instance.py

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging


import re
from pathlib import Path


class CVRPInstance:
    """
    Represents a Capacitated Vehicle Routing Problem (CVRP) instance.

    This class parses a .vrp file and stores the problem data. It can handle
    files with either node coordinates (NODE_COORD_SECTION) or explicit distance
    matrices (EDGE_WEIGHT_SECTION).

    Attributes:
        name (str): The name of the problem instance.
        num_customers (int): The number of customers (excluding the depot).
        capacity (int): The capacity of each vehicle.
        depot_id (int): The ID of the depot node, typically 0.
        nodes (Optional[np.ndarray]): NumPy array of node coordinates, if available.
        demands (np.ndarray): NumPy array of customer demands.
        dist_matrix (np.ndarray): A square matrix of distances between all nodes.
    """

    def __init__(self, name: str, num_customers: int, capacity: int,
                 demands: np.ndarray, num_vehicles: int, nodes: Optional[np.ndarray] = None,
                 dist_matrix: Optional[np.ndarray] = None):
        """Initializes the CVRPInstance."""
        self.name = name
        self.num_customers = num_customers
        self.capacity = capacity
        self.depot_id = 0
        self.demands = demands
        self.nodes = nodes
        self.num_vehicles = num_vehicles
        
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        elif nodes is not None:
            self.dist_matrix = self._compute_distance_matrix()
        else:
            raise ValueError("Instance must have either node coordinates or a distance matrix.")

    @property
    def num_nodes(self) -> int:
        """Returns the total number of nodes (customers + depot)."""
        return self.num_customers + 1

    def _compute_distance_matrix(self) -> np.ndarray:
        """Computes the Euclidean distance matrix between all nodes."""
        if self.nodes is None:
            raise ValueError("Cannot compute distance matrix without node coordinates.")
        num_nodes = self.num_nodes
        dist_matrix = np.zeros((num_nodes, num_nodes))
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                dist = np.linalg.norm(self.nodes[i] - self.nodes[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    @classmethod
    def from_file(cls, filepath: str) -> CVRPInstance:
        """
        Parses a .vrp file and creates a CVRPInstance.
        Handles both coordinate-based and explicit matrix-based formats.
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        name = ""
        num_customers = 0
        capacity = 0
        num_vehicles = 0
        node_coords: Dict[int, Tuple[float, float]] = {}
        demands: Dict[int, int] = {}
        edge_weights: List[int] = []
        is_explicit_matrix = False
        
        section = ""
        for line in lines:
            if not line or line.startswith("COMMENT"): continue
            
            if line.startswith("NAME"): name = line.split(":")[1].strip()
            elif line.startswith("DIMENSION"): num_customers = int(line.split(":")[1].strip()) - 1
            elif line.startswith("CAPACITY"): capacity = int(line.split(":")[1].strip())
            elif line.startswith("EDGE_WEIGHT_TYPE") and "EXPLICIT" in line: is_explicit_matrix = True
            elif line.startswith("NODE_COORD_SECTION"): section = "COORDS"
            elif line.startswith("DEMAND_SECTION"): section = "DEMANDS"
            elif line.startswith("DEPOT_SECTION"): section = "DEPOT"
            elif line.startswith("EDGE_WEIGHT_SECTION"): section = "EDGES"
            elif section and "SECTION" not in line and "EOF" not in line:
                parts = line.split()
                if not parts: continue
                
                if section == "COORDS":
                    node_id = int(parts[0]) - 1
                    node_coords[node_id] = (float(parts[1]), float(parts[2]))
                elif section == "DEMANDS":
                    node_id = int(parts[0]) - 1
                    demands[node_id] = int(parts[1])
                elif section == "EDGES":
                    edge_weights.extend(map(int, parts))
                elif line == "-1":
                    section = ""

        # --- Finalize parsed data ---
        num_nodes = num_customers + 1
        nodes_arr: Optional[np.ndarray] = None
        dist_matrix_arr: Optional[np.ndarray] = None

        if node_coords:
            nodes_arr = np.array([node_coords[i] for i in range(num_nodes)])
        
        if is_explicit_matrix and edge_weights:
            dist_matrix_arr = np.zeros((num_nodes, num_nodes))
            idx = 0
            # Assumes LOWER_ROW format for the explicit matrix
            for i in range(num_nodes):
                for j in range(i):
                    dist_matrix_arr[i, j] = dist_matrix_arr[j, i] = edge_weights[idx]
                    idx += 1

        if not demands:
            raise ValueError(f"Failed to parse DEMAND_SECTION from {filepath}")
        demands_arr = np.array([demands[i] for i in range(num_nodes)])

        match = re.search(r'-k(\d+)', Path(filepath).stem)
        if match:
            num_vehicles = int(match.group(1))
        
        if num_vehicles == 0:
            logging.warning(f"Could not parse num_vehicles from filename: {filepath}. Defaulting to 0.")



        return cls(name, num_customers, capacity, demands_arr, num_vehicles,
                   nodes=nodes_arr, dist_matrix=dist_matrix_arr)

    def __repr__(self) -> str:
        return (f"CVRPInstance(name='{self.name}', "
                f"num_customers={self.num_customers}, "
                f"capacity={self.capacity}, "
                f"num_vehicles={self.num_vehicles})") 