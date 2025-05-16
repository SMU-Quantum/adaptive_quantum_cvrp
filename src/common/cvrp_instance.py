# cvrp_tripartite_solver/src/common/cvrp_instance.py

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any

@dataclass
class CVRPInstance:
    """
    Represents a Capacitated Vehicle Routing Problem instance.
    All node indices (including the depot) are 0-based internally.
    Distances are integers, following common CVRPLIB practice (e.g., NINT for EUC_2D).
    """
    name: str
    dimension: int  # Number of nodes, including the depot
    capacity: int
    distance_matrix: List[List[int]]
    demands: List[int]  # 0 for depot
    depot: int  # 0-based index
    
    # Optional fields
    coordinates: Optional[List[Tuple[float, float]]] = None
    num_vehicles_comment: Optional[int] = None # Parsed from comments, often a hint
    edge_weight_type: Optional[str] = None
    edge_weight_format: Optional[str] = None
    full_comment: Optional[str] = None

    def __post_init__(self):
        """Basic validation after instance creation."""
        if not (0 <= self.depot < self.dimension):
            raise ValueError(f"Instance {self.name}: Depot index ({self.depot}) out of bounds for dimension {self.dimension}.")
        if len(self.demands) != self.dimension:
            raise ValueError(f"Instance {self.name}: Demands list length ({len(self.demands)}) != dimension ({self.dimension}).")
        if len(self.distance_matrix) != self.dimension or \
           any(len(row) != self.dimension for row in self.distance_matrix):
            raise ValueError(f"Instance {self.name}: Distance matrix is not {self.dimension}x{self.dimension}.")
        if self.coordinates and len(self.coordinates) != self.dimension:
            raise ValueError(f"Instance {self.name}: Coordinates list length ({len(self.coordinates)}) != dimension ({self.dimension}).")
        # Standard CVRP: depot demand should be 0.
        # This parser reflects the file's demand for the depot.
        # If a solver requires the depot demand to be 0, it should handle it.
        # For instance, one might log a warning or adjust it:
        # if self.demands[self.depot] != 0:
        #     print(f"Warning: Instance {self.name}: Depot demand is {self.demands[self.depot]}. Adjusting to 0 for standard model.")
        #     self.demands[self.depot] = 0


def _calculate_distances_from_coords(
    coordinates: List[Tuple[float, float]],
    dimension: int,
    dist_type: str
) -> List[List[int]]:
    """
    Calculates integer distances from coordinates based on the specified type.
    - EUC_2D: Rounded Euclidean distance (NINT).
    - CEIL_2D: Euclidean distance rounded up.
    - ATT: Pseudo-Euclidean distance from TSPLIB.
    """
    if len(coordinates) != dimension:
        raise ValueError(f"{dist_type}: Expected {dimension} coordinates, found {len(coordinates)}.")
    
    matrix = [[0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(i, dimension): # Symmetric matrix
            if i == j:
                matrix[i][j] = 0
                continue
            
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[j]
            xd, yd = x1 - x2, y1 - y2
            dist_euc_sq = xd**2 + yd**2 # Euclidean distance squared

            val: int
            if dist_type == "EUC_2D":
                val = int(round(math.sqrt(dist_euc_sq))) # NINT
            elif dist_type == "CEIL_2D":
                val = int(math.ceil(math.sqrt(dist_euc_sq)))
            elif dist_type == "ATT":
                # Formula: d(i,j) = NINT( sqrt( ((x_i-x_j)^2 + (y_i-y_j)^2) / 10.0 ) )
                # TSPLIB's NINT for ATT can be specific.
                # A common interpretation is to round to the nearest integer.
                rij = math.sqrt(dist_euc_sq / 10.0)
                val = int(round(rij))
                # Some ATT implementations use a specific rounding rule:
                # if val < rij: val = val + 1
                # For simplicity, we use standard NINT unless a dataset explicitly requires the other.
            else:
                raise NotImplementedError(f"Distance type '{dist_type}' from coordinates not supported.")
            
            matrix[i][j] = matrix[j][i] = val
    return matrix

def _parse_explicit_edge_weights(
    edge_weights_str_list: List[str],
    dimension: int,
    edge_weight_format: str
) -> List[List[int]]:
    """Parses EDGE_WEIGHT_SECTION based on format into an integer matrix."""
    matrix = [[0] * dimension for _ in range(dimension)]
    
    raw_weights: List[float] = [] # Read as float for flexibility, then round to int
    for item_line in edge_weights_str_list:
        raw_weights.extend(map(float, item_line.split())) 

    idx = 0
    if edge_weight_format == "FULL_MATRIX":
        if len(raw_weights) != dimension * dimension:
            raise ValueError(f"FULL_MATRIX: Expected {dimension*dimension} weights, got {len(raw_weights)}")
        for r in range(dimension):
            for c in range(dimension):
                matrix[r][c] = int(round(raw_weights[idx]))
                idx += 1
    elif edge_weight_format == "UPPER_ROW":
        expected_count = dimension * (dimension - 1) // 2
        if len(raw_weights) != expected_count:
            raise ValueError(f"UPPER_ROW: Expected {expected_count} weights, got {len(raw_weights)}")
        for r in range(dimension - 1):
            for c in range(r + 1, dimension):
                val = int(round(raw_weights[idx]))
                matrix[r][c] = matrix[c][r] = val
                idx += 1
    elif edge_weight_format == "LOWER_ROW":
        expected_count = dimension * (dimension - 1) // 2
        if len(raw_weights) != expected_count:
            raise ValueError(f"LOWER_ROW: Expected {expected_count} weights, got {len(raw_weights)}")
        for r in range(1, dimension):
            for c in range(r):
                val = int(round(raw_weights[idx]))
                matrix[r][c] = matrix[c][r] = val
                idx += 1
    # Add other formats like *_DIAG_* (e.g., UPPER_DIAG_ROW, LOWER_DIAG_ROW) if needed
    else:
        raise NotImplementedError(f"EDGE_WEIGHT_FORMAT '{edge_weight_format}' for EXPLICIT weights not supported.")
    return matrix


def load_cvrp_instance(file_path: str) -> CVRPInstance:
    """
    Loads a CVRP instance from a CVRPLIB formatted file.
    Converts depot and node indices to be 0-based.
    Distances are assumed/converted to integers.
    """
    instance_name_from_path = file_path.split('/')[-1].split('\\')[-1] # Basic name from path

    # Parsed values
    parsed_name: Optional[str] = None
    dimension: Optional[int] = None
    capacity: Optional[int] = None
    edge_weight_type: Optional[str] = None
    edge_weight_format: Optional[str] = None
    
    # Buffers for sections
    node_coords_lines: List[str] = []
    edge_weights_lines: List[str] = []
    demands_lines: List[str] = []
    depot_lines: List[str] = []
    
    comments_list: List[str] = []
    num_vehicles_comment: Optional[int] = None
    active_section: Optional[str] = None

    with open(file_path, 'r') as f:
        for line_raw in f:
            line = line_raw.strip()
            if not line or line.startswith("#"): # Skip empty lines or simple comments
                continue

            if line.startswith("COMMENT"):
                comment_content = line.split(":", 1)[1].strip() if ":" in line else line[7:].strip()
                comments_list.append(comment_content)
                # Attempt to parse number of vehicles from comment
                match_trucks = (
                    re.search(r"(?:No of trucks|Min no of trucks|Vehicles)\s*[:=]?\s*(\d+)", comment_content, re.IGNORECASE) or
                    re.search(r"k\s*=\s*(\d+)", comment_content, re.IGNORECASE) # Common in some comments
                )
                if match_trucks:
                    num_vehicles_comment = int(match_trucks.group(1))
                continue

            # Try to match keyword: value or just keyword
            # Handles "KEYWORD : VALUE", "KEYWORD: VALUE", "KEYWORD :VALUE", "KEYWORD:VALUE"
            keyword_match = re.match(r"([A-Z_0-9]+)\s*:\s*(.*)", line)
            if not keyword_match: # Try keyword without colon (for section headers)
                keyword_match = re.match(r"([A-Z_0-9]+)", line)
                if keyword_match: # If it's a known section keyword
                    section_keyword = keyword_match.group(1)
                    # Check if it's a section keyword that doesn't have a value on the same line
                    if section_keyword in ["NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", 
                                           "DEMAND_SECTION", "DEPOT_SECTION", "EOF",
                                           "TOUR_SECTION", "EDGE_DATA_SECTION", "FIXED_EDGES_SECTION",
                                           "DISPLAY_DATA_SECTION"]: # Add more section keywords if needed
                        keyword, value = section_keyword, ""
                    else: 
                        keyword, value = None, None 
                else:
                    keyword, value = None, None
            else: # keyword: value
                 keyword, value = keyword_match.groups()
                 value = value.strip()

            if keyword:
                # When a keyword is processed, it usually means we are no longer in a multi-line data section,
                # unless the keyword itself IS a section header.
                if keyword not in ["NODE_COORD_SECTION", "EDGE_WEIGHT_SECTION", "DEMAND_SECTION", "DEPOT_SECTION"]:
                    active_section = None # Reset section mode for most keywords
                
                if keyword == "NAME":
                    parsed_name = value
                elif keyword == "TYPE":
                    if value not in ["CVRP", "TSP", "VRP", "CVRPTW"]: # Basic check, CVRPTW for future
                        print(f"Warning: File {instance_name_from_path}: Potentially unsupported problem TYPE: {value}")
                elif keyword == "DIMENSION":
                    dimension = int(value)
                elif keyword == "CAPACITY":
                    capacity = int(value)
                elif keyword == "EDGE_WEIGHT_TYPE":
                    edge_weight_type = value
                elif keyword == "EDGE_WEIGHT_FORMAT":
                    edge_weight_format = value
                elif keyword == "NODE_COORD_SECTION":
                    active_section = "NODE_COORD"
                elif keyword == "EDGE_WEIGHT_SECTION":
                    active_section = "EDGE_WEIGHT"
                elif keyword == "DEMAND_SECTION":
                    active_section = "DEMAND"
                elif keyword == "DEPOT_SECTION":
                    active_section = "DEPOT"
                elif keyword == "EOF":
                    break
                # Add other keywords if needed, e.g., for CVRPTW
                continue # Processed keyword line

            # If not a keyword line, it must be data for an active section
            if active_section == "NODE_COORD":
                node_coords_lines.append(line)
            elif active_section == "EDGE_WEIGHT":
                edge_weights_lines.append(line)
            elif active_section == "DEMAND":
                demands_lines.append(line)
            elif active_section == "DEPOT":
                depot_lines.append(line)
    
    # --- Post-parsing validation and data construction ---
    final_name = parsed_name if parsed_name else instance_name_from_path
    if dimension is None: raise ValueError(f"Instance {final_name}: DIMENSION not found.")
    if capacity is None: raise ValueError(f"Instance {final_name}: CAPACITY not found.")

    # Coordinates
    coordinates: Optional[List[Tuple[float, float]]] = None
    if node_coords_lines:
        temp_coords = [ (0.0,0.0) ] * dimension # Pre-allocate assuming 1-based indexing in file
        parsed_coord_count = 0
        for item in node_coords_lines:
            parts = item.split()
            if len(parts) != 3: raise ValueError(f"Instance {final_name}: Malformed NODE_COORD entry: '{item}'")
            node_idx_1_based, x_coord, y_coord = int(parts[0]), float(parts[1]), float(parts[2])
            if not (1 <= node_idx_1_based <= dimension):
                 raise ValueError(f"Instance {final_name}: Node coord index {node_idx_1_based} out of bounds [1, {dimension}]")
            temp_coords[node_idx_1_based - 1] = (x_coord, y_coord)
            parsed_coord_count +=1
        if parsed_coord_count != dimension: # Check if all nodes got coordinates
            # Some files might not list all nodes if EDGE_WEIGHT_TYPE is EXPLICIT
            # but if NODE_COORD_SECTION exists, it should ideally be complete for types like EUC_2D
            if edge_weight_type in ["EUC_2D", "CEIL_2D", "ATT"]:
                 raise ValueError(f"Instance {final_name}: Expected {dimension} coord entries for {edge_weight_type}, got {parsed_coord_count}")
            # If not a coordinate-based distance, partial coordinates might be for display only.
        coordinates = temp_coords


    # Demands (1-based node_idx in file, store as 0-indexed list)
    final_demands = [0] * dimension
    if not demands_lines: raise ValueError(f"Instance {final_name}: DEMAND_SECTION missing or empty.")
    parsed_demand_count = 0
    for item in demands_lines:
        parts = item.split()
        if len(parts) != 2: raise ValueError(f"Instance {final_name}: Malformed DEMAND entry: '{item}'")
        node_idx_1_based, demand_val = int(parts[0]), int(parts[1])
        if not (1 <= node_idx_1_based <= dimension):
            raise ValueError(f"Instance {final_name}: Demand node_idx {node_idx_1_based} out of bounds [1, {dimension}]")
        final_demands[node_idx_1_based - 1] = demand_val
        parsed_demand_count +=1
    if parsed_demand_count != dimension: # All nodes must have a demand
        raise ValueError(f"Instance {final_name}: Expected {dimension} demand entries, parsed {parsed_demand_count}")

    # Depot (1-based in file, store as 0-indexed)
    depot_0_based: Optional[int] = None
    if not depot_lines: raise ValueError(f"Instance {final_name}: DEPOT_SECTION missing or empty.")
    parsed_depots_1_based = []
    for item in depot_lines:
        val = int(item.strip())
        if val == -1: break # Terminator
        if not (1 <= val <= dimension):
            raise ValueError(f"Instance {final_name}: Depot node_idx {val} out of bounds [1, {dimension}]")
        parsed_depots_1_based.append(val)
    
    if not parsed_depots_1_based: raise ValueError(f"Instance {final_name}: No depot found in DEPOT_SECTION.")
    # For standard CVRP, use the first depot listed.
    depot_0_based = parsed_depots_1_based[0] - 1


    # Distance Matrix
    distance_matrix: List[List[int]]
    if edge_weight_type in ["EUC_2D", "CEIL_2D", "ATT"]:
        if not coordinates: # Coordinates must have been parsed
            raise ValueError(f"Instance {final_name}: EDGE_WEIGHT_TYPE is {edge_weight_type} but no coordinates found/parsed.")
        # Ensure all coordinates were actually found if this type is specified
        if len(coordinates) != dimension or any(c is None for c in coordinates): # Check for None if pre-allocated with Nones
             raise ValueError(f"Instance {final_name}: Incomplete coordinate data for {edge_weight_type}.")
        distance_matrix = _calculate_distances_from_coords(coordinates, dimension, edge_weight_type)
    elif edge_weight_type == "EXPLICIT":
        if not edge_weight_format:
            raise ValueError(f"Instance {final_name}: EDGE_WEIGHT_TYPE is EXPLICIT but EDGE_WEIGHT_FORMAT is missing.")
        if not edge_weights_lines:
            raise ValueError(f"Instance {final_name}: EDGE_WEIGHT_SECTION is missing for EXPLICIT type.")
        distance_matrix = _parse_explicit_edge_weights(edge_weights_lines, dimension, edge_weight_format)
    else:
        # Fallback or error if no specific edge weight calculation method is found
        # Some files might have NODE_COORD_SECTION for display and EXPLICIT weights.
        # If edge_weights_lines is populated, one could assume EXPLICIT even if type is missing, but it's risky.
        if edge_weights_lines and edge_weight_format : # Attempt to parse if data seems to be there
             print(f"Warning: Instance {final_name}: EDGE_WEIGHT_TYPE '{edge_weight_type}' is unknown/missing, but EDGE_WEIGHT_SECTION and FORMAT are present. Attempting EXPLICIT parse.")
             distance_matrix = _parse_explicit_edge_weights(edge_weights_lines, dimension, edge_weight_format)
        else:
            raise NotImplementedError(f"Instance {final_name}: EDGE_WEIGHT_TYPE '{edge_weight_type}' not supported or insufficient data to build distance matrix.")
    

    return CVRPInstance(
        name=final_name,
        dimension=dimension,
        capacity=capacity,
        distance_matrix=distance_matrix,
        demands=final_demands,
        depot=depot_0_based,
        coordinates=coordinates,
        num_vehicles_comment=num_vehicles_comment,
        edge_weight_type=edge_weight_type,
        edge_weight_format=edge_weight_format,
        full_comment="\n".join(comments_list) if comments_list else None
    )
