# cvrp_tripartite_solver/tests/common/test_cvrp_instance.py

import pytest
import os
import math
from common.cvrp_instance import load_cvrp_instance, CVRPInstance # Corrected import

# Helper function to create a dummy .vrp file content string
def create_dummy_vrp_content(
    name="dummy_euc",
    dimension=4,
    capacity=100,
    edge_weight_type="EUC_2D",
    coords=[(10,10), (10,20), (20,20), (20,10)], # Node 1 (depot), 2, 3, 4
    demands=[0, 10, 20, 15], # Demand for node 1, 2, 3, 4
    depot_node=1, # 1-based
    num_trucks_comment="No of trucks: 2"
) -> str:
    content = f"NAME : {name}\n"
    if num_trucks_comment:
        content += f"COMMENT : {num_trucks_comment}\n"
    content += f"TYPE : CVRP\n"
    content += f"DIMENSION : {dimension}\n"
    content += f"EDGE_WEIGHT_TYPE : {edge_weight_type}\n"
    content += f"CAPACITY : {capacity}\n"
    
    if edge_weight_type in ["EUC_2D", "CEIL_2D", "ATT"]:
        content += "NODE_COORD_SECTION\n"
        for i, (x,y) in enumerate(coords):
            content += f"{i+1} {x} {y}\n"
            
    content += "DEMAND_SECTION\n"
    # Ensure demands list matches dimension for the helper
    actual_demands = demands
    if len(demands) != dimension:
        # Pad or truncate demands if necessary for consistent dummy creation
        # This is a simplification for the helper; real files must be exact.
        actual_demands = (demands + [0] * dimension)[:dimension]


    for i, d in enumerate(actual_demands):
        content += f"{i+1} {d}\n"
        
    content += "DEPOT_SECTION\n"
    content += f"{depot_node}\n"
    content += "-1\n"
    content += "EOF\n"
    return content

def create_dummy_explicit_vrp_content(
    name="dummy_explicit",
    dimension=3,
    capacity=50,
    edge_weight_format="FULL_MATRIX",
    weights_flat=[0, 10, 20, 10, 0, 15, 20, 15, 0], # row-major
    demands=[0, 10, 12],
    depot_node=1
):
    content = f"NAME : {name}\n"
    content += f"TYPE : CVRP\n"
    content += f"DIMENSION : {dimension}\n"
    content += f"EDGE_WEIGHT_TYPE : EXPLICIT\n"
    content += f"EDGE_WEIGHT_FORMAT : {edge_weight_format}\n"
    content += f"CAPACITY : {capacity}\n"
    content += "EDGE_WEIGHT_SECTION\n"
    if edge_weight_format == "FULL_MATRIX":
        idx = 0
        for r in range(dimension):
            line_weights = []
            for c in range(dimension):
                line_weights.append(str(weights_flat[idx]))
                idx+=1
            content += " ".join(line_weights) + "\n"

    content += "DEMAND_SECTION\n"
    for i, d in enumerate(demands):
        content += f"{i+1} {d}\n"
        
    content += "DEPOT_SECTION\n"
    content += f"{depot_node}\n"
    content += "-1\n"
    content += "EOF\n"
    return content


# Test function for EUC_2D instance
def test_load_euc_2d_instance(tmp_path):
    """Tests loading a basic EUC_2D instance."""
    file_content = create_dummy_vrp_content(
        name="test_euc",
        dimension=4,
        capacity=100,
        coords=[(0,0), (0,10), (10,10), (10,0)], # Depot, N1, N2, N3
        demands=[0, 10, 20, 30], # Depot demand is 0
        depot_node=1,
        num_trucks_comment="Vehicles: 3"
    )
    p = tmp_path / "test_euc.vrp"
    p.write_text(file_content)

    instance = load_cvrp_instance(str(p))

    assert instance.name == "test_euc"
    assert instance.dimension == 4
    assert instance.capacity == 100
    assert instance.depot == 0  # 0-based
    assert instance.num_vehicles_comment == 3
    assert instance.edge_weight_type == "EUC_2D"
    
    expected_demands = [0, 10, 20, 30]
    assert instance.demands == expected_demands
    
    expected_coords = [(0.0,0.0), (0.0,10.0), (10.0,10.0), (10.0,0.0)]
    assert instance.coordinates == expected_coords
    
    dist_0_1 = int(round(math.sqrt((0-0)**2 + (0-10)**2))) # 10
    dist_0_2 = int(round(math.sqrt((0-10)**2 + (0-10)**2))) # 14
    dist_0_3 = int(round(math.sqrt((0-10)**2 + (0-0)**2))) # 10
    dist_1_2 = int(round(math.sqrt((0-10)**2 + (10-10)**2))) # 10
    dist_1_3 = int(round(math.sqrt((0-10)**2 + (10-0)**2))) # 14
    dist_2_3 = int(round(math.sqrt((10-10)**2 + (10-0)**2))) # 10

    expected_dist_matrix = [
        [0,      dist_0_1, dist_0_2, dist_0_3],
        [dist_0_1, 0,      dist_1_2, dist_1_3],
        [dist_0_2, dist_1_2, 0,      dist_2_3],
        [dist_0_3, dist_1_3, dist_2_3, 0     ]
    ]
    assert instance.distance_matrix == expected_dist_matrix
    assert "Vehicles: 3" in instance.full_comment

# Test function for EXPLICIT FULL_MATRIX instance
def test_load_explicit_full_matrix_instance(tmp_path):
    """Tests loading an EXPLICIT FULL_MATRIX instance."""
    file_content = create_dummy_explicit_vrp_content(
        name="test_explicit_full",
        dimension=3,
        capacity=75,
        weights_flat=[0, 12, 22, 12, 0, 15, 22, 15, 0],
        demands=[0, 25, 35],
        depot_node=1
    )
    p = tmp_path / "test_explicit.vrp"
    p.write_text(file_content)

    instance = load_cvrp_instance(str(p))

    assert instance.name == "test_explicit_full"
    assert instance.dimension == 3
    assert instance.capacity == 75
    assert instance.depot == 0 # 0-based
    assert instance.edge_weight_type == "EXPLICIT"
    assert instance.edge_weight_format == "FULL_MATRIX"
    
    expected_demands = [0, 25, 35]
    assert instance.demands == expected_demands
    
    expected_dist_matrix = [
        [0, 12, 22],
        [12, 0, 15],
        [22, 15, 0]
    ]
    assert instance.distance_matrix == expected_dist_matrix

# Test for parsing number of vehicles from comment "k = X"
def test_load_num_vehicles_k_format(tmp_path):
    file_content = create_dummy_vrp_content(num_trucks_comment="Optimal solution k = 5")
    p = tmp_path / "test_k_format.vrp"
    p.write_text(file_content)
    instance = load_cvrp_instance(str(p))
    assert instance.num_vehicles_comment == 5

# Test for missing dimension
def test_missing_dimension(tmp_path):
    file_content = create_dummy_vrp_content()
    # More robustly remove the DIMENSION line
    lines = file_content.splitlines()
    lines = [line for line in lines if not line.startswith("DIMENSION")]
    file_content_modified = "\n".join(lines)
    
    p = tmp_path / "missing_dim.vrp"
    p.write_text(file_content_modified)
    with pytest.raises(ValueError, match="DIMENSION not found"):
        load_cvrp_instance(str(p))

# Test for malformed demand section
def test_malformed_demand(tmp_path):
    # Default dummy content has demands: [0, 10, 20, 15] for nodes 1, 2, 3, 4
    # The line for node 2's demand is "2 10"
    # The line for node 3's demand is "3 20"
    # The line for node 4's demand is "4 15"
    file_content = create_dummy_vrp_content(
        demands=[0, 10, 999, 15] # Using a unique demand value for node 3
    )
    # Specifically target the line for node 3's demand to malform it.
    # Original line for node 3 would be "3 999"
    file_content_malformed = file_content.replace("3 999", "3") # Malform node 3's demand line

    p = tmp_path / "malformed_demand.vrp"
    p.write_text(file_content_malformed)
    with pytest.raises(ValueError, match="Malformed DEMAND entry"):
        load_cvrp_instance(str(p))

# Test for depot out of bounds
def test_depot_out_of_bounds(tmp_path):
    # dimension is 4 by default, so nodes are 1, 2, 3, 4
    file_content = create_dummy_vrp_content(depot_node=5) # Depot node 5 is out of bounds
    p = tmp_path / "depot_oob.vrp"
    p.write_text(file_content)
    with pytest.raises(ValueError, match="Depot node_idx 5 out of bounds"):
        load_cvrp_instance(str(p))

# Add more tests for:
# - Different EDGE_WEIGHT_FORMATS (LOWER_ROW, UPPER_ROW)
# - Different EDGE_WEIGHT_TYPES (CEIL_2D, ATT)
# - Files with missing optional sections (e.g., no NODE_COORD_SECTION when EXPLICIT)
# - Files with depot demand not being 0 (and how your __post_init__ handles it)
# - Edge cases in parsing logic
