import os
import random
import math

def generate_cvrp_instances(
    num_instances,
    output_dir,
    num_customers_min=5,
    num_customers_max=22,
    num_vehicles_min=2,
    num_vehicles_max=8,
    capacity_min=1000,
    capacity_max=2000,
    demand_min=50,
    demand_max=200
):
    """
    Generates multiple CVRP instances in .vrp format and saves them to a folder.

    Parameters
    ----------
    num_instances : int
        The number of .vrp files to generate.
    output_dir : str
        The path to the directory where the files will be saved.
    num_customers_min : int, optional
        Minimum number of customers per instance.
    num_customers_max : int, optional
        Maximum number of customers per instance (must be <= 22).
    num_vehicles_min : int, optional
        Minimum number of vehicles per instance.
    num_vehicles_max : int, optional
        Maximum number of vehicles per instance (must be <= 8).
    capacity_min : int, optional
        Minimum vehicle capacity.
    capacity_max : int, optional
        Maximum vehicle capacity.
    demand_min : int, optional
        Minimum customer demand.
    demand_max : int, optional
        Maximum customer demand.
    """
    if num_customers_max > 22 or num_vehicles_max > 8:
        print("Warning: The maximum number of customers or vehicles exceeds the specified limits.")
        print("Continuing with the provided limits.")

    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {num_instances} CVRP instances...")

    for i in range(num_instances):
        # 1. Generate random parameters
        n_customers = random.randint(num_customers_min, num_customers_max)
        n_nodes = n_customers + 1
        n_vehicles = random.randint(num_vehicles_min, num_vehicles_max)
        capacity = random.randint(capacity_min, capacity_max)

        coords = {0: (0, 0)}
        for customer_id in range(1, n_nodes):
            coords[customer_id] = (random.randint(1, 100), random.randint(1, 100))

        dist_matrix = [[0] * n_nodes for _ in range(n_nodes)]
        for i_node in range(n_nodes):
            for j_node in range(i_node + 1, n_nodes):
                x1, y1 = coords[i_node]
                x2, y2 = coords[j_node]
                distance = int(math.hypot(x1 - x2, y1 - y2) + 0.5)
                dist_matrix[i_node][j_node] = distance
                dist_matrix[j_node][i_node] = distance

        demands = {0: 0}
        total_demand = 0
        for customer_id in range(1, n_nodes):
            demands[customer_id] = random.randint(demand_min, demand_max)
            total_demand += demands[customer_id]
        
        if total_demand > n_vehicles * capacity:
            scale_factor = (n_vehicles * capacity) / total_demand * 0.8
            for customer_id in range(1, n_nodes):
                demands[customer_id] = int(demands[customer_id] * scale_factor)

        # 5. Build the .vrp file content
        file_content = [
            f"NAME : C-n{n_nodes}-k{n_vehicles}_inst{i+1}",
            f"COMMENT : (Generated instance for RL, number of trucks: {n_vehicles})",
            "TYPE : CVRP",
            f"DIMENSION : {n_nodes}",
            "EDGE_WEIGHT_TYPE : EXPLICIT",
            "EDGE_WEIGHT_FORMAT: LOWER_ROW",
            "DISPLAY_DATA_TYPE: NO_DISPLAY",
            f"CAPACITY : {capacity}",
            "EDGE_WEIGHT_SECTION"
        ]

        edge_weight_section_lines = []
        line_values = []
        for row in range(1, n_nodes):
            for col in range(row):
                line_values.append(str(dist_matrix[row][col]))
            
            line_str = "    " + "    ".join(line_values)
            edge_weight_section_lines.append(line_str)
            line_values = []
        
        file_content.extend(edge_weight_section_lines)

        file_content.append("DEMAND_SECTION")
        for customer_id in range(1, n_nodes):
            file_content.append(f"{customer_id} {demands[customer_id]}")
        file_content.append(f"{n_nodes} 0")

        file_content.append("DEPOT_SECTION")
        file_content.append(f"1")
        file_content.append("-1")
        #file_content.append("EOF")

        file_name = f"C-n{n_nodes}-k{n_vehicles}_inst{i+1}.vrp"
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "w") as f:
            f.write("\n".join(file_content))

        print(f"Generated {file_name}")

    print("Generation complete.")

# Example Usage:
if __name__ == "__main__":
    generate_cvrp_instances(
        num_instances=1,
        output_dir="quantum_data",
        num_customers_min=2,
        num_customers_max=5,
        num_vehicles_min=2,
        num_vehicles_max=3
    )



