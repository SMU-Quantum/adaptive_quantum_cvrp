
# import argparse
# import os
# import sys
# import time
# import json
# from datetime import datetime

# # Ensure the src directory is in the Python path
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# sys.path.insert(0, project_root)
# sys.path.insert(0, os.path.join(project_root, "src"))


# from common.cvrp_instance import load_cvrp_instance, CVRPInstance
# from common.cvrp_solution import CVRPSolution
# from alm.alm_optimizer import AlmOptimizer
# from common.cvrp_evaluator import check_solution_feasibility 

# def format_route_for_display(route: list[int], depot_node: int) -> str:
#     """Formats a single route to display only customer nodes."""
#     # Filter out depot nodes from the main part of the route for display
#     # Assumes route starts and ends with depot, e.g., [0, 1, 2, 0]
#     # We want to display "1 2"
#     if len(route) <= 2: # Empty route like [0,0] or invalid
#         return ""
    
#     # Display 1-based indexing if that's the convention for the output format
#     # CVRPLIB instances are 1-based, our internal representation is 0-based.
#     # For display, let's show 1-based customer numbers.
#     return " ".join(str(node_idx + 1) for node_idx in route if node_idx != depot_node)


# def run_experiment(instance_path: str, output_dir: str, alm_params: dict):
#     """
#     Runs a single ALM experiment on a given CVRP instance.

#     Args:
#         instance_path: Path to the .vrp instance file.
#         output_dir: Directory to save results.
#         alm_params: Dictionary of parameters for AlmOptimizer.
#     """
#     print(f"--- Running Experiment for Instance: {instance_path} ---")

#     # 1. Load Instance
#     try:
#         print(f"Loading instance from: {instance_path}")
#         instance = load_cvrp_instance(instance_path)
#         print(f"Instance '{instance.name}' loaded: Dimension={instance.dimension}, Capacity={instance.capacity}, Depot={instance.depot}")
#     except Exception as e:
#         print(f"Error loading instance {instance_path}: {e}")
#         return

#     # 2. Initialize AlmOptimizer
#     print(f"\nInitializing AlmOptimizer with parameters: {alm_params}")
#     # Pass the verbose flag from alm_params if it exists, otherwise default to 0
#     optimizer_verbose = alm_params.pop("verbose", 0) # Remove verbose so it's not passed as unknown kwarg
#     optimizer = AlmOptimizer(instance=instance, verbose=optimizer_verbose, **alm_params)


#     # 3. Solve
#     print("\nStarting ALM optimization...")
#     start_time = time.time()
#     best_solution = optimizer.solve()
#     end_time = time.time()
#     solve_duration = end_time - start_time
#     print(f"ALM optimization finished in {solve_duration:.2f} seconds.")

#     # 4. Print Summary and Save Results
#     instance_name_for_file = instance.name.replace(" ", "_").replace(":", "_").replace("/", "_")
#     instance_results_dir = os.path.join(output_dir, instance_name_for_file)
#     os.makedirs(instance_results_dir, exist_ok=True)

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     summary_lines = []
#     summary_lines.append(f"--- Experiment Summary for {instance.name} ({timestamp}) ---")
#     summary_lines.append(f"Instance Path: {instance_path}")
#     summary_lines.append(f"Solve Duration: {solve_duration:.2f} seconds")
#     summary_lines.append(f"ALM Parameters: {alm_params}") # alm_params no longer contains 'verbose'

#     solution_file_content = [] # For saving to solution file
#     solution_file_content.append(f"Instance: {instance.name}")


#     if best_solution:
#         summary_lines.append(f"\nBest Solution Found:")
#         summary_lines.append(f"  Is Feasible: {best_solution.is_feasible}")
#         if not best_solution.is_feasible:
#             summary_lines.append(f"  Feasibility Details: {best_solution.feasibility_details}")
        
#         # Add to solution file content
#         solution_file_content.append(f"Is Feasible: {best_solution.is_feasible}")
#         if best_solution.feasibility_details:
#              solution_file_content.append(f"Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}")

#         # Format and print routes as requested
#         summary_lines.append("\nRoutes (customers shown with 1-based indexing):")
#         solution_file_content.append("\nRoutes (customers shown with 1-based indexing, raw 0-based in brackets):")
#         for i, route in enumerate(best_solution.routes):
#             # Customers are typically 1-indexed in VRP problem descriptions.
#             # Our internal representation is 0-indexed. Depot is instance.depot.
#             # We filter out the depot and add 1 to customer indices for display.
            
#             # Filter out depot nodes for display string
#             customer_nodes_in_route_str = " ".join(
#                 str(node_idx + 1) for node_idx in route if node_idx != instance.depot
#             )
#             # Handle case where a route might be just [depot, depot] if no customers served by it
#             if not customer_nodes_in_route_str and len(route) == 2 and route[0] == instance.depot and route[1] == instance.depot:
#                  customer_nodes_in_route_str = "(empty)"


#             summary_lines.append(f"  Route #{i+1}: {customer_nodes_in_route_str}")
#             solution_file_content.append(f"  Route #{i+1}: {customer_nodes_in_route_str}  (Raw: {route})")
        
#         summary_lines.append(f"Cost {best_solution.total_cost:.2f}") # Using .2f for consistency
#         solution_file_content.append(f"\nCost {best_solution.total_cost:.0f}") # As per user example, integer cost

#         # Save solution routes
#         solution_file_path = os.path.join(instance_results_dir, f"solution_{timestamp}.txt")
#         with open(solution_file_path, 'w') as f:
#             f.write("\n".join(solution_file_content))
#         summary_lines.append(f"  Solution details saved to: {solution_file_path}")

#     else:
#         summary_lines.append("\nNo feasible solution found by ALM.")
#         solution_file_content.append("\nNo feasible solution found by ALM.")
#         # Save a minimal solution file even if no solution
#         solution_file_path = os.path.join(instance_results_dir, f"solution_NO_SOLUTION_{timestamp}.txt")
#         with open(solution_file_path, 'w') as f:
#             f.write("\n".join(solution_file_content))


#     # Save iteration log
#     log_file_path = os.path.join(instance_results_dir, f"iteration_log_{timestamp}.json")
#     with open(log_file_path, 'w') as f:
#         json.dump(optimizer.iteration_log, f, indent=2)
#     summary_lines.append(f"  Iteration log saved to: {log_file_path}")
    
#     # Print summary to console
#     print("\n" + "\n".join(summary_lines))

#     # Save summary to a file
#     summary_file_path = os.path.join(instance_results_dir, f"summary_{timestamp}.txt")
#     with open(summary_file_path, 'w') as f:
#         f.write("\n".join(summary_lines))
#     print(f"Summary report saved to: {summary_file_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run Augmented Lagrangian Method for CVRP.")
#     parser.add_argument(
#         "instance_file",
#         type=str,
#         help="Path to the CVRP instance file (.vrp)."
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default=os.path.join(project_root, "results", "alm"),
#         help="Directory to save experiment results."
#     )
#     parser.add_argument("--initial_rho", type=float, default=1.0, help="Initial penalty rate (rho).")
#     parser.add_argument("--rho_increase_factor", type=float, default=1.1, help="Penalty increase factor.")
#     parser.add_argument("--max_rho", type=float, default=1000.0, help="Maximum penalty rate.")
#     parser.add_argument("--max_alm_iter", type=int, default=100, help="Maximum ALM iterations.")
#     parser.add_argument("--conv_tolerance", type=float, default=1e-3, help="Convergence tolerance for constraint violations.")
#     parser.add_argument("--subproblem_max_vehicles", type=int, default=None, help="Max vehicles/routes for subproblem.")
#     parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2], help="Verbosity level for AlmOptimizer (0=minimal, 1=medium, 2=detailed).")


#     args = parser.parse_args()

#     alm_parameters = {
#         "initial_penalty_rate": args.initial_rho,
#         "penalty_increase_factor": args.rho_increase_factor,
#         "max_penalty_rate": args.max_rho,
#         "max_alm_iterations": args.max_alm_iter,
#         "convergence_tolerance": args.conv_tolerance,
#         "subproblem_max_vehicles": args.subproblem_max_vehicles,
#         "verbose": args.verbose # Pass verbose level to AlmOptimizer
#     }

#     os.makedirs(args.output_dir, exist_ok=True)
#     run_experiment(args.instance_file, args.output_dir, alm_parameters)

# cvrp_tripartite_solver/experiments/run_alm_experiment.py

import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import List # Added for type hinting

# Ensure the src directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


from common.cvrp_instance import load_cvrp_instance, CVRPInstance
from common.cvrp_solution import CVRPSolution
from alm.alm_optimizer import AlmOptimizer
from common.cvrp_evaluator import check_solution_feasibility 

# format_route_for_display function is not strictly needed as logic is embedded below
# def format_route_for_display(route: List[int], depot_node: int) -> str:
# ...

def run_experiment(instance_path: str, output_dir: str, alm_params: dict):
    """
    Runs a single ALM experiment on a given CVRP instance.

    Args:
        instance_path: Path to the .vrp instance file.
        output_dir: Directory to save results.
        alm_params: Dictionary of parameters for AlmOptimizer.
    """
    print(f"--- Running Experiment for Instance: {instance_path} ---")

    # 1. Load Instance
    try:
        print(f"Loading instance from: {instance_path}")
        instance = load_cvrp_instance(instance_path)
        print(f"Instance '{instance.name}' loaded: Dimension={instance.dimension}, Capacity={instance.capacity}, Depot={instance.depot}")
    except Exception as e:
        print(f"Error loading instance {instance_path}: {e}")
        return

    # 2. Initialize AlmOptimizer
    # Extract verbose for AlmOptimizer, keep other params for summary
    optimizer_verbose = alm_params.get("verbose", 0) 
    actual_alm_params_for_optimizer = {k: v for k, v in alm_params.items() if k != "verbose"}
    
    print(f"\nInitializing AlmOptimizer with parameters: {actual_alm_params_for_optimizer}, verbose={optimizer_verbose}")
    optimizer = AlmOptimizer(instance=instance, verbose=optimizer_verbose, **actual_alm_params_for_optimizer)


    # 3. Solve
    print("\nStarting ALM optimization...")
    start_time = time.time()
    best_solution = optimizer.solve()
    end_time = time.time()
    solve_duration = end_time - start_time
    print(f"ALM optimization finished in {solve_duration:.2f} seconds.")

    # 4. Print Summary and Save Results
    # Sanitize instance name for file/directory creation
    sanitized_instance_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in instance.name)
    instance_name_for_file = sanitized_instance_name if sanitized_instance_name else "unnamed_instance"

    instance_results_dir = os.path.join(output_dir, instance_name_for_file)
    os.makedirs(instance_results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_lines = []
    summary_lines.append(f"--- Experiment Summary for {instance.name} ({timestamp}) ---")
    summary_lines.append(f"Instance Path: {instance_path}")
    summary_lines.append(f"Solve Duration: {solve_duration:.2f} seconds")
    summary_lines.append(f"ALM Parameters Used: {alm_params}") 

    solution_file_content = [] 
    solution_file_content.append(f"Instance: {instance.name}")


    if best_solution:
        summary_lines.append(f"\nBest Solution Found:")
        summary_lines.append(f"  Is Feasible: {best_solution.is_feasible}")
        if not best_solution.is_feasible and best_solution.feasibility_details:
            summary_lines.append(f"  Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}")
        
        solution_file_content.append(f"Is Feasible: {best_solution.is_feasible}")
        if best_solution.feasibility_details:
             solution_file_content.append(f"Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}")

        summary_lines.append("\nRoutes (customers shown with 1-based indexing as per typical VRP output):")
        solution_file_content.append("\nRoutes (customers shown with 1-based indexing, raw 0-based internal in brackets):")
        
        for i, route in enumerate(best_solution.routes):
            # Internal route is 0-indexed, depot is instance.depot (e.g., 0)
            # Displayed customer IDs should be their original 1-based IDs from the file.
            # If depot was 1 in file, internal is 0. Internal customer 1 was file node 2. Display as 2.
            # So, display internal_node_id + 1 (since depot is 0 internally).
            
            customer_nodes_in_route_display = []
            for node_idx in route:
                if node_idx != instance.depot:
                    # Assuming customer node N in file becomes internal node N-1 (if depot is 1 in file)
                    # or N if depot is 0 in file and customers start from 1.
                    # Given our loader makes internal depot 0, and customers are then 1...dim-1
                    # Displaying node_idx + 1 gives 1-based indexing for these internal indices.
                    customer_nodes_in_route_display.append(str(node_idx + 1)) 
            
            route_str_display = " ".join(customer_nodes_in_route_display)
            if not route_str_display and len(route) == 2 and route[0] == instance.depot and route[1] == instance.depot:
                 route_str_display = "(empty)" # For a vehicle that only goes depot-depot

            summary_lines.append(f"  Route #{i+1}: {route_str_display}")
            solution_file_content.append(f"  Route #{i+1}: {route_str_display}  (Raw internal 0-indexed: {route})")
        
        summary_lines.append(f"Cost {best_solution.total_cost:.2f}") 
        solution_file_content.append(f"\nCost {best_solution.total_cost:.0f}") 

        # Add final penalty (rho) values to summary and solution file
        final_rho_info_summary = ["\nFinal Penalty Parameters (rho):"]
        final_rho_info_file = ["\nFinal Penalty Parameters (rho):"]

        if optimizer.customer_indices and optimizer.rhos and len(optimizer.customer_indices) == len(optimizer.rhos):
            for i, internal_cust_idx in enumerate(optimizer.customer_indices):
                # Display original 1-based customer ID for rho
                display_cust_id = internal_cust_idx + 1 
                rho_val_str = f"  Customer {display_cust_id} (internal 0-idx {internal_cust_idx}): {optimizer.rhos[i]:.2f}"
                final_rho_info_summary.append(rho_val_str)
                final_rho_info_file.append(rho_val_str)
        else:
            no_rho_msg = "  No customer penalty (rho) values to display (or mismatch in lengths)."
            final_rho_info_summary.append(no_rho_msg)
            final_rho_info_file.append(no_rho_msg)
        
        # 1) Add ρ (visit-constraint multipliers)
        summary_lines.extend(final_rho_info_summary)
        solution_file_content.extend(final_rho_info_file)

        # 2) Add μ (capacity-constraint multipliers)
        summary_lines.append("\nFinal Capacity Multipliers (μ):")
        solution_file_content.append("\nFinal Capacity Multipliers (μ):")
        for k, mu in enumerate(optimizer.mus_capacity):
            summary_lines.append(f"  Vehicle {k+1}: {mu:.2f}")
            solution_file_content.append(f"  Vehicle {k+1}: {mu:.2f}")


        solution_file_path = os.path.join(instance_results_dir, f"solution_{timestamp}.txt")
        with open(solution_file_path, 'w', encoding="utf-8") as f:
            f.write("\n".join(solution_file_content))
        summary_lines.append(f"  Solution details (incl. rho) saved to: {solution_file_path}")

    else:
        summary_lines.append("\nNo feasible solution found by ALM.")
        solution_file_content.append("\nNo feasible solution found by ALM.")
        solution_file_path = os.path.join(instance_results_dir, f"solution_NO_SOLUTION_{timestamp}.txt")
        with open(solution_file_path, 'w', encoding="utf-8") as f:
            f.write("\n".join(solution_file_content))


    log_file_path = os.path.join(instance_results_dir, f"iteration_log_{timestamp}.json")
    with open(log_file_path, 'w', encoding="utf-8") as f:
        json.dump(optimizer.iteration_log, f, indent=2)
    summary_lines.append(f"  Iteration log saved to: {log_file_path}")
    
    print("\n" + "\n".join(summary_lines))

    summary_file_path = os.path.join(instance_results_dir, f"summary_{timestamp}.txt")
    with open(summary_file_path, 'w', encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"Summary report saved to: {summary_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Augmented Lagrangian Method for CVRP.")
    parser.add_argument(
        "instance_file",
        type=str,
        help="Path to the CVRP instance file (.vrp)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "results", "alm"),
        help="Directory to save experiment results."
    )
    parser.add_argument("--initial_rho", type=float, default=1.0, help="Initial penalty rate (rho).")
    parser.add_argument("--rho_increase_factor", type=float, default=1.1, help="Penalty increase factor.")
    parser.add_argument("--max_rho", type=float, default=1000.0, help="Maximum penalty rate.")
    parser.add_argument("--max_alm_iter", type=int, default=100, help="Maximum ALM iterations.")
    parser.add_argument("--conv_tolerance", type=float, default=1e-3, help="Convergence tolerance for constraint violations.")
    parser.add_argument("--subproblem_max_vehicles", type=int, default=None, help="Max vehicles/routes for subproblem.")
    parser.add_argument("--verbose", type=int, default=0, choices=[0,1,2], help="Verbosity level for AlmOptimizer (0=minimal, 1=medium, 2=detailed).")


    args = parser.parse_args()

    alm_parameters = {
        "initial_penalty_rate": args.initial_rho,
        "penalty_increase_factor": args.rho_increase_factor,
        "max_penalty_rate": args.max_rho,
        "max_alm_iterations": args.max_alm_iter,
        "convergence_tolerance": args.conv_tolerance,
        "subproblem_max_vehicles": args.subproblem_max_vehicles,
        "verbose": args.verbose 
    }

    os.makedirs(args.output_dir, exist_ok=True)
    run_experiment(args.instance_file, args.output_dir, alm_parameters)



# python experiments/run_alm_experiment.py data/cvrplib_instances/E-n13-k4.vrp
# python experiments/run_alm_experiment.py data/cvrplib_instances/
# python experiments/run_alm_experiment.py data/cvrplib_instances/ --max_alm_iter 50 --verbose 1