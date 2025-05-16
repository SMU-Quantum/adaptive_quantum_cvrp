# cvrp_tripartite_solver/experiments/run_alm_experiment.py

import argparse
import os
import sys
import time
import json
import glob # For finding files matching a pattern
from datetime import datetime
from typing import List 

# Ensure the src directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


from common.cvrp_instance import load_cvrp_instance, CVRPInstance
from common.cvrp_solution import CVRPSolution
from alm.alm_optimizer import AlmOptimizer
from common.cvrp_evaluator import check_solution_feasibility 

def run_single_experiment(instance_path: str, output_dir: str, alm_params: dict, overall_summary_list: List[str]):
    """
    Runs a single ALM experiment on a given CVRP instance and appends to summary.
    """
    print(f"\n--- Running Experiment for Instance: {instance_path} ---")
    summary_lines = [] # For this specific instance's console summary

    # 1. Load Instance
    try:
        # print(f"Loading instance from: {instance_path}") # Already printed by main loop
        instance = load_cvrp_instance(instance_path)
        summary_lines.append(f"Instance '{instance.name}' loaded: Dimension={instance.dimension}, Capacity={instance.capacity}, Depot={instance.depot}")
        print(f"Instance '{instance.name}' loaded: Dimension={instance.dimension}, Capacity={instance.capacity}, Depot={instance.depot}")
    except Exception as e:
        error_msg = f"Error loading instance {instance_path}: {e}"
        print(error_msg)
        overall_summary_list.append(f"Instance: {instance_path}\n  Status: FAILED_TO_LOAD\n  Error: {e}\n")
        return

    # 2. Initialize AlmOptimizer
    optimizer_verbose = alm_params.get("verbose", 0) 
    actual_alm_params_for_optimizer = {k: v for k, v in alm_params.items() if k != "verbose"}
    
    summary_lines.append(f"Initializing AlmOptimizer with parameters: {actual_alm_params_for_optimizer}, verbose={optimizer_verbose}")
    optimizer = AlmOptimizer(instance=instance, verbose=optimizer_verbose, **actual_alm_params_for_optimizer)

    # 3. Solve
    summary_lines.append("Starting ALM optimization...")
    print("Starting ALM optimization...") # Also print to console during run
    start_time = time.time()
    best_solution = optimizer.solve()
    end_time = time.time()
    solve_duration = end_time - start_time
    summary_lines.append(f"ALM optimization finished in {solve_duration:.2f} seconds.")
    print(f"ALM optimization finished in {solve_duration:.2f} seconds.")


    # 4. Prepare for Saving Results
    sanitized_instance_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in instance.name)
    instance_name_for_file = sanitized_instance_name if sanitized_instance_name else "unnamed_instance"
    
    # Create a subdirectory for this specific instance's results
    # The main output_dir (e.g., results/alm) is created by the main function
    instance_specific_results_dir = os.path.join(output_dir, instance_name_for_file)
    os.makedirs(instance_specific_results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Add to overall summary for batch run
    overall_summary_list.append(f"Instance: {instance.name} ({os.path.basename(instance_path)})")
    overall_summary_list.append(f"  Solve Duration: {solve_duration:.2f}s")

    solution_file_content = [] 
    solution_file_content.append(f"Instance: {instance.name}")
    solution_file_content.append(f"File Path: {instance_path}")
    solution_file_content.append(f"Solve Duration: {solve_duration:.2f} seconds")
    solution_file_content.append(f"ALM Parameters Used: {alm_params}")

    if best_solution:
        summary_lines.append(f"Best Solution Found:")
        summary_lines.append(f"  Is Feasible: {best_solution.is_feasible}")
        overall_summary_list.append(f"  Cost: {best_solution.total_cost:.2f}, Feasible: {best_solution.is_feasible}, Vehicles: {best_solution.num_vehicles_used}")

        if not best_solution.is_feasible and best_solution.feasibility_details:
            summary_lines.append(f"  Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}")
        
        solution_file_content.append(f"Is Feasible: {best_solution.is_feasible}")
        if best_solution.feasibility_details:
             solution_file_content.append(f"Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}")

        summary_lines.append("Routes (customers shown with 1-based indexing as per typical VRP output):")
        solution_file_content.append("\nRoutes (customers shown with 1-based indexing, raw 0-based internal in brackets):")
        
        for i, route in enumerate(best_solution.routes):
            customer_nodes_in_route_display = []
            for node_idx in route:
                if node_idx != instance.depot:
                    customer_nodes_in_route_display.append(str(node_idx + 1)) 
            
            route_str_display = " ".join(customer_nodes_in_route_display)
            if not route_str_display and len(route) == 2 and route[0] == instance.depot and route[1] == instance.depot:
                 route_str_display = "(empty)"

            summary_lines.append(f"  Route #{i+1}: {route_str_display}")
            solution_file_content.append(f"  Route #{i+1}: {route_str_display}  (Raw internal 0-indexed: {route})")
        
        summary_lines.append(f"Cost {best_solution.total_cost:.2f}") 
        solution_file_content.append(f"\nCost {best_solution.total_cost:.0f}") 

        final_rho_info_summary = ["\nFinal Penalty Parameters (rho):"]
        final_rho_info_file = ["\nFinal Penalty Parameters (rho):"]
        if optimizer.customer_indices and optimizer.rhos and len(optimizer.customer_indices) == len(optimizer.rhos):
            for i, internal_cust_idx in enumerate(optimizer.customer_indices):
                display_cust_id = internal_cust_idx + 1 
                rho_val_str = f"  Customer {display_cust_id} (internal 0-idx {internal_cust_idx}): {optimizer.rhos[i]:.2f}"
                final_rho_info_summary.append(rho_val_str)
                final_rho_info_file.append(rho_val_str)
        else:
            no_rho_msg = "  No customer penalty (rho) values to display."
            final_rho_info_summary.append(no_rho_msg)
            final_rho_info_file.append(no_rho_msg)
        
        summary_lines.extend(final_rho_info_summary)
        solution_file_content.extend(final_rho_info_file)

        solution_file_path = os.path.join(instance_specific_results_dir, f"solution_{timestamp}.txt")
        with open(solution_file_path, 'w') as f:
            f.write("\n".join(solution_file_content))
        summary_lines.append(f"  Solution details (incl. rho) saved to: {solution_file_path}")

    else:
        summary_lines.append("No feasible solution found by ALM.")
        overall_summary_list.append(f"  Status: NO_FEASIBLE_SOLUTION_FOUND")
        solution_file_content.append("\nNo feasible solution found by ALM.")
        solution_file_path = os.path.join(instance_specific_results_dir, f"solution_NO_SOLUTION_{timestamp}.txt")
        with open(solution_file_path, 'w') as f:
            f.write("\n".join(solution_file_content))

    log_file_path = os.path.join(instance_specific_results_dir, f"iteration_log_{timestamp}.json")
    with open(log_file_path, 'w') as f:
        json.dump(optimizer.iteration_log, f, indent=2)
    summary_lines.append(f"  Iteration log saved to: {log_file_path}")
    
    # Print per-instance summary to console (if verbose for optimizer is low)
    if optimizer_verbose < 2: # Avoid double printing if ALMOptimizer is already very verbose
        print("\n" + "\n".join(summary_lines))

    # Save per-instance summary to its own file
    instance_summary_file_path = os.path.join(instance_specific_results_dir, f"summary_{timestamp}.txt")
    with open(instance_summary_file_path, 'w') as f:
        f.write(f"--- Experiment Summary for {instance.name} ({timestamp}) ---\n")
        f.write(f"Instance Path: {instance_path}\n")
        f.write(f"Solve Duration: {solve_duration:.2f} seconds\n")
        f.write(f"ALM Parameters Used: {alm_params}\n") # These are the ones passed to run_experiment
        if best_solution:
            f.write(f"\nBest Solution Found:\n")
            f.write(f"  Is Feasible: {best_solution.is_feasible}\n")
            if not best_solution.is_feasible and best_solution.feasibility_details:
                 f.write(f"  Feasibility Details: {json.dumps(best_solution.feasibility_details, indent=2)}\n")
            f.write("\nRoutes (customers shown with 1-based indexing as per typical VRP output):\n")
            for i, route in enumerate(best_solution.routes):
                customer_nodes_in_route_display = []
                for node_idx in route:
                    if node_idx != instance.depot:
                        customer_nodes_in_route_display.append(str(node_idx + 1)) 
                route_str_display = " ".join(customer_nodes_in_route_display)
                if not route_str_display and len(route) == 2 and route[0] == instance.depot and route[1] == instance.depot:
                    route_str_display = "(empty)"
                f.write(f"  Route #{i+1}: {route_str_display}\n")
            f.write(f"Cost {best_solution.total_cost:.2f}\n")
            f.write("\n".join(final_rho_info_file)) # Add rho info
        else:
            f.write("\nNo feasible solution found by ALM.\n")
        f.write(f"\nIteration log saved to: {log_file_path}\n")
        f.write(f"Solution details saved to: {solution_file_path}\n")

    print(f"Per-instance summary report saved to: {instance_summary_file_path}")
    overall_summary_list.append("-" * 30) # Separator for overall summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Augmented Lagrangian Method for CVRP on multiple instances.")
    parser.add_argument(
        "instance_dir_or_file", # Can now be a directory or a single file
        type=str,
        help="Path to the CVRP instance file (.vrp) or a directory containing .vrp files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(project_root, "results", "alm"),
        help="Base directory to save experiment results."
    )
    # Add arguments for ALM parameters (same as before)
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

    # Ensure base output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine if input is a file or directory
    instance_files_to_process: List[str] = []
    if os.path.isfile(args.instance_dir_or_file) and args.instance_dir_or_file.lower().endswith(".vrp"):
        instance_files_to_process.append(args.instance_dir_or_file)
    elif os.path.isdir(args.instance_dir_or_file):
        # Find all .vrp files in the directory (non-recursive)
        search_pattern = os.path.join(args.instance_dir_or_file, "*.vrp")
        instance_files_to_process = glob.glob(search_pattern)
        if not instance_files_to_process:
             search_pattern_lower = os.path.join(args.instance_dir_or_file, "*.VRP") # Try uppercase
             instance_files_to_process = glob.glob(search_pattern_lower)

        if not instance_files_to_process:
            print(f"No .vrp files found in directory: {args.instance_dir_or_file}")
            sys.exit(1)
        instance_files_to_process.sort() # Process in a consistent order
    else:
        print(f"Error: Path '{args.instance_dir_or_file}' is not a valid .vrp file or directory.")
        sys.exit(1)

    print(f"Found {len(instance_files_to_process)} instance(s) to process.")
    
    batch_run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_summary_data: List[str] = []
    overall_summary_data.append(f"=== ALM Batch Run Summary - {batch_run_timestamp} ===")
    overall_summary_data.append(f"Processed Directory/File: {args.instance_dir_or_file}")
    overall_summary_data.append(f"ALM Parameters Used for all instances: {alm_parameters}\n")


    for instance_file_path in instance_files_to_process:
        run_single_experiment(instance_file_path, args.output_dir, alm_params=alm_parameters.copy(), overall_summary_list=overall_summary_data)
    
    # Save the overall batch summary
    overall_summary_filename = f"batch_summary_{batch_run_timestamp}.txt"
    overall_summary_filepath = os.path.join(args.output_dir, overall_summary_filename)
    with open(overall_summary_filepath, 'w') as f:
        f.write("\n".join(overall_summary_data))
    print(f"\n\n=== Overall Batch Summary saved to: {overall_summary_filepath} ===")

