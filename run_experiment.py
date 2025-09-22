# run_experiment.py

"""
It loads a configuration, sets up the required components 
(like the solver and environment) based on that configuration, and runs the specified experiment.
"""

import argparse
import logging
from pathlib import Path
import json
import numpy as np

# (Imports remain the same)
from src.adaptive_quantum_cvrp.common.cvrp_instance import CVRPInstance
from src.adaptive_quantum_cvrp.alm.optimizer import ALMOptimizer
from src.adaptive_quantum_cvrp.alm.classical_solver import ClassicalSolver
from src.adaptive_quantum_cvrp.quantum.solver import QuantumSolver
from src.adaptive_quantum_cvrp.rl.agent import SACAgent
from src.adaptive_quantum_cvrp.rl.environment import ALMPenaltyEnv
from src.adaptive_quantum_cvrp.utils.config_loader import load_config
from src.adaptive_quantum_cvrp.utils.logging_config import setup_logging

def run_vanilla_alm(config, instance, solver, output_dir):
    """Runs a standard ALM optimization without RL for a single instance."""
    logging.info(f"Starting Vanilla ALM experiment for instance: {instance.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    alm_config = config["alm"]
    optimizer = ALMOptimizer(instance, alm_config["max_iterations"], solver)
    
    results = optimizer.solve(alm_config["initial_penalty_mu"])
    
    logging.info(f"Best feasible cost for {instance.name}: {results['best_cost']}")
    
    solution_path = output_dir / "solution.json"
    with open(solution_path, "w") as f:
        json.dump({
            "instance": instance.name,
            "cost": results['best_cost'],
            "routes": results['best_solution'].routes if results['best_solution'] else "None"
        }, f, indent=2)
    logging.info(f"Solution for {instance.name} saved to {solution_path}")

def run_rl_alm(config, instance, solver, output_dir):
    """Runs an RL-driven ALM optimization for a single instance."""
    logging.info(f"Starting RL-ALM training for instance: {instance.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rl_config = config["rl"]
    
    env = ALMPenaltyEnv(instance, solver, rl_config["max_alm_steps"])
    agent = SACAgent(
        input_dims=env.observation_space.shape[0],
        n_actions=env.action_space.shape[0],
        action_space=env.action_space,
        lr=rl_config["learning_rate"],
        gamma=rl_config["gamma"],
        tau=rl_config["tau"]
    )

    scores = []
    # Variables to track the best solution ---
    best_instance_cost = float('inf')
    best_instance_solution = None

    for i in range(rl_config["num_episodes"]):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if the environment found a new best solution ---
            if "solution" in info:
                if info["cost"] < best_instance_cost:
                    best_instance_cost = info["cost"]
                    best_instance_solution = info["solution"]
                    logging.info(f"New best feasible solution found for {instance.name} with cost: {best_instance_cost:.2f}")

            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, done)
            agent.learn(rl_config["batch_size"])
            score += reward
            obs = next_obs
        
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        logging.info(f"Episode {i+1} | Score: {score:.2f} | Avg Score: {avg_score:.2f}")

    # Save the best found solution to a JSON file ---
    logging.info(f"RL training for {instance.name} complete. Best cost found: {best_instance_cost}")
    solution_path = output_dir / "solution.json"
    with open(solution_path, "w") as f:
        json.dump({
            "instance": instance.name,
            "cost": best_instance_cost,
            "routes": best_instance_solution.routes if best_instance_solution else "None"
        }, f, indent=2)
    logging.info(f"Best solution for {instance.name} saved to {solution_path}")
    



def run(config_path: Path):
    """
    Main function to run an experiment for one or more instances.
    """
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return

    base_output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(base_output_dir, config["experiment"]["log_level"])
    logging.info(f"Loaded configuration from: {config_path}")

    #  Handle both single instance_path and batch instance_folder ---
    instance_paths = []
    if "instance_folder" in config["data"]:
        instance_folder = Path(config["data"]["instance_folder"])
        if not instance_folder.is_dir():
            logging.error(f"Instance folder not found: {instance_folder}")
            return
        instance_paths = sorted(list(instance_folder.glob("*.vrp")))
        logging.info(f"Found {len(instance_paths)} instances to process in '{instance_folder}'")
    elif "instance_path" in config["data"]:
        instance_path = Path(config["data"]["instance_path"])
        if not instance_path.is_file():
            logging.error(f"Instance file not found: {instance_path}")
            return
        instance_paths.append(instance_path)
        logging.info(f"Found 1 instance to process: '{instance_path}'")
    else:
        raise ValueError("Config file must specify 'instance_folder' or 'instance_path' under the 'data' key.")
    

    # Factory for Subproblem Solver (created once)
    solver_type = config["solver"]["type"]
    if solver_type == "classical":
        solver = ClassicalSolver()
        logging.info("Using Classical Subproblem Solver for all instances.")
    elif solver_type == "quantum":
        solver = QuantumSolver(**config["solver"]["quantum_options"])
        logging.info("Using Quantum Subproblem Solver for all instances.")
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")

    # Loop through each instance and solve it
    for instance_path in instance_paths:
        try:
            instance_name = instance_path.stem
            logging.info(f"--- Processing instance: {instance_name} ---")
            
            # Create a dedicated output directory for this instance's results
            instance_output_dir = base_output_dir / instance_name
            
            instance = CVRPInstance.from_file(instance_path)

            exp_type = config["experiment"]["type"]
            if exp_type == "vanilla_alm":
                run_vanilla_alm(config, instance, solver, instance_output_dir)
            elif exp_type == "rl_alm":
                run_rl_alm(config, instance, solver, instance_output_dir)
            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")

        except Exception as e:
            logging.error(f"Failed to process instance {instance_path.name}. Error: {e}", exc_info=True)
            continue

    logging.info("--- All processing complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CVRP experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the YAML configuration file."
    )
    args = parser.parse_args()
    run(args.config)