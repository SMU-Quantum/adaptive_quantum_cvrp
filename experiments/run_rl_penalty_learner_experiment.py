# cvrp_tripartite_solver/experiments/run_rl_penalty_learner_experiment.py

import argparse
import os
import random
import time
import json # For saving results
from collections import deque
from pathlib import Path # For easier path manipulation
from typing import Optional, List, Dict, Any, Tuple # Added Optional, List, Dict, Any, Tuple

import numpy as np
import torch

from rl_classical.alm_penalty_env import AlmPenaltyEnv # N_INSTANCE_FEATURES is defined here
from rl_classical.penalty_agent import SACAgent, DEFAULT_SAC_HYPERPARAMS
from common.cvrp_instance import load_cvrp_instance # For filtering by dimension

def get_instance_files_and_filter(
    path: str, 
    max_dimension: Optional[int] = None
) -> List[str]: # Changed to List from list for consistency with typing
    """
    Gets all .vrp files from a given path (file or directory)
    and optionally filters them by maximum dimension.
    """
    potential_files: List[str] = [] # Type hint
    if os.path.isfile(path) and path.endswith(".vrp"):
        potential_files = [path]
    elif os.path.isdir(path):
        potential_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".vrp")]
    else:
        raise ValueError(f"Invalid instance path: {path}. Must be a .vrp file or a directory.")

    if not potential_files:
        return []

    if max_dimension is None:
        return potential_files

    filtered_files: List[str] = [] # Type hint
    print(f"Filtering instances: Max dimension set to {max_dimension}")
    for f_path in potential_files:
        try:
            # Briefly load instance to check dimension
            # This adds a bit of overhead at the start but ensures correct filtering.
            instance = load_cvrp_instance(f_path)
            if instance.dimension <= max_dimension:
                filtered_files.append(f_path)
            else:
                print(f"  Excluding (too large): {instance.name} (Dim: {instance.dimension})")
        except Exception as e:
            print(f"  Warning: Could not load/check instance {f_path} for filtering: {e}")
    
    if not filtered_files and potential_files: # If filtering removed all files
        print(f"Warning: All instances from {path} were filtered out by max_dimension={max_dimension}.")

    return filtered_files


def save_instance_run_results(
    results_dir: str,
    instance_name: str,
    episode: int,
    data_to_save: Dict[str, Any] # Type hint
):
    """Saves the results of a single ALM run for an instance."""
    instance_folder = Path(results_dir) / instance_name
    instance_folder.mkdir(parents=True, exist_ok=True)
    
    # Sanitize instance_name if it contains problematic characters for filenames (though usually not for CVRPLIB)
    sanitized_instance_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in instance_name)

    file_name = f"ep{episode}_{sanitized_instance_name}_results.json"
    file_path = instance_folder / file_name
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data_to_save, f, indent=4, default=lambda o: '<not serializable>') # Handle non-serializable
        # print(f"  Saved results for {instance_name} ep {episode} to {file_path}") # Optional: verbose saving log
    except Exception as e:
        print(f"  Error saving results for {instance_name} ep {episode}: {e}")


def evaluate_agent(
    env: AlmPenaltyEnv, 
    agent: SACAgent, 
    results_dir: str, 
    eval_episodes: int = 5,
    current_train_episode: int = 0
    ) -> Tuple[float, float, float]: # Type hint for return
    total_rewards = 0.0
    total_alm_costs = 0.0
    total_alm_iterations = 0
    feasible_solutions_count = 0
    
    print(f"\n--- Evaluating agent for {eval_episodes} episodes (Train Ep: {current_train_episode}) ---")
    
    for i in range(eval_episodes):
        state, reset_info = env.reset() 
        instance_name = reset_info.get("instance_name", f"eval_instance_{i}")
        
        action = agent.select_action(np.array(state), evaluate=True)
        _, reward, _, _, step_info = env.step(action)
        
        total_rewards += reward
        alm_cost = step_info.get("solution_cost", float('inf'))
        if alm_cost != float('inf'):
            total_alm_costs += alm_cost
            feasible_solutions_count += 1
        total_alm_iterations += step_info.get("alm_iterations", 0)
        
        print(f"Eval Ep {i+1}/{eval_episodes} | Inst: {instance_name} | Reward: {reward:.2f} | ALM Cost: {alm_cost:.2f} | ALM Iters: {step_info.get('alm_iterations', 0)}")

        alm_solution_obj = step_info.get("alm_solution_object")
        routes_to_save = alm_solution_obj.routes if alm_solution_obj and hasattr(alm_solution_obj, 'routes') else None # Check attribute
        
        eval_data_to_save: Dict[str, Any] = { # Type hint
            "episode_type": "evaluation",
            "training_episode_ref": current_train_episode,
            "eval_episode_num": i + 1,
            "instance_name": instance_name,
            "chosen_rho": float(action[0]),
            "chosen_sigma": float(action[1]),
            "alm_cost": alm_cost if alm_cost != float('inf') else "inf",
            "alm_iterations": step_info.get("alm_iterations", 0),
            "time_taken_alm_solve": step_info.get("time_taken_alm_solve", 0),
            "solution_feasible": step_info.get("solution_feasible", False),
            "routes": routes_to_save
        }
        save_instance_run_results(results_dir, f"eval_{instance_name}", current_train_episode * 1000 + i, eval_data_to_save)


    avg_reward = total_rewards / eval_episodes if eval_episodes > 0 else 0.0
    avg_alm_cost = total_alm_costs / feasible_solutions_count if feasible_solutions_count > 0 else float('inf')
    avg_alm_iterations = total_alm_iterations / eval_episodes if eval_episodes > 0 else 0.0
    feasibility_rate = feasible_solutions_count / eval_episodes if eval_episodes > 0 else 0.0
    
    print(f"--- Evaluation Summary (Train Ep: {current_train_episode}) ---")
    print(f"Avg Reward: {avg_reward:.2f}")
    print(f"Avg ALM Cost (for feasible): {avg_alm_cost:.2f}")
    print(f"Avg ALM Iterations: {avg_alm_iterations:.2f}")
    print(f"Feasibility Rate: {feasibility_rate:.2%}")
    print(f"--------------------------\n")
    return avg_reward, avg_alm_cost, feasibility_rate


def main(args: argparse.Namespace): # Type hint for args
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device == torch.device("cuda"): # pragma: no cover
            torch.cuda.manual_seed_all(args.seed)
        print(f"Set random seed to: {args.seed}")

    train_instance_paths = get_instance_files_and_filter(
        args.train_instances_path,
        args.max_instance_dimension_train
    )
    if not train_instance_paths:
        print(f"No training instances left after filtering from {args.train_instances_path}. Exiting.")
        return
    print(f"Using {len(train_instance_paths)} training instances after filtering.")

    eval_instance_paths: List[str] = [] # Type hint
    if args.eval_instances_path:
        eval_instance_paths = get_instance_files_and_filter(
            args.eval_instances_path,
            args.max_instance_dimension_eval
        )
        if not eval_instance_paths:
            print(f"Warning: No evaluation instance files found or all filtered from {args.eval_instances_path}")
        else:
            print(f"Using {len(eval_instance_paths)} evaluation instances after filtering.")
    
    train_env = AlmPenaltyEnv(
        instance_paths=train_instance_paths,
        max_alm_iterations_override=args.max_alm_iters
    )
    
    eval_env: Optional[AlmPenaltyEnv] = None # Type hint
    if eval_instance_paths:
        eval_env = AlmPenaltyEnv(
            instance_paths=eval_instance_paths,
            max_alm_iterations_override=args.max_alm_iters
        )

    sac_hp = DEFAULT_SAC_HYPERPARAMS.copy()
    sac_hp["lr_actor"] = args.lr_actor
    sac_hp["lr_critic"] = args.lr_critic
    sac_hp["lr_alpha"] = args.lr_alpha
    sac_hp["gamma"] = args.gamma
    sac_hp["tau"] = args.tau
    sac_hp["alpha"] = args.alpha_init
    sac_hp["learn_alpha"] = args.learn_alpha
    sac_hp["buffer_size"] = args.buffer_size
    sac_hp["batch_size"] = args.batch_size
    sac_hp["hidden_dim"] = args.hidden_dim

    agent = SACAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        hyperparams=sac_hp,
        device=device
    )
    print(f"SAC Agent initialized with action_dim={train_env.action_space.shape[0]}, state_dim={train_env.observation_space.shape[0]}")

    run_output_base_dir = Path(args.run_output_dir)
    run_output_base_dir.mkdir(parents=True, exist_ok=True)
    
    model_save_dir = run_output_base_dir / "rl_agent_models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    instance_results_root_dir = Path(args.instance_results_dir)
    instance_results_root_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    episode_rewards: deque[float] = deque(maxlen=args.log_interval) # Type hint
    episode_alm_costs: deque[float] = deque(maxlen=args.log_interval) # Type hint
    episode_alm_iters: deque[float] = deque(maxlen=args.log_interval) # Type hint

    print(f"\nStarting training for {args.total_episodes} episodes...")
    print(f"Models will be saved in: {model_save_dir}")
    print(f"Instance run results will be saved in: {instance_results_root_dir}")

    for episode in range(1, args.total_episodes + 1):
        state, reset_info = train_env.reset()
        instance_name = reset_info.get("instance_name", "unknown_instance")

        action = agent.select_action(np.array(state), evaluate=False)
        dummy_next_state, reward, terminated, truncated, step_info = train_env.step(action)
        
        agent.store_experience(np.array(state), action, reward, np.array(dummy_next_state), terminated)

        if len(agent.replay_buffer) >= args.batch_size and episode >= args.start_learning_after_episodes:
            for _ in range(args.updates_per_step):
                agent.update_parameters()
        
        episode_rewards.append(reward)
        current_alm_cost = step_info.get("solution_cost", float('inf'))
        if current_alm_cost != float('inf'):
            episode_alm_costs.append(current_alm_cost)
        episode_alm_iters.append(float(step_info.get("alm_iterations", 0))) # Ensure float for deque

        alm_solution_obj = step_info.get("alm_solution_object")
        routes_to_save = alm_solution_obj.routes if alm_solution_obj and hasattr(alm_solution_obj, 'routes') else None
        
        data_to_save: Dict[str, Any] = { # Type hint
            "episode_type": "training",
            "episode_num": episode,
            "instance_name": instance_name,
            "chosen_rho": float(action[0]),
            "chosen_sigma": float(action[1]),
            "alm_cost": current_alm_cost if current_alm_cost != float('inf') else "inf",
            "alm_iterations": step_info.get("alm_iterations", 0),
            "time_taken_alm_solve": step_info.get("time_taken_alm_solve", 0),
            "solution_feasible": step_info.get("solution_feasible", False),
            "reward_received": reward,
            "routes": routes_to_save
        }
        save_instance_run_results(str(instance_results_root_dir), instance_name, episode, data_to_save)

        if episode % args.log_interval == 0:
            avg_reward = np.mean(list(episode_rewards)) if episode_rewards else float('nan') # Convert deque to list for np.mean
            avg_alm_cost = np.mean(list(episode_alm_costs)) if episode_alm_costs else float('nan')
            avg_alm_iter = np.mean(list(episode_alm_iters)) if episode_alm_iters else float('nan')
            elapsed_time = time.time() - start_time
            print(f"Ep: {episode}/{args.total_episodes} | Inst: {instance_name} | Avg Reward (last {args.log_interval}): {avg_reward:.2f} | "
                  f"Avg ALM Cost: {avg_alm_cost:.2f} | Avg ALM Iter: {avg_alm_iter:.1f} | "
                  f"Alpha: {agent.alpha:.4f} | Time: {elapsed_time:.1f}s")

        if episode % args.eval_interval == 0 and eval_env:
            eval_results_dir = instance_results_root_dir / "evaluation_runs" 
            eval_results_dir.mkdir(parents=True, exist_ok=True)
            evaluate_agent(eval_env, agent, str(eval_results_dir), 
                           eval_episodes=args.eval_episodes, current_train_episode=episode)
            agent.actor.train() # Ensure agent is back in training mode
            agent.critic.train()

        if episode % args.save_interval == 0:
            agent.save_model(directory=str(model_save_dir), filename_prefix=f"sac_penalty_learner_ep{episode}")

    agent.save_model(directory=str(model_save_dir), filename_prefix="sac_penalty_learner_final")
    print(f"Training finished. Total time: {time.time() - start_time:.1f}s")
    print(f"Final models saved in: {model_save_dir}")
    print(f"Instance run results saved in: {instance_results_root_dir}")
    train_env.close()
    if eval_env:
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent to learn ALM penalty parameters for CVRP.")
    
    parser.add_argument("--train_instances_path", type=str, required=True, help="Path to train .vrp file(s) or directory.")
    parser.add_argument("--eval_instances_path", type=str, default=None, help="Path to eval .vrp file(s) or directory.")
    parser.add_argument("--max_instance_dimension_train", type=int, default=100,
                        help="Max instance dimension (nodes) to include for training. Others are filtered out.")
    parser.add_argument("--max_instance_dimension_eval", type=int, default=150, 
                        help="Max instance dimension for evaluation instances.")

    parser.add_argument("--total_episodes", type=int, default=10000)
    parser.add_argument("--max_alm_iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--start_learning_after_episodes", type=int, default=1000)
    parser.add_argument("--updates_per_step", type=int, default=1)

    parser.add_argument("--lr_actor", type=float, default=DEFAULT_SAC_HYPERPARAMS["lr_actor"])
    parser.add_argument("--lr_critic", type=float, default=DEFAULT_SAC_HYPERPARAMS["lr_critic"])
    parser.add_argument("--lr_alpha", type=float, default=DEFAULT_SAC_HYPERPARAMS["lr_alpha"])
    parser.add_argument("--gamma", type=float, default=DEFAULT_SAC_HYPERPARAMS["gamma"])
    parser.add_argument("--tau", type=float, default=DEFAULT_SAC_HYPERPARAMS["tau"])
    parser.add_argument("--alpha_init", type=float, default=DEFAULT_SAC_HYPERPARAMS["alpha"])
    parser.add_argument("--learn_alpha", type=lambda x: (str(x).lower() == 'true'), default=DEFAULT_SAC_HYPERPARAMS["learn_alpha"])
    parser.add_argument("--buffer_size", type=int, default=DEFAULT_SAC_HYPERPARAMS["buffer_size"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_SAC_HYPERPARAMS["batch_size"])
    parser.add_argument("--hidden_dim", type=int, default=DEFAULT_SAC_HYPERPARAMS["hidden_dim"])

    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=5000)
    
    parser.add_argument("--run_output_dir", type=str, default="./rl_training_output", 
                        help="Main directory to save all outputs for this run (models, logs).")
    parser.add_argument("--instance_results_dir", type=str, default="results/rl_classical_instance_runs",
                        help="Directory to save detailed results for each instance run (relative to project root or absolute).")

    args = parser.parse_args()
    main(args)
