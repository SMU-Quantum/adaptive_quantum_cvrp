# experiments/run_rl_penalty_learner_q_alm_experiment.py

import argparse
import os
import random
import time
import json
from collections import deque
import glob

import numpy as np
import torch

import sys # Import sys to modify path

# --- Add project root to sys.path ---
# This allows imports from the 'src' directory when running from 'experiments'
# Assumes 'experiments' is a subdirectory of the project root, and 'src' is also under the project root.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End of sys.path modification ---


# Assuming your project structure allows these imports
# CORRECTED IMPORT: Changed sac_agent to penalty_agent
from src.rl_classical.penalty_agent import SACAgent # Your SAC Agent implementation
from src.rl_classical.replay_buffer import ReplayBuffer # Your ReplayBuffer
# Import the NEW Quantum ALM Environment
from src.rl_classical.alm_penalty_env_qubo_vqe import AlmPenaltyEnvQuBoVqe, N_INSTANCE_FEATURES 
from src.common.cvrp_instance import load_cvrp_instance # For loading eval instances directly if needed


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using CUDA
    np.random.seed(seed)
    random.seed(seed)

def get_instance_files(path_pattern: str) -> list[str]:
    """Gets a list of .vrp files from a directory or a single file."""
    if not path_pattern: 
        return []
    if os.path.isfile(path_pattern) and path_pattern.endswith(".vrp"):
        return [path_pattern]
    elif os.path.isdir(path_pattern):
        return glob.glob(os.path.join(path_pattern, "*.vrp"))
    elif path_pattern.endswith(".vrp") and ("*" in path_pattern or "?" in path_pattern): 
        return glob.glob(path_pattern)
    return []

def evaluate_agent(env, agent, eval_episodes=10, current_training_episode=0, instance_results_dir=None):
    """Evaluates the agent's performance on the evaluation environment."""
    print(f"\n--- Evaluating Agent at Training Episode {current_training_episode} ---")
    total_rewards = 0.0
    total_alm_costs = 0.0
    total_alm_iters = 0
    feasible_solutions_count = 0
    num_eval_instances_processed = 0

    if hasattr(agent, 'actor') and hasattr(agent.actor, 'eval'): agent.actor.eval()
    if hasattr(agent, 'critic') and hasattr(agent.critic, 'eval'): agent.critic.eval()

    for i in range(eval_episodes):
        obs, info = env.reset()
        action = agent.select_action(obs, evaluate=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        total_rewards += reward
        instance_name = step_info.get("instance_name", "unknown_eval_instance")
        print(f"  Eval Ep {i+1}/{eval_episodes} | Inst: {instance_name} | Reward: {reward:.2f} | "
              f"ALM Cost: {step_info.get('solution_cost', 'N/A')} | "
              f"ALM Iters: {step_info.get('alm_iterations', 0)} | Feasible: {step_info.get('is_feasible', False)}")

        if step_info.get('is_feasible', False) and step_info.get('solution_cost') is not None:
            total_alm_costs += step_info['solution_cost']
            feasible_solutions_count += 1
        total_alm_iters += step_info.get('alm_iterations', 0)
        num_eval_instances_processed +=1

        if instance_results_dir:
            eval_run_subdir = os.path.join(instance_results_dir, f"train_ep_{current_training_episode}")
            os.makedirs(eval_run_subdir, exist_ok=True)
            eval_instance_file = os.path.join(eval_run_subdir, f"eval_inst{i+1}_{instance_name.replace('.vrp','')}_results.json")
            try:
                with open(eval_instance_file, 'w') as f:
                    json.dump(step_info, f, indent=4, cls=NpEncoder)
            except Exception as e:
                print(f"    Warning: Could not save eval instance results for {instance_name}: {e}")

    avg_reward = total_rewards / eval_episodes if eval_episodes > 0 else 0
    avg_alm_cost = total_alm_costs / feasible_solutions_count if feasible_solutions_count > 0 else float('inf')
    avg_alm_iters = total_alm_iters / num_eval_instances_processed if num_eval_instances_processed > 0 else 0
    feasibility_rate = feasible_solutions_count / num_eval_instances_processed if num_eval_instances_processed > 0 else 0
    
    print(f"--- Evaluation Summary ---")
    print(f"  Avg Reward: {avg_reward:.2f} | Avg Feasible ALM Cost: {avg_alm_cost:.2f} | "
          f"Avg ALM Iters: {avg_alm_iters:.2f} | Feasibility Rate: {feasibility_rate:.2%}")
    print(f"--------------------------\n")

    if hasattr(agent, 'actor') and hasattr(agent.actor, 'train'): agent.actor.train()
    if hasattr(agent, 'critic') and hasattr(agent.critic, 'train'): agent.critic.train()

    return avg_reward, avg_alm_cost, feasibility_rate

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        elif isinstance(obj, np.floating): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_instance_files = get_instance_files(args.train_instances_path)
    if not train_instance_files:
        raise FileNotFoundError(f"No training instance files found at or via pattern: {args.train_instances_path}")
    print(f"Found {len(train_instance_files)} training instances from: {args.train_instances_path}")

    eval_instance_files = []
    if args.eval_instances_path:
        eval_instance_files = get_instance_files(args.eval_instances_path)
        if not eval_instance_files: print(f"Warning: No evaluation instance files found at or via pattern: {args.eval_instances_path}")
        else: print(f"Found {len(eval_instance_files)} evaluation instances from: {args.eval_instances_path}")
    else:
        print("No evaluation instance path provided, skipping evaluation runs during training.")

    alm_config_for_env = {
        'max_alm_iterations': args.max_alm_iters_env,
        'penalty_increase_factor': 1.0,
        'capacity_penalty_increase_factor': 1.0,
        'verbose_alm': args.verbose_alm,
        'subproblem_max_vehicles': args.alm_subproblem_max_vehicles 
    }
    quantum_solver_config_for_env = {
        'max_customers_in_quantum_subproblem': args.q_max_cust_in_subproblem,
        'constraint_penalty_factor': args.q_constraint_penalty,
        'vqe_reps': args.q_vqe_reps,
        'vqe_max_iter': args.q_vqe_max_iter, 
        'vqe_optimizer_method': args.q_vqe_optimizer,
        'plot_folder_prefix': os.path.join(args.run_output_dir, "quantum_plots/plot_") 
    }
    os.makedirs(os.path.join(args.run_output_dir, "quantum_plots"), exist_ok=True)

    env = AlmPenaltyEnvQuBoVqe(
        instance_files=train_instance_files,
        alm_config=alm_config_for_env,
        quantum_solver_config=quantum_solver_config_for_env,
        max_instance_dimension=args.max_instance_dimension_train
    )

    eval_env = None
    if eval_instance_files:
        eval_env = AlmPenaltyEnvQuBoVqe(
            instance_files=eval_instance_files,
            alm_config=alm_config_for_env, 
            quantum_solver_config=quantum_solver_config_for_env, 
            max_instance_dimension=args.max_instance_dimension_eval
        )
    else:
        print("Evaluation environment not created as no eval instances were loaded.")

    agent = SACAgent( # SACAgent class is now correctly imported from penalty_agent
        state_dim=N_INSTANCE_FEATURES, 
        action_dim=env.action_space.shape[0],
        # hyperparams argument for SACAgent constructor:
        hyperparams={ 
            "hidden_dim": args.hidden_dim,
            "lr_actor": args.lr_actor,
            "lr_critic": args.lr_critic,
            "lr_alpha": args.lr_alpha,
            "gamma": args.gamma,
            "tau": args.tau,
            "alpha": args.alpha_sac, 
            "learn_alpha": args.learn_alpha,
            "target_entropy_scale": args.target_entropy_scale,
            "buffer_size": args.buffer_size, # Pass buffer_size here if SACAgent initializes its own buffer
            "batch_size": args.batch_size   # Pass batch_size here if SACAgent uses it internally beyond update_params
        },
        device=device
    )
    print(f"SAC Agent initialized with action_dim={env.action_space.shape[0]}, state_dim={N_INSTANCE_FEATURES}")

    # Replay buffer is now managed by the SACAgent itself as per your penalty_agent.py
    # So, we don't create a separate ReplayBuffer here.
    # replay_buffer = ReplayBuffer(args.buffer_size, args.seed) # This line is removed.

    obs, info = env.reset(seed=args.seed) 
    
    rewards_deque = deque(maxlen=args.log_interval)
    alm_costs_deque = deque(maxlen=args.log_interval)
    alm_iters_deque = deque(maxlen=args.log_interval)
    episode_times_deque = deque(maxlen=args.log_interval)

    total_steps = 0
    start_train_time = time.time()

    print(f"\nStarting training for {args.total_episodes} episodes...")
    print(f"WARNING: Each episode involves a full ALM run with VQE and will be VERY SLOW.")
    print(f"Consider very low 'total_episodes' and 'q_vqe_max_iter' for initial tests.")

    for episode in range(1, args.total_episodes + 1):
        episode_start_time = time.time()
        
        if episode > 1: 
             obs, info = env.reset() 

        instance_name = info.get("instance_name", "unknown_instance")
        action = agent.select_action(obs, evaluate=False)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        total_steps += 1 

        # Use agent's internal method to store experience
        agent.store_experience(obs, action, reward, next_obs, terminated) 
        obs = next_obs 

        if args.instance_results_dir:
            os.makedirs(args.instance_results_dir, exist_ok=True)
            instance_file_path = os.path.join(args.instance_results_dir, f"ep{episode}_{instance_name.replace('.vrp','')}_results.json")
            try:
                with open(instance_file_path, 'w') as f:
                    json.dump(step_info, f, indent=4, cls=NpEncoder)
            except Exception as e:
                print(f"  Warning: Could not save instance results for {instance_name}: {e}")

        rewards_deque.append(reward)
        if step_info.get('is_feasible') and step_info.get('solution_cost') is not None:
            alm_costs_deque.append(step_info['solution_cost'])
        alm_iters_deque.append(step_info.get('alm_iterations', 0))
        episode_times_deque.append(time.time() - episode_start_time)

        # Agent update uses its internal buffer
        if episode >= args.start_learning_after_episodes and len(agent.replay_buffer) >= agent.hp["batch_size"]:
            for _ in range(args.updates_per_step): 
                # agent.update_parameters now directly uses its internal buffer and batch_size
                agent.update_parameters() 
        
        if episode % args.log_interval == 0:
            avg_reward = np.mean(rewards_deque) if rewards_deque else -1
            avg_alm_cost = np.mean(alm_costs_deque) if alm_costs_deque else -1
            avg_alm_iters = np.mean(alm_iters_deque) if alm_iters_deque else -1
            avg_ep_time = np.mean(episode_times_deque) if episode_times_deque else -1
            
            print(f"Ep: {episode}/{args.total_episodes} | Step: {total_steps} | Inst: {instance_name} | "
                  f"Avg Reward (last {args.log_interval}): {avg_reward:.2f} | "
                  f"Avg ALM Cost: {avg_alm_cost:.2f} | Avg ALM It: {avg_alm_iters:.1f} | "
                  f"Avg Ep Time: {avg_ep_time:.2f}s")
            print(f"  Last penalties (rho, sigma): {step_info.get('chosen_rho',0):.2f}, {step_info.get('chosen_sigma',0):.2f} | "
                  f"Last ALM time: {step_info.get('time_taken_alm_solve',0):.2f}s")
            if hasattr(agent, 'alpha'): print(f"  SAC Alpha: {agent.alpha.item():.4f}" if isinstance(agent.alpha, torch.Tensor) else f"  SAC Alpha: {agent.alpha:.4f}")

        if args.eval_episodes > 0 and eval_env and episode % args.eval_interval == 0:
            eval_results_path = None
            if args.instance_results_dir:
                eval_results_path = os.path.join(args.instance_results_dir, "evaluation_runs")
                os.makedirs(eval_results_path, exist_ok=True) 
            evaluate_agent(eval_env, agent, args.eval_episodes, episode, eval_results_path)
            
        if episode % args.save_interval == 0:
            model_save_dir = os.path.join(args.run_output_dir, "rl_agent_models")
            os.makedirs(model_save_dir, exist_ok=True) 
            agent.save_model(directory=model_save_dir, filename_prefix=f"sac_q_alm_ep{episode}")
            print(f"Saved model at episode {episode} to {model_save_dir}")

    final_model_save_dir = os.path.join(args.run_output_dir, "rl_agent_models")
    os.makedirs(final_model_save_dir, exist_ok=True) 
    agent.save_model(directory=final_model_save_dir, filename_prefix="sac_q_alm_final")
    print(f"Saved final model to {final_model_save_dir}")
    
    total_train_time = time.time() - start_train_time
    print(f"\nTotal Training Time: {total_train_time / 3600:.2f} hours ({total_train_time:.2f} seconds)")
    env.close()
    if eval_env:
        eval_env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC agent to learn ALM penalties with Quantum ALM.")
    
    parser.add_argument("--train_instances_path", type=str, default="data/cvrplib_instances_train/", help="Path to directory or .vrp file(s) for training TINY instances.")
    parser.add_argument("--eval_instances_path", type=str, default="data/cvrplib_instances_test/", help="Path to directory or .vrp file(s) for TINY evaluation instances.")
    parser.add_argument("--run_output_dir", type=str, default="./rl_q_alm_output", help="Directory to save models, plots, and logs.")
    parser.add_argument("--instance_results_dir", type=str, default=None, help="Directory to save detailed JSON results for each instance run. If None, not saved. Subdirs for eval will be created.")
    parser.add_argument("--save_interval", type=int, default=1000, help="Save model every N episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA training if available.")
    parser.add_argument("--total_episodes", type=int, default=1000, help="Total number of training episodes (ALM runs). WARNING: Quantum is SLOW.")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Replay buffer size (used by SACAgent internally).")
    parser.add_argument("--batch_size", type=int, default=64, help="SAC batch size (used by SACAgent internally).")
    parser.add_argument("--hidden_dim", type=int, default=256, help="SAC actor/critic hidden layer dimensions.")
    parser.add_argument("--lr_actor", type=float, default=3e-4, help="Actor learning rate.")
    parser.add_argument("--lr_critic", type=float, default=3e-4, help="Critic learning rate.")
    parser.add_argument("--lr_alpha", type=float, default=3e-4, help="Alpha learning rate (if learn_alpha).")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update coefficient.")
    parser.add_argument("--alpha_sac", type=float, default=0.2, help="Initial SAC temperature alpha.")
    parser.add_argument("--learn_alpha", type=bool, default=True, help="Whether alpha is learnable.")
    parser.add_argument("--target_entropy_scale", type=float, default=0.8, help="Scale for target entropy if learn_alpha (e.g., 0.8 * -action_dim).")
    parser.add_argument("--start_learning_after_episodes", type=int, default=100, help="Number of episodes to fill agent's buffer before learning starts.")
    parser.add_argument("--updates_per_step", type=int, default=1, help="Number of gradient updates per episode/step.")
    parser.add_argument("--max_instance_dimension_train", type=int, default=5, help="Max dimension of training instances (nodes). MUST BE TINY.")
    parser.add_argument("--max_instance_dimension_eval", type=int, default=5, help="Max dimension of evaluation instances. MUST BE TINY.")
    parser.add_argument("--max_alm_iters_env", type=int, default=20, help="Max ALM iterations within each environment step. Keep low.")
    parser.add_argument("--alm_subproblem_max_vehicles", type=int, default=None, help="Max routes ALM tries to generate per iter. None uses instance default.")
    parser.add_argument("--verbose_alm", type=int, default=0, help="Verbosity for ALM runs within env (0=silent).")
    parser.add_argument("--q_max_cust_in_subproblem", type=int, default=2, help="Max customers (excluding depot) in each quantum TSP subproblem.")
    parser.add_argument("--q_constraint_penalty", type=float, default=500.0, help="Penalty for QuadraticProgramToQubo converter for TSP constraints.")
    parser.add_argument("--q_vqe_reps", type=int, default=1, help="Ansatz repetitions for VQE.")
    parser.add_argument("--q_vqe_max_iter", type=int, default=30, help="Max iterations for VQE's classical optimizer. CRITICAL FOR SPEED.")
    parser.add_argument("--q_vqe_optimizer", type=str, default="Powell", help="Classical optimizer for VQE (e.g., SPSA, COBYLA).")
    parser.add_argument("--log_interval", type=int, default=10, help="Log average results every N episodes.")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of episodes for evaluation run.")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluate agent every N training episodes.")

    args = parser.parse_args()
    
    os.makedirs(args.run_output_dir, exist_ok=True)
    if args.instance_results_dir:
        os.makedirs(args.instance_results_dir, exist_ok=True)

    args_path = os.path.join(args.run_output_dir, "args_config.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Saved run arguments to {args_path}")

    main(args)


"""""
python experiments/run_rl_penalty_learner_q_alm_experiment.py \
    --train_instances_path ./data/cvrplib_instances_train/ \
    --eval_instances_path ./data/cvrplib_instances_test/ \
    --run_output_dir ./rl_q_alm_run_1_output \
    --instance_results_dir ./rl_q_alm_run_1_output/instance_details \
    --total_episodes 100 \
    --start_learning_after_episodes 20 \
    --log_interval 10 \
    --eval_episodes 5 \
    --eval_interval 50 \
    --max_instance_dimension_train 4 \
    --max_instance_dimension_eval 4 \
    --max_alm_iters_env 15 \
    --q_max_cust_in_subproblem 2 \
    --q_vqe_max_iter 25 \
    --q_constraint_penalty 500.0 \
    --save_interval 50 \
    --batch_size 16 \
    --buffer_size 500 \
    --seed 42
"""

"""
python experiments\run_rl_penalty_learner_q_alm_experiment.py --train_instances_path .\data\cvrplib_instances_train\ --eval_instances_path .\data\cvrplib_instances_test\ --run_output_dir .\rl_q_alm_run_longer_output --instance_results_dir .\rl_q_alm_run_longer_output\instance_details --total_episodes 10000 --start_learning_after_episodes 1000 --log_interval 100 --eval_episodes 10 --eval_interval 500 --max_instance_dimension_train 4 --max_instance_dimension_eval 4 --max_alm_iters_env 20 --q_max_cust_in_subproblem 2 --q_vqe_max_iter 30 --q_constraint_penalty 500.0 --save_interval 1000 --batch_size 64 --buffer_size 10000 --seed 42
"""