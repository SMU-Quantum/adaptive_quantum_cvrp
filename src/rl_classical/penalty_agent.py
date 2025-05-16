# cvrp_tripartite_solver/src/rl_classical/penalty_agent.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy # For target network updates
import os # For save/load model
import random # For the __main__ block

# Assuming these are in the same directory or PYTHONPATH is configured
from .penalty_networks import (
    Actor, Critic, N_INSTANCE_FEATURES,
    ACTION_SCALE_RHO, ACTION_BIAS_RHO, # Import scaling constants
    ACTION_SCALE_SIGMA, ACTION_BIAS_SIGMA, # Make sure these are defined in penalty_networks
    MIN_PENALTY_RHO, MAX_PENALTY_RHO, MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA # Also needed for clamping
)
from .replay_buffer import ReplayBuffer

# Default SAC hyperparameters
DEFAULT_SAC_HYPERPARAMS = {
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.2,
    "lr_actor": 3e-4,
    "lr_critic": 3e-4,
    "lr_alpha": 3e-4,
    "hidden_dim": 256,
    "buffer_size": int(1e6),
    "batch_size": 256,
    "learn_alpha": True,
    "target_entropy_scale": 1.0 # Typically -action_dim for auto alpha tuning, but can be scaled
}

class SACAgent:
    def __init__(self,
                 state_dim: int = N_INSTANCE_FEATURES,
                 action_dim: int = 2, # rho and sigma
                 hyperparams: dict = None,
                 device: torch.device = None):

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.hp = DEFAULT_SAC_HYPERPARAMS.copy()
        if hyperparams:
            self.hp.update(hyperparams)

        self.actor = Actor(state_dim, action_dim, self.hp["hidden_dim"]).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.hp["lr_actor"])

        self.critic = Critic(state_dim, action_dim, self.hp["hidden_dim"]).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.hp["lr_critic"])

        self.critic_target = Critic(state_dim, action_dim, self.hp["hidden_dim"]).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        if self.hp["learn_alpha"]:
            # Target entropy is often set to -action_dim or a fraction of it
            self.target_entropy = -torch.prod(torch.Tensor((float(action_dim),))).item() * self.hp["target_entropy_scale"]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.hp["lr_alpha"])
            self.alpha = self.log_alpha.exp().item() # Keep a Python float for current alpha
        else:
            self.alpha = self.hp["alpha"]
            self.target_entropy = None 
            self.log_alpha = None
            self.alpha_optimizer = None

        self.replay_buffer = ReplayBuffer(state_dim, action_dim, self.hp["buffer_size"], self.device)
        self.total_it = 0 # Counter for learning iterations

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            # For deterministic evaluation, use the mean of the policy distribution,
            # then tanh, then scale.
            mean_unscaled, _ = self.actor.forward(state_tensor) # Get raw mean from actor's forward pass
            y_t = torch.tanh(mean_unscaled) # Apply tanh squashing

            # Scale the squashed mean to the actual action range
            action_rho_scaled = y_t[:, 0:1] * ACTION_SCALE_RHO + ACTION_BIAS_RHO
            action_sigma_scaled = y_t[:, 1:2] * ACTION_SCALE_SIGMA + ACTION_BIAS_SIGMA
            
            # Clamp to ensure actions are strictly within defined penalty limits
            action_rho_clamped = torch.clamp(action_rho_scaled, MIN_PENALTY_RHO, MAX_PENALTY_RHO)
            action_sigma_clamped = torch.clamp(action_sigma_scaled, MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA)

            action = torch.cat([action_rho_clamped, action_sigma_clamped], dim=1)

        else: # Sample stochastically
            # actor.sample() already returns scaled and clamped actions
            action, _, _ = self.actor.sample(state_tensor) 
        
        return action.cpu().data.numpy().flatten()

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def update_parameters(self):
        self.total_it += 1
        if len(self.replay_buffer) < self.hp["batch_size"]:
            return None # Return None or specific values indicating no update

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.hp["batch_size"])

        # --- Update Critic ---
        with torch.no_grad():
            next_actions_scaled, next_log_pi, _ = self.actor.sample(next_states)
            q1_target_next, q2_target_next = self.critic_target(next_states, next_actions_scaled)
            q_target_next = torch.min(q1_target_next, q2_target_next)
            
            current_alpha_val = self.log_alpha.exp().detach() if self.hp["learn_alpha"] else self.alpha
            target_q_values = rewards + (1 - dones) * self.hp["gamma"] * (q_target_next - current_alpha_val * next_log_pi)

        q1_current, q2_current = self.critic(states, actions) # actions are already scaled from buffer
        
        critic_loss_q1 = F.mse_loss(q1_current, target_q_values)
        critic_loss_q2 = F.mse_loss(q2_current, target_q_values)
        critic_loss = critic_loss_q1 + critic_loss_q2

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        pi_actions_scaled, log_pi, _ = self.actor.sample(states) # scaled actions
        q1_pi, q2_pi = self.critic(states, pi_actions_scaled) # Use current critic
        q_pi = torch.min(q1_pi, q2_pi)

        current_alpha_val = self.log_alpha.exp().detach() if self.hp["learn_alpha"] else self.alpha
        actor_loss = (current_alpha_val * log_pi - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Alpha (Temperature) if learning alpha ---
        alpha_loss_val = 0.0 # Default if not learning alpha
        if self.hp["learn_alpha"] and self.alpha_optimizer and self.log_alpha is not None:
            # log_pi here should be from the latest actions from the current policy
            alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item() # Update Python float alpha
            alpha_loss_val = alpha_loss.item()

        # --- Soft update target critic networks ---
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.hp["tau"] * param.data + (1 - self.hp["tau"]) * target_param.data)
        
        return critic_loss.item(), actor_loss.item(), alpha_loss_val


    def save_model(self, directory="./models", filename_prefix="sac_penalty_learner"):
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        torch.save(self.actor.state_dict(), f'{directory}/{filename_prefix}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename_prefix}_critic.pth')
        if self.hp["learn_alpha"] and self.log_alpha is not None:
            torch.save(self.log_alpha, f'{directory}/{filename_prefix}_log_alpha.pth')
        # print(f"Models saved to {directory}") # Optional: uncomment for verbose saving

    def load_model(self, directory="./models", filename_prefix="sac_penalty_learner"):
        self.actor.load_state_dict(torch.load(f'{directory}/{filename_prefix}_actor.pth', map_location=self.device))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename_prefix}_critic.pth', map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict()) # Keep target in sync

        if self.hp["learn_alpha"]:
            try:
                loaded_log_alpha = torch.load(f'{directory}/{filename_prefix}_log_alpha.pth', map_location=self.device)
                # Ensure log_alpha is a parameter that requires grad for the optimizer
                if not isinstance(loaded_log_alpha, torch.nn.Parameter):
                     self.log_alpha = torch.nn.Parameter(loaded_log_alpha.data.clone().detach().requires_grad_(True))
                else: # If already a parameter, ensure it's correctly set up
                    self.log_alpha = loaded_log_alpha
                    self.log_alpha.requires_grad_(True)

                # Re-initialize alpha_optimizer with the loaded log_alpha
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.hp["lr_alpha"])
                self.alpha = self.log_alpha.exp().item()
            except FileNotFoundError:
                print(f"Warning: Log_alpha file not found at {directory}/{filename_prefix}_log_alpha.pth. Re-initializing alpha.")
                # If file not found, re-initialize log_alpha and its optimizer as in __init__
                self.target_entropy = -torch.prod(torch.Tensor((float(self.actor.mean_layer.out_features),))).item() * self.hp["target_entropy_scale"]
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.hp["lr_alpha"])
                self.alpha = self.log_alpha.exp().item()
        
        # Set networks to evaluation mode after loading, if primarily for inference.
        # For continued training, this might be set back to train() mode by the training loop.
        self.actor.eval()
        self.critic.eval()
        self.critic_target.eval()
        # print(f"Models loaded from {directory}") # Optional: uncomment for verbose loading


if __name__ == '__main__':
    # import random # Already imported at the top
    print("--- SACAgent Basic Test ---")
    
    # Define hyperparams for the test, ensuring learn_alpha is consistent
    test_hp = DEFAULT_SAC_HYPERPARAMS.copy()
    test_hp["learn_alpha"] = True # Explicitly set for the test
    test_hp["batch_size"] = 32   # Smaller batch for faster test
    test_hp["buffer_size"] = 500 # Smaller buffer

    agent = SACAgent(state_dim=N_INSTANCE_FEATURES, action_dim=2, hyperparams=test_hp)
    print(f"Using device: {agent.device}")
    print(f"Learnable alpha: {agent.hp['learn_alpha']}, Initial alpha: {agent.alpha:.4f}")
    if agent.hp['learn_alpha']:
        print(f"Target entropy: {agent.target_entropy:.4f}")

    dummy_s = np.random.rand(N_INSTANCE_FEATURES).astype(np.float32)
    
    action_stochastic = agent.select_action(dummy_s, evaluate=False)
    print(f"\nSelected stochastic action: {action_stochastic}")
    assert action_stochastic.shape == (2,)

    # --- Store some experiences and update parameters ---
    for _ in range(agent.hp["batch_size"] + 10): # Ensure enough for a few updates
        s = np.random.rand(N_INSTANCE_FEATURES).astype(np.float32)
        # Use stochastic actions to populate buffer, as in training
        a = agent.select_action(s, evaluate=False) 
        r = float(np.random.rand())
        s_next = np.random.rand(N_INSTANCE_FEATURES).astype(np.float32)
        d = bool(random.choice([True, False]))
        agent.store_experience(s, a, r, s_next, d)
    
    print(f"\nReplay buffer size before update: {len(agent.replay_buffer)}")

    if len(agent.replay_buffer) >= agent.hp["batch_size"]:
        print("\nAttempting agent parameter update...")
        # Set actor to train mode before update if it was in eval
        agent.actor.train()
        agent.critic.train()
        # critic_target should remain in eval or not require grad

        update_result = agent.update_parameters()
        if update_result:
            c_loss, a_loss, alpha_l = update_result
            print(f"Critic Loss: {c_loss:.4f}")
            print(f"Actor Loss: {a_loss:.4f}")
            if agent.hp['learn_alpha']:
                print(f"Alpha Loss: {alpha_l:.4f}, New Alpha: {agent.alpha:.4f}")
            print("Update successful.")
        else:
            print("Update skipped.")
    else:
        print("Not enough experiences to perform an update.")

    # --- Test save and load ---
    # Get deterministic action from the *updated* agent
    agent.actor.eval() # Set to eval mode for deterministic action
    deterministic_action_before_save = agent.select_action(dummy_s, evaluate=True)
    print(f"\nSelected deterministic action (after update, before save): {deterministic_action_before_save}")


    print("\nAttempting to save and load model...")
    model_save_dir = "./temp_test_models" # Use a temporary specific directory
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        
    agent.save_model(directory=model_save_dir, filename_prefix="test_sac_agent")
    
    # Create a new agent with the same HPs to load into
    new_agent_hp = agent.hp.copy()
    new_agent = SACAgent(state_dim=N_INSTANCE_FEATURES, action_dim=2, hyperparams=new_agent_hp, device=agent.device)
    new_agent.load_model(directory=model_save_dir, filename_prefix="test_sac_agent")
    
    # new_agent.actor is already in eval mode due to load_model()
    action_loaded = new_agent.select_action(dummy_s, evaluate=True) 
    print(f"Action from loaded agent (deterministic): {action_loaded}")
    
    assert np.allclose(deterministic_action_before_save, action_loaded, atol=1e-5), \
        f"Deterministic actions differ after load: Original (updated) {deterministic_action_before_save}, Loaded {action_loaded}"
    print("Save/Load and deterministic action check successful.")

    # Clean up dummy model files and directory
    files_to_remove = ["test_sac_agent_actor.pth", "test_sac_agent_critic.pth", "test_sac_agent_log_alpha.pth"]
    for f_name in files_to_remove:
        f_path = os.path.join(model_save_dir, f_name)
        if os.path.exists(f_path): os.remove(f_path)
    if os.path.exists(model_save_dir) and not os.listdir(model_save_dir): 
        os.rmdir(model_save_dir)
        
    print("\nSACAgent basic test finished.")

