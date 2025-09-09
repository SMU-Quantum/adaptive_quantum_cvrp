# src/adaptive_quantum_cvrp/rl/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from .replay_buffer import ReplayBuffer

class CriticNetwork(nn.Module):
    """The Critic network evaluates state-action pairs."""
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(input_dims + n_actions, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

    def forward(self, state, action):
        # To handle actions being scaled, we normalize them back to [-1, 1] for the critic
        # Assuming max_action is symmetrical, e.g., action_space is [-max, +max]
        # Our action space is [0.1, 10.0], so we will just pass it as is for simplicity here.
        return self.q(torch.cat([state, action], dim=1))

class ActorNetwork(nn.Module):
    """The Actor network (policy) decides which action to take."""
    def __init__(self, input_dims, n_actions, fc1_dims=256, fc2_dims=256):
        super().__init__()
        self.reparam_noise = 1e-6

        self.pi = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU()
        )
        self.mu = nn.Linear(fc2_dims, n_actions)
        self.log_sigma = nn.Linear(fc2_dims, n_actions)

    def forward(self, state):
        prob = self.pi(state)
        mu = self.mu(prob)
        log_sigma = self.log_sigma(prob)
        
        # Clamp log_sigma for stability
        log_sigma = torch.clamp(log_sigma, min=-20, max=2)
        
        return mu, log_sigma

    def sample(self, state, reparameterize=True):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp() # Guarantees sigma is positive
        
        probabilities = Normal(mu, sigma)
        
        if reparameterize:
            # Sample with reparameterization trick for backpropagation
            u = probabilities.rsample()
        else:
            u = probabilities.sample()
        
        # Apply the tanh squashing function to bound the action between -1 and 1
        action_tanh = torch.tanh(u)
        
        # Calculate log probability, correcting for the tanh transformation
        log_probs = probabilities.log_prob(u) - torch.log(1 - action_tanh.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return action_tanh, log_probs

class SACAgent:
    """Soft Actor-Critic Agent."""
    def __init__(self, input_dims, n_actions, action_space, lr=3e-4, gamma=0.99, tau=0.005):
        self.gamma = gamma
        self.tau = tau
        # self.action_space = action_space
        self.memory = ReplayBuffer(1_000_000, input_dims, n_actions)
        

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        

        self.actor = ActorNetwork(input_dims, n_actions)
        self.critic_1 = CriticNetwork(input_dims, n_actions)
        self.critic_2 = CriticNetwork(input_dims, n_actions)
        self.target_critic_1 = CriticNetwork(input_dims, n_actions)
        self.target_critic_2 = CriticNetwork(input_dims, n_actions)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -torch.prod(torch.Tensor((n_actions,))).item()

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = torch.tensor(np.array([observation]), dtype=torch.float32)
        action_tanh, _ = self.actor.sample(state, reparameterize=False)
        
        # Scale the action using the tensor attributes
        scaled_action = action_tanh * self.action_scale + self.action_bias
        
        return scaled_action.detach().numpy()[0]

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self, batch_size=256):
        if self.memory.mem_counter < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        with torch.no_grad():
            next_actions_tanh, next_log_probs = self.actor.sample(next_states)
            # Scale next actions for critic input using the tensor attributes
            next_actions_scaled = next_actions_tanh * self.action_scale + self.action_bias
            
            q1_next_target = self.target_critic_1(next_states, next_actions_scaled)
            q2_next_target = self.target_critic_2(next_states, next_actions_scaled)
            q_next_target = torch.min(q1_next_target, q2_next_target) - self.log_alpha.exp() * next_log_probs
            target = rewards.unsqueeze(1) + self.gamma * q_next_target * (~dones).unsqueeze(1)

        q1 = self.critic_1(states, actions)
        q2 = self.critic_2(states, actions)
        critic_1_loss = F.mse_loss(q1, target)
        critic_2_loss = F.mse_loss(q2, target)

        self.critic_1_optimizer.zero_grad(); self.critic_2_optimizer.zero_grad()
        (critic_1_loss + critic_2_loss).backward()
        self.critic_1_optimizer.step(); self.critic_2_optimizer.step()

        new_actions_tanh, log_probs = self.actor.sample(states, reparameterize=True)
        # Scale new actions for critic input using the tensor attributes
        new_actions_scaled = new_actions_tanh * self.action_scale + self.action_bias

        q1_new = self.critic_1(states, new_actions_scaled)
        q2_new = self.critic_2(states, new_actions_scaled)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.log_alpha.exp() * log_probs - q_new).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        alpha_loss = -(self.log_alpha.exp() * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None: tau = self.tau
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)