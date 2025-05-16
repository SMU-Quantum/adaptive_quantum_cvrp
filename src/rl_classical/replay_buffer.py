# cvrp_tripartite_solver/src/rl_classical/replay_buffer.py

import numpy as np
import random
import torch
from typing import Tuple

class ReplayBuffer:
    """
    A simple Replay Buffer for storing and sampling experiences.
    """
    def __init__(self, state_dim: int, action_dim: int, max_size: int = int(1e6), device: torch.device = None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32) # Store 'terminated' flags

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            next_state: np.ndarray, 
            done: bool):
        """
        Adds a new experience to the buffer.
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state observed.
            done: Boolean indicating if the episode terminated.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done) # Store as float

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of experiences from the buffer.
        Args:
            batch_size: The number of experiences to sample.
        Returns:
            A tuple of Tensors: (states, actions, rewards, next_states, dones)
        """
        if self.size < batch_size:
            # Not enough samples yet, could raise an error or return None/empty
            # For simplicity, let's allow sampling with replacement if size < batch_size,
            # or just indicate not ready. Better to wait for enough samples.
            # This part might be handled by the agent's learning condition.
            # For now, let's assume self.size >= batch_size when called by agent.
            indices = np.random.randint(0, self.size, size=batch_size)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.FloatTensor(self.states[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(self.next_states[indices]).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size

if __name__ == '__main__':
    # Example Usage
    state_dim_test = 6  # N_INSTANCE_FEATURES
    action_dim_test = 2 # rho, sigma
    buffer_size_test = 1000
    batch_size_test = 32

    replay_buffer = ReplayBuffer(state_dim_test, action_dim_test, buffer_size_test)

    print(f"Initial buffer size: {len(replay_buffer)}")

    # Add some dummy data
    for i in range(50):
        dummy_state = np.random.rand(state_dim_test).astype(np.float32)
        dummy_action = np.random.rand(action_dim_test).astype(np.float32)
        dummy_reward = float(np.random.rand())
        dummy_next_state = np.random.rand(state_dim_test).astype(np.float32)
        dummy_done = bool(random.choice([True, False]))
        replay_buffer.add(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)

    print(f"Buffer size after adding 50 experiences: {len(replay_buffer)}")
    assert len(replay_buffer) == 50

    # Sample from buffer
    if len(replay_buffer) >= batch_size_test:
        states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size_test)
        print("\nSampled batch shapes:")
        print(f"States: {states_b.shape}")
        print(f"Actions: {actions_b.shape}")
        print(f"Rewards: {rewards_b.shape}")
        print(f"Next States: {next_states_b.shape}")
        print(f"Dones: {dones_b.shape}")

        assert states_b.shape == (batch_size_test, state_dim_test)
        assert actions_b.shape == (batch_size_test, action_dim_test)
        assert rewards_b.shape == (batch_size_test, 1)
        assert next_states_b.shape == (batch_size_test, state_dim_test)
        assert dones_b.shape == (batch_size_test, 1)
        print("Device of sampled tensors:", states_b.device)
    else:
        print(f"Not enough samples ({len(replay_buffer)}) to draw a batch of {batch_size_test}.")

    # Fill buffer and test pointer wrapping
    for i in range(buffer_size_test): # Add more to fill and wrap
        dummy_state = np.random.rand(state_dim_test).astype(np.float32)
        dummy_action = np.random.rand(action_dim_test).astype(np.float32)
        dummy_reward = float(np.random.rand())
        dummy_next_state = np.random.rand(state_dim_test).astype(np.float32)
        dummy_done = bool(random.choice([True, False]))
        replay_buffer.add(dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done)
    
    print(f"Buffer size after filling: {len(replay_buffer)}")
    assert len(replay_buffer) == buffer_size_test
    print(f"Buffer pointer: {replay_buffer.ptr}") # Should have wrapped around

    print("\nReplayBuffer basic test finished.")