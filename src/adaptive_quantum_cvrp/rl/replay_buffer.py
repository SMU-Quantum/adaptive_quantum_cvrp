# src/adaptive_quantum_cvrp/rl/replay_buffer.py

import numpy as np
from typing import Tuple

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for off-policy RL agents.

    Attributes:
        mem_size (int): Maximum size of the buffer.
        mem_counter (int): Current number of stored experiences.
    """
    def __init__(self, mem_size: int, input_dims: int, n_actions: int):
        """
        Initializes the replay buffer.

        Args:
            mem_size: The maximum number of experiences to store.
            input_dims: The dimensionality of the state space.
            n_actions: The dimensionality of the action space.
        """
        self.mem_size = mem_size
        self.mem_counter = 0
        
        # Initialize memory arrays
        self.state_memory = np.zeros((mem_size, input_dims), dtype=np.float32)
        self.next_state_memory = np.zeros((mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros((mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(mem_size, dtype=np.bool_)

    def store_transition(self, state: np.ndarray, action: np.ndarray,
                         reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Stores an experience tuple in the buffer.

        If the buffer is full, the oldest experience is overwritten.
        """
        index = self.mem_counter % self.mem_size
        
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_state_memory[index] = next_state
        self.terminal_memory[index] = done
        
        self.mem_counter += 1

    def sample_buffer(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Samples a random batch of experiences from the buffer.

        Args:
            batch_size: The number of experiences to sample.

        Returns:
            A tuple containing batches of states, actions, rewards,
            next_states, and done flags.
        """
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_states = self.next_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, next_states, dones