# cvrp_tripartite_solver/tests/rl_classical/test_replay_buffer.py

import pytest
import numpy as np
import torch
import random

# Assuming your project structure and pyproject.toml allow these imports
from rl_classical.replay_buffer import ReplayBuffer

# Constants for testing
STATE_DIM_TEST = 6
ACTION_DIM_TEST = 2
BUFFER_SIZE_TEST = 100
BATCH_SIZE_TEST = 16

@pytest.fixture
def replay_buffer_fixture():
    """Fixture to create a ReplayBuffer instance."""
    return ReplayBuffer(state_dim=STATE_DIM_TEST, action_dim=ACTION_DIM_TEST, max_size=BUFFER_SIZE_TEST)

@pytest.fixture
def filled_replay_buffer_fixture():
    """Fixture to create a ReplayBuffer instance filled with some data."""
    buffer = ReplayBuffer(state_dim=STATE_DIM_TEST, action_dim=ACTION_DIM_TEST, max_size=BUFFER_SIZE_TEST)
    for _ in range(BATCH_SIZE_TEST * 2): # Fill more than a batch
        state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        action = np.random.rand(ACTION_DIM_TEST).astype(np.float32)
        reward = float(np.random.rand())
        next_state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        done = bool(random.choice([True, False]))
        buffer.add(state, action, reward, next_state, done)
    return buffer

def test_replay_buffer_initialization(replay_buffer_fixture):
    buffer = replay_buffer_fixture
    assert buffer.max_size == BUFFER_SIZE_TEST
    assert buffer.ptr == 0
    assert buffer.size == 0
    assert buffer.states.shape == (BUFFER_SIZE_TEST, STATE_DIM_TEST)
    assert buffer.actions.shape == (BUFFER_SIZE_TEST, ACTION_DIM_TEST)
    assert buffer.rewards.shape == (BUFFER_SIZE_TEST, 1)
    assert buffer.next_states.shape == (BUFFER_SIZE_TEST, STATE_DIM_TEST)
    assert buffer.dones.shape == (BUFFER_SIZE_TEST, 1)
    assert buffer.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_replay_buffer_add_single_experience(replay_buffer_fixture):
    buffer = replay_buffer_fixture
    state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
    action = np.random.rand(ACTION_DIM_TEST).astype(np.float32)
    reward = 0.5
    next_state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
    done = False

    buffer.add(state, action, reward, next_state, done)

    assert buffer.size == 1
    assert buffer.ptr == 1
    assert np.allclose(buffer.states[0], state)
    assert np.allclose(buffer.actions[0], action)
    assert buffer.rewards[0] == reward
    assert np.allclose(buffer.next_states[0], next_state)
    assert buffer.dones[0] == float(done)

def test_replay_buffer_add_multiple_experiences_and_len(replay_buffer_fixture):
    buffer = replay_buffer_fixture
    num_experiences = 10
    for i in range(num_experiences):
        state = np.random.rand(STATE_DIM_TEST).astype(np.float32) * (i + 1)
        action = np.random.rand(ACTION_DIM_TEST).astype(np.float32) * (i + 1)
        reward = float(i)
        next_state = np.random.rand(STATE_DIM_TEST).astype(np.float32) * (i + 1)
        done = bool(i % 2 == 0)
        buffer.add(state, action, reward, next_state, done)

    assert len(buffer) == num_experiences
    assert buffer.size == num_experiences
    assert buffer.ptr == num_experiences
    assert buffer.rewards[num_experiences - 1] == float(num_experiences - 1)

def test_replay_buffer_wrapping(replay_buffer_fixture):
    buffer = replay_buffer_fixture # max_size is BUFFER_SIZE_TEST (100)
    for i in range(BUFFER_SIZE_TEST + 10): # Add more than max_size
        state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        action = np.random.rand(ACTION_DIM_TEST).astype(np.float32)
        reward = float(i)
        next_state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        done = False
        buffer.add(state, action, reward, next_state, done)

    assert buffer.size == BUFFER_SIZE_TEST
    assert buffer.ptr == 10 # (BUFFER_SIZE_TEST + 10) % BUFFER_SIZE_TEST
    # Check if the latest data is at ptr-1 (index 9)
    assert buffer.rewards[9] == float(BUFFER_SIZE_TEST + 9)
    # Check if the oldest data (index 10) was overwritten by data from iteration BUFFER_SIZE_TEST + 10
    assert buffer.rewards[10] == float(10) # Original 11th item (index 10) was overwritten

def test_replay_buffer_sample_when_full(filled_replay_buffer_fixture):
    buffer = filled_replay_buffer_fixture # Filled with BATCH_SIZE_TEST * 2 experiences
    
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE_TEST)

    assert states.shape == (BATCH_SIZE_TEST, STATE_DIM_TEST)
    assert actions.shape == (BATCH_SIZE_TEST, ACTION_DIM_TEST)
    assert rewards.shape == (BATCH_SIZE_TEST, 1)
    assert next_states.shape == (BATCH_SIZE_TEST, STATE_DIM_TEST)
    assert dones.shape == (BATCH_SIZE_TEST, 1)

    assert states.device == buffer.device
    assert actions.device == buffer.device
    assert rewards.device == buffer.device
    assert next_states.device == buffer.device
    assert dones.device == buffer.device

    assert states.dtype == torch.float32
    assert actions.dtype == torch.float32
    assert rewards.dtype == torch.float32
    assert next_states.dtype == torch.float32
    assert dones.dtype == torch.float32


def test_replay_buffer_sample_when_not_enough_strict_no_replace(replay_buffer_fixture):
    buffer = replay_buffer_fixture
    # Add fewer than batch_size experiences
    num_to_add = BATCH_SIZE_TEST // 2
    for _ in range(num_to_add):
        state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        action = np.random.rand(ACTION_DIM_TEST).astype(np.float32)
        reward = float(np.random.rand())
        next_state = np.random.rand(STATE_DIM_TEST).astype(np.float32)
        done = False
        buffer.add(state, action, reward, next_state, done)

    # The current ReplayBuffer.sample() samples with replacement if size < batch_size
    # So, this test will verify that behavior.
    states, actions, _, _, _ = buffer.sample(BATCH_SIZE_TEST)
    assert states.shape[0] == BATCH_SIZE_TEST
    assert actions.shape[0] == BATCH_SIZE_TEST
    assert buffer.size == num_to_add


def test_replay_buffer_device_setting():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        buffer_cuda = ReplayBuffer(STATE_DIM_TEST, ACTION_DIM_TEST, BUFFER_SIZE_TEST, device=device)
        assert buffer_cuda.device == device

        # Add one item and sample to check tensor device
        buffer_cuda.add(np.random.rand(STATE_DIM_TEST).astype(np.float32),
                        np.random.rand(ACTION_DIM_TEST).astype(np.float32),
                        0.0,
                        np.random.rand(STATE_DIM_TEST).astype(np.float32),
                        False)
        s, _, _, _, _ = buffer_cuda.sample(1)
        assert s.device == device

    device_cpu = torch.device("cpu")
    buffer_cpu = ReplayBuffer(STATE_DIM_TEST, ACTION_DIM_TEST, BUFFER_SIZE_TEST, device=device_cpu)
    assert buffer_cpu.device == device_cpu
    buffer_cpu.add(np.random.rand(STATE_DIM_TEST).astype(np.float32),
                   np.random.rand(ACTION_DIM_TEST).astype(np.float32),
                   0.0,
                   np.random.rand(STATE_DIM_TEST).astype(np.float32),
                   False)
    s_cpu, _, _, _, _ = buffer_cpu.sample(1)
    assert s_cpu.device == device_cpu

if __name__ == "__main__":
    # This block allows running pytest on this file directly if needed,
    # e.g., `python tests/rl_classical/test_replay_buffer.py`
    # though `pytest` command from root is standard.
    pytest.main([__file__])
