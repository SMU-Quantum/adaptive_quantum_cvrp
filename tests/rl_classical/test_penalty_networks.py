# cvrp_tripartite_solver/tests/rl_classical/test_penalty_networks.py

import pytest
import torch
import torch.nn as nn

# Assuming your project structure and pyproject.toml allow these imports
from rl_classical.penalty_networks import (
    Actor,
    Critic,
    N_INSTANCE_FEATURES,
    MIN_PENALTY_RHO, MAX_PENALTY_RHO,
    MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA,
    LOG_SIG_MIN, LOG_SIG_MAX
)

# Default parameters for tests
STATE_DIM = N_INSTANCE_FEATURES
ACTION_DIM = 2  # rho and sigma
HIDDEN_DIM = 64 # Using a smaller hidden dim for tests to speed them up slightly
BATCH_SIZE = 4

@pytest.fixture
def actor_network():
    """Fixture to create an Actor network instance."""
    return Actor(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

@pytest.fixture
def critic_network():
    """Fixture to create a Critic network instance."""
    return Critic(state_dim=STATE_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM)

@pytest.fixture
def dummy_state_tensor():
    """Fixture to create a dummy state tensor."""
    return torch.randn(BATCH_SIZE, STATE_DIM)

@pytest.fixture
def dummy_action_tensor():
    """Fixture to create a dummy action tensor (scaled to penalty ranges)."""
    # Create actions that are roughly in the middle of their respective valid ranges
    rho_actions = torch.rand(BATCH_SIZE, 1) * (MAX_PENALTY_RHO - MIN_PENALTY_RHO) + MIN_PENALTY_RHO
    sigma_actions = torch.rand(BATCH_SIZE, 1) * (MAX_PENALTY_SIGMA - MIN_PENALTY_SIGMA) + MIN_PENALTY_SIGMA
    return torch.cat([rho_actions, sigma_actions], dim=1)

# --- Actor Network Tests ---
def test_actor_creation(actor_network):
    assert isinstance(actor_network, nn.Module)
    assert actor_network.fc1.in_features == STATE_DIM
    assert actor_network.mean_layer.out_features == ACTION_DIM
    assert actor_network.log_std_layer.out_features == ACTION_DIM

def test_actor_forward_pass_shapes(actor_network, dummy_state_tensor):
    mean, log_std = actor_network.forward(dummy_state_tensor)
    assert mean.shape == (BATCH_SIZE, ACTION_DIM)
    assert log_std.shape == (BATCH_SIZE, ACTION_DIM)

def test_actor_log_std_clamping(actor_network, dummy_state_tensor):
    # Test with extreme inputs to encourage log_std to go out of bounds if not clamped
    # Though with random init, it's unlikely to hit exact bounds without training.
    # The clamp is in the forward pass itself.
    for _ in range(5): # Run a few times with different random states
        state = torch.randn(BATCH_SIZE, STATE_DIM) * 100 # Larger scale input
        _, log_std = actor_network.forward(state)
        assert torch.all(log_std >= LOG_SIG_MIN)
        assert torch.all(log_std <= LOG_SIG_MAX)

def test_actor_sample_output_shapes(actor_network, dummy_state_tensor):
    action, log_prob, mean = actor_network.sample(dummy_state_tensor)
    assert action.shape == (BATCH_SIZE, ACTION_DIM)
    assert log_prob.shape == (BATCH_SIZE, 1)
    assert mean.shape == (BATCH_SIZE, ACTION_DIM) # mean from the distribution

def test_actor_sample_action_ranges(actor_network, dummy_state_tensor):
    action, _, _ = actor_network.sample(dummy_state_tensor)
    rho_actions = action[:, 0]
    sigma_actions = action[:, 1]

    assert torch.all(rho_actions >= MIN_PENALTY_RHO)
    assert torch.all(rho_actions <= MAX_PENALTY_RHO)
    assert torch.all(sigma_actions >= MIN_PENALTY_SIGMA)
    assert torch.all(sigma_actions <= MAX_PENALTY_SIGMA)

def test_actor_sample_log_prob_validity(actor_network, dummy_state_tensor):
    _, log_prob, _ = actor_network.sample(dummy_state_tensor)
    assert not torch.isnan(log_prob).any()
    assert not torch.isinf(log_prob).any()


# --- Critic Network Tests ---
def test_critic_creation(critic_network):
    assert isinstance(critic_network, nn.Module)
    assert critic_network.fc1_q1.in_features == STATE_DIM + ACTION_DIM
    assert critic_network.fc3_q1.out_features == 1
    assert critic_network.fc1_q2.in_features == STATE_DIM + ACTION_DIM
    assert critic_network.fc3_q2.out_features == 1

def test_critic_forward_pass_shapes(critic_network, dummy_state_tensor, dummy_action_tensor):
    q1, q2 = critic_network.forward(dummy_state_tensor, dummy_action_tensor)
    assert q1.shape == (BATCH_SIZE, 1)
    assert q2.shape == (BATCH_SIZE, 1)

def test_critic_forward_with_actor_output(actor_network, critic_network, dummy_state_tensor):
    # Test if critic can take actions directly from actor's sample method
    sampled_actions, _, _ = actor_network.sample(dummy_state_tensor)
    
    # Detach actions from graph if they were to be used for gradient-based tests later
    # For shape testing, it's fine as is.
    q1, q2 = critic_network.forward(dummy_state_tensor, sampled_actions.detach())
    assert q1.shape == (BATCH_SIZE, 1)
    assert q2.shape == (BATCH_SIZE, 1)
    assert not torch.isnan(q1).any()
    assert not torch.isnan(q2).any()

if __name__ == "__main__":
    # You can run pytest programmatically, but it's usually done from the CLI
    # For a quick check if this file has syntax errors and basic structure:
    print(f"N_INSTANCE_FEATURES: {N_INSTANCE_FEATURES}")
    print(f"MIN_PENALTY_RHO: {MIN_PENALTY_RHO}, MAX_PENALTY_RHO: {MAX_PENALTY_RHO}")

    actor = Actor(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    state = torch.randn(BATCH_SIZE, STATE_DIM)
    action, log_p, m = actor.sample(state)
    print("Actor sample action:", action)
    print("Action ranges:", action.min(dim=0).values, action.max(dim=0).values)

    critic = Critic(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    q1_val, q2_val = critic(state, action)
    print("Critic Q1:", q1_val)