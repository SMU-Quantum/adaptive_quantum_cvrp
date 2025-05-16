# cvrp_tripartite_solver/tests/rl_classical/test_penalty_agent.py

import pytest
import torch
import torch.optim as optim
import numpy as np
import os
from unittest.mock import MagicMock, patch

from rl_classical.penalty_networks import (
    MIN_PENALTY_RHO, MAX_PENALTY_RHO,
    MIN_PENALTY_SIGMA, MAX_PENALTY_SIGMA
)


# Assuming project structure and pyproject.toml allow these imports
from rl_classical.penalty_agent import SACAgent, DEFAULT_SAC_HYPERPARAMS, N_INSTANCE_FEATURES
from rl_classical.penalty_networks import (
    Actor, Critic,
    ACTION_SCALE_RHO, ACTION_BIAS_RHO,
    ACTION_SCALE_SIGMA, ACTION_BIAS_SIGMA
)
from rl_classical.replay_buffer import ReplayBuffer


STATE_DIM_AGENT_TEST = N_INSTANCE_FEATURES
ACTION_DIM_AGENT_TEST = 2 # rho, sigma
DEFAULT_HP_TEST = DEFAULT_SAC_HYPERPARAMS.copy()

@pytest.fixture
def sac_agent_fixture(tmp_path):
    agent = SACAgent(state_dim=STATE_DIM_AGENT_TEST, action_dim=ACTION_DIM_AGENT_TEST, hyperparams=DEFAULT_HP_TEST)
    return agent

@pytest.fixture
def sac_agent_fixed_alpha_fixture():
    hp = DEFAULT_HP_TEST.copy()
    hp["learn_alpha"] = False
    hp["alpha"] = 0.15
    agent = SACAgent(state_dim=STATE_DIM_AGENT_TEST, action_dim=ACTION_DIM_AGENT_TEST, hyperparams=hp)
    return agent


def test_sac_agent_initialization_learn_alpha(sac_agent_fixture):
    agent = sac_agent_fixture
    assert isinstance(agent.actor, Actor)
    assert isinstance(agent.critic, Critic)
    assert isinstance(agent.critic_target, Critic)
    assert isinstance(agent.actor_optimizer, optim.Adam)
    assert isinstance(agent.critic_optimizer, optim.Adam)

    assert agent.hp["learn_alpha"] is True
    assert agent.log_alpha is not None
    assert agent.log_alpha.requires_grad is True
    assert isinstance(agent.alpha_optimizer, optim.Adam)
    assert agent.alpha == pytest.approx(torch.exp(agent.log_alpha).item())
    # Ensure target_entropy is float, as action_dim is int
    expected_target_entropy = -float(ACTION_DIM_AGENT_TEST) * agent.hp["target_entropy_scale"]
    assert agent.target_entropy == pytest.approx(expected_target_entropy)


    for target_param, param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
        assert torch.allclose(target_param.data, param.data)
    assert all(p.requires_grad is False for p in agent.critic_target.parameters())

    assert isinstance(agent.replay_buffer, ReplayBuffer)
    assert agent.replay_buffer.max_size == agent.hp["buffer_size"]

def test_sac_agent_initialization_fixed_alpha(sac_agent_fixed_alpha_fixture):
    agent = sac_agent_fixed_alpha_fixture
    assert agent.hp["learn_alpha"] is False
    assert agent.log_alpha is None
    assert agent.alpha_optimizer is None
    assert agent.alpha == 0.15
    assert agent.target_entropy is None

def test_select_action_stochastic(sac_agent_fixture):
    agent = sac_agent_fixture
    dummy_state = np.random.rand(STATE_DIM_AGENT_TEST).astype(np.float32)
    
    # actor.sample already returns scaled actions. We just check the call and output shape/type.
    with patch.object(agent.actor, 'sample') as mock_actor_sample:
        # Define what the mocked sample should return: (scaled_action_tensor, log_prob_tensor, mean_tensor)
        # Ensure the action tensor is on the correct device for .cpu().data.numpy()
        mock_scaled_action = torch.rand(1, ACTION_DIM_AGENT_TEST, device=agent.device) 
        mock_log_prob = torch.rand(1, 1, device=agent.device)
        mock_mean = torch.rand(1, ACTION_DIM_AGENT_TEST, device=agent.device)
        mock_actor_sample.return_value = (mock_scaled_action, mock_log_prob, mock_mean)
        
        action = agent.select_action(dummy_state, evaluate=False)
        mock_actor_sample.assert_called_once() 
    
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM_AGENT_TEST,)

def test_select_action_deterministic(sac_agent_fixture):
    agent = sac_agent_fixture
    dummy_state = np.random.rand(STATE_DIM_AGENT_TEST).astype(np.float32)

    # Mock only agent.actor.forward for this test
    # It should return (mean_unscaled_tensor, log_std_unscaled_tensor)
    unscaled_mean = torch.tensor([[0.2, -0.3]], device=agent.device) 
    # log_std is not directly used by select_action in eval=True mode, but forward returns it
    dummy_log_std = torch.tensor([[-1.0, -1.0]], device=agent.device) 
    
    with patch.object(agent.actor, 'forward', return_value=(unscaled_mean, dummy_log_std)) as mock_forward:
        action = agent.select_action(dummy_state, evaluate=True)
        mock_forward.assert_called_once() # Check if actor.forward was called
    
    assert isinstance(action, np.ndarray)
    assert action.shape == (ACTION_DIM_AGENT_TEST,)
    
    # Verify scaling logic based on the mocked unscaled_mean
    # ACTION_SCALE_... and ACTION_BIAS_... are imported from penalty_networks
    expected_y_t = torch.tanh(unscaled_mean.cpu()) # Perform ops on CPU for np conversion
    expected_rho_np = (expected_y_t[0, 0].item() * ACTION_SCALE_RHO) + ACTION_BIAS_RHO
    expected_sigma_np = (expected_y_t[0, 1].item() * ACTION_SCALE_SIGMA) + ACTION_BIAS_SIGMA
    
    assert action[0] == pytest.approx(expected_rho_np)
    assert action[1] == pytest.approx(expected_sigma_np)


def test_store_experience(sac_agent_fixture):
    agent = sac_agent_fixture
    with patch.object(agent.replay_buffer, 'add') as mock_buffer_add:
        state = np.random.rand(STATE_DIM_AGENT_TEST)
        action = np.random.rand(ACTION_DIM_AGENT_TEST)
        reward = 1.0
        next_state = np.random.rand(STATE_DIM_AGENT_TEST)
        done = False
        agent.store_experience(state, action, reward, next_state, done)
        mock_buffer_add.assert_called_once_with(state, action, reward, next_state, done)

def test_update_parameters_not_enough_samples(sac_agent_fixture):
    agent = sac_agent_fixture
    agent.replay_buffer.size = agent.hp["batch_size"] - 1
    
    with patch.object(agent.critic_optimizer, 'step') as mock_critic_step, \
         patch.object(agent.actor_optimizer, 'step') as mock_actor_step:
        
        result = agent.update_parameters()
        assert result is None 
        mock_critic_step.assert_not_called()
        mock_actor_step.assert_not_called()

@patch.object(ReplayBuffer, 'sample') 
def test_update_parameters_flow_learn_alpha(mock_buffer_sample, sac_agent_fixture):
    agent = sac_agent_fixture # This fixture already has learn_alpha=True by default
    # Ensure target_entropy is correctly set for this test instance
    agent.target_entropy = -float(ACTION_DIM_AGENT_TEST) * agent.hp["target_entropy_scale"]
    # Ensure log_alpha and its optimizer are properly initialized for a learn_alpha=True agent
    if not agent.hp["learn_alpha"]: # Should not happen with sac_agent_fixture
        agent.hp["learn_alpha"] = True
        agent.log_alpha = torch.zeros(1, requires_grad=True, device=agent.device)
        agent.alpha_optimizer = optim.Adam([agent.log_alpha], lr=agent.hp["lr_alpha"])


    batch_s = torch.randn(agent.hp["batch_size"], STATE_DIM_AGENT_TEST, device=agent.device)
    # Actions in replay buffer are already scaled to environment's action space
    batch_a_rho = torch.rand(agent.hp["batch_size"], 1, device=agent.device) * (MAX_PENALTY_RHO - MIN_PENALTY_RHO) + MIN_PENALTY_RHO
    batch_a_sigma = torch.rand(agent.hp["batch_size"], 1, device=agent.device) * (MAX_PENALTY_SIGMA - MIN_PENALTY_SIGMA) + MIN_PENALTY_SIGMA
    batch_a = torch.cat([batch_a_rho, batch_a_sigma], dim=1)

    batch_r = torch.randn(agent.hp["batch_size"], 1, device=agent.device)
    batch_s_next = torch.randn(agent.hp["batch_size"], STATE_DIM_AGENT_TEST, device=agent.device)
    batch_d = torch.zeros(agent.hp["batch_size"], 1, device=agent.device)
    mock_buffer_sample.return_value = (batch_s, batch_a, batch_r, batch_s_next, batch_d)
    
    agent.replay_buffer.size = agent.hp["batch_size"] 

    with patch.object(agent.actor, 'sample', wraps=agent.actor.sample) as mock_actor_sample, \
         patch.object(agent.critic, 'forward', wraps=agent.critic.forward) as mock_critic_forward, \
         patch.object(agent.critic_target, 'forward', wraps=agent.critic_target.forward) as mock_target_critic_forward, \
         patch.object(agent.critic_optimizer, 'step') as mock_critic_opt_step, \
         patch.object(agent.actor_optimizer, 'step') as mock_actor_opt_step, \
         patch.object(agent.alpha_optimizer, 'step') as mock_alpha_opt_step: # Ensure alpha_optimizer exists

        # Set networks to train mode for the update
        agent.actor.train()
        agent.critic.train()

        update_result = agent.update_parameters()
        assert update_result is not None
        critic_loss, actor_loss, alpha_loss = update_result


        mock_buffer_sample.assert_called_once_with(agent.hp["batch_size"])
        
        assert mock_target_critic_forward.call_count > 0 
        assert mock_critic_forward.call_count > 0      
        mock_critic_opt_step.assert_called_once()
        assert isinstance(critic_loss, float)

        assert mock_actor_sample.call_count >= 2 
        mock_actor_opt_step.assert_called_once()
        assert isinstance(actor_loss, float)

        if agent.hp["learn_alpha"]:
            mock_alpha_opt_step.assert_called_once()
            assert isinstance(alpha_loss, float)
        else: # Should not happen with this fixture
            mock_alpha_opt_step.assert_not_called()


def test_save_and_load_model_learn_alpha(sac_agent_fixture, tmp_path):
    agent1 = sac_agent_fixture
    # Ensure learn_alpha is True for this test
    if not agent1.hp["learn_alpha"]:
        agent1.hp["learn_alpha"] = True
        agent1.target_entropy = -float(ACTION_DIM_AGENT_TEST) * agent1.hp["target_entropy_scale"]
        agent1.log_alpha = torch.zeros(1, requires_grad=True, device=agent1.device) # Re-init if changed
        agent1.alpha_optimizer = optim.Adam([agent1.log_alpha], lr=agent1.hp["lr_alpha"])

    # Set a specific log_alpha to test saving/loading it
    with torch.no_grad():
        agent1.log_alpha.data.fill_(np.log(0.5)) # So alpha will be 0.5
    agent1.alpha = agent1.log_alpha.exp().item()


    model_dir = tmp_path / "test_models_learn_alpha"
    prefix = "test_agent_la"

    with torch.no_grad():
        for param in agent1.actor.parameters(): param.data += 0.123
        original_actor_param_sample = next(agent1.actor.parameters()).clone()
        original_log_alpha_val = agent1.log_alpha.data.clone()


    agent1.save_model(directory=str(model_dir), filename_prefix=prefix)

    assert (model_dir / f"{prefix}_actor.pth").exists()
    assert (model_dir / f"{prefix}_critic.pth").exists()
    assert (model_dir / f"{prefix}_log_alpha.pth").exists()

    # Create new agent with same HPs, important for learn_alpha setting
    agent2_hp = agent1.hp.copy()
    agent2 = SACAgent(state_dim=STATE_DIM_AGENT_TEST, action_dim=ACTION_DIM_AGENT_TEST, hyperparams=agent2_hp, device=agent1.device)
    agent2.load_model(directory=str(model_dir), filename_prefix=prefix)

    loaded_actor_param_sample = next(agent2.actor.parameters())
    assert torch.allclose(original_actor_param_sample, loaded_actor_param_sample)
    
    assert agent2.hp["learn_alpha"] is True
    assert agent2.log_alpha is not None
    assert torch.allclose(original_log_alpha_val, agent2.log_alpha.data)
    assert agent2.log_alpha.requires_grad is True
    assert agent2.alpha == pytest.approx(0.5)
    assert agent2.alpha_optimizer is not None # Check optimizer re-initialization

    for target_param, param in zip(agent2.critic_target.parameters(), agent2.critic.parameters()):
        assert torch.allclose(target_param.data, param.data)

if __name__ == "__main__":
    pytest.main([__file__])
