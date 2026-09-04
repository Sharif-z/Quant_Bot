import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.dirichlet import Dirichlet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found. DRL Agent training requires PyTorch. Will run in mock mode.")

class DirichletPolicyNetwork(nn.Module if TORCH_AVAILABLE else object):
    """
    Actor Network outputting concentration parameters (alpha > 0) for a Dirichlet distribution.
    This guarantees outputs sum to 1.0 and lie within (0, 1).
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
            
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Softplus ensures alpha concentration parameters are strictly > 0
        self.softplus = nn.Softplus()

    def forward(self, state):
        x = self.net(state)
        # Add a small epsilon to prevent alpha = 0 (which breaks Dirichlet)
        alpha = self.softplus(x) + 1e-4
        return alpha

class ValueNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Critic Network for estimating state-value V(s)."""
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
            
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent for continuous portfolio allocation.
    """
    def __init__(self, state_dim: int, action_dim: int, lr: float = 3e-4, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        if TORCH_AVAILABLE:
            self.actor = DirichletPolicyNetwork(state_dim, action_dim)
            self.critic = ValueNetwork(state_dim)
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
    def select_action(self, state: np.ndarray, deterministic: bool = False):
        """Samples an action from the Dirichlet policy."""
        if not TORCH_AVAILABLE:
            # Mock behavior: return equal weight vector
            return np.ones(self.action_dim) / self.action_dim, 0.0
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            alphas = self.actor(state_tensor)
            dist = Dirichlet(alphas)
            
            if deterministic:
                # The mean of a Dirichlet distribution is alpha_i / sum(alpha)
                action = alphas / alphas.sum()
            else:
                action = dist.sample()
                
            log_prob = dist.log_prob(action)
            
        return action.squeeze().numpy(), log_prob.item()
        
    def train_step(self, states, actions, old_log_probs, rewards, next_states, dones):
        """
        Executes a single PPO update step.
        (Simplified implementation for structural verification).
        """
        if not TORCH_AVAILABLE:
            return 0.0, 0.0
            
        # Convert to tensors
        states_ts = torch.FloatTensor(np.array(states))
        actions_ts = torch.FloatTensor(np.array(actions))
        rewards_ts = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_ts = torch.FloatTensor(np.array(next_states))
        dones_ts = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Compute Targets
        with torch.no_grad():
            next_values = self.critic(next_states_ts)
            target_values = rewards_ts + self.gamma * next_values * (1 - dones_ts)
            
        # Update Critic
        current_values = self.critic(states_ts)
        critic_loss = nn.MSELoss()(current_values, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update Actor (Simplified REINFORCE-style update for demonstration)
        # Note: A full PPO requires GAE and clipping the probability ratio.
        advantages = (target_values - current_values.detach()).squeeze()
        
        alphas = self.actor(states_ts)
        dist = Dirichlet(alphas)
        new_log_probs = dist.log_prob(actions_ts)
        
        actor_loss = -(new_log_probs * advantages).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
