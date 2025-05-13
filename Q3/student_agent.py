import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# 1) Ensure everything defaults to float64 (double precision)
torch.set_default_dtype(torch.float64)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.l1       = nn.Linear(obs_dim, 256)
        self.l2       = nn.Linear(256,     256)
        self.mu_head  = nn.Linear(256,     act_dim)
        self.sig_head = nn.Linear(256,     act_dim)

    def forward(self, s: torch.Tensor, deterministic: bool = False):
        x  = F.relu(self.l1(s))
        x  = F.relu(self.l2(x))
        mu = self.mu_head(x)

        if deterministic:
            return torch.tanh(mu)

        # (unused during evaluation)
        log_sigma = torch.clamp(self.sig_head(x), -20.0, 2.0)
        sigma     = torch.exp(log_sigma)
        dist      = Normal(mu, sigma)
        z         = dist.rsample()
        return torch.tanh(z)

class Agent:
    """Humanoid Walk agent using a pretrained SAC actor (double precision)."""
    def __init__(self):
        obs_dim = 67
        act_dim = 21

        # 2) Build the actor and convert to float64
        self.actor = PolicyNet(obs_dim, act_dim).double()
        self.device = torch.device('cpu')

        # 3) Load your float64 weights
        weights_path = os.path.join(os.path.dirname(__file__), 'actor_ep2600.pth')
        state_dict   = torch.load(weights_path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

    def act(self, observation: np.ndarray) -> np.ndarray:
        # 4) Convert obs to float64 tensor
        obs_tensor = torch.tensor(observation, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            a = self.actor(obs_tensor.unsqueeze(0), deterministic=True).cpu().numpy()[0]
        # 5) Clip to [-1, 1] just in case
        return np.clip(a, -1.0, 1.0)