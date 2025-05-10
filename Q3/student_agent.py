import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make float64 the default (training code used float64)
torch.set_default_dtype(torch.float64)

# ─────────────────────────── Networks ────────────────────────────────
class ActorNet(nn.Module):
    """Same architecture as in train3.py (256-256, tanh squashed)."""
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, act_dim)        # mean branch
        self.log_sigma = nn.Linear(256, act_dim) # (unused at inference)

    def forward(self, s, deterministic: bool = False):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu = self.mu(x)

        # Evaluation needs a *deterministic* action
        if deterministic:
            return torch.tanh(mu)

        # Stochastic branch (not used during leaderboard runs)
        log_sigma = torch.clamp(self.log_sigma(x), -20.0, 2.0)
        sigma = torch.exp(log_sigma)
        dist = torch.distributions.Normal(mu, sigma)
        z = dist.rsample()
        return torch.tanh(z)

# ────────────────────────── Agent class ──────────────────────────────
class Agent(object):
    """Loads a trained SAC policy and returns actions for Humanoid-Walk."""
    def __init__(self):
        # Dimensions are fixed by the assignment specs
        self.obs_dim = 67
        self.act_dim = 21
        self.device = torch.device("cpu")

        # Build policy network and load weights
        self.actor = ActorNet(self.obs_dim, self.act_dim).to(self.device)
        ckpt = torch.load("sac_checkpoint3.pth", map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()   # set to inference mode

    # ------------------------ helpers -------------------------------
    @staticmethod
    def _flatten_obs(obs):
        """
        The scorer may pass either a dict (DeepMind Control) or a flat np.array.
        This helper converts dict observations into a single flat vector.
        """
        if isinstance(obs, dict):
            parts = []
            for v in obs.values():
                v_arr = np.asarray(v, dtype=np.float64)
                parts.append(v_arr.ravel() if v_arr.shape else v_arr.reshape(1))
            return np.concatenate(parts, axis=0)
        return obs  # already a flat vector

    # -------------------------- API ---------------------------------
    def act(self, observation):
        """
        Parameters
        ----------
        observation : np.ndarray or dict
            The flattened 67-D state vector (or dict from DMC wrapper).

        Returns
        -------
        np.ndarray
            A 21-element action clipped to [-1, 1] as required by the env.
        """
        obs_vec = self._flatten_obs(observation)
        with torch.no_grad():
            s = torch.from_numpy(obs_vec).unsqueeze(0).to(self.device)
            action = self.actor(s, deterministic=True).cpu().numpy()[0]
        # Safety: make sure numerical noise never leaves the range
        return np.clip(action, -1.0, 1.0)
