import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# ─── Define Actor Network ───
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)

# ─── Define Agent ───
class Agent(object):
    def __init__(self):
        # Pendulum-v1: State (3,), Action (1,) in [-2, 2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 3
        self.action_dim = 1
        self.max_action = 2.0

        # Load trained Actor
        self.actor = Actor(self.state_dim, self.action_dim, self.max_action).to(self.device)
        self.actor.load_state_dict(torch.load("ddpg_actor.pth", map_location=self.device))
        self.actor.eval()

    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)  # (1, 3)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        return action.clip(-self.max_action, self.max_action)  # Safe clipping
