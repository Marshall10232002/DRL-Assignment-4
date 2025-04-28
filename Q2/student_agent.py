# File: Q2/student_agent.py

import torch
import torch.nn as nn
import numpy as np

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)

class Agent:
    def __init__(self):
        # No need to import or make env here
        obs_dim = 5  # Based on your env setup
        act_dim = 1  # Based on your env setup
        act_limit = 1.0  # Action range is [-1, 1]

        self.device = torch.device('cpu')
        self.actor = Actor(obs_dim, act_dim, act_limit).to(self.device)
        self.actor.load_state_dict(torch.load('Q2_model/ddpg_actor.pth', map_location=self.device))
        self.actor.eval()

    def act(self, observation):
        obs = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy().flatten()
        return action
