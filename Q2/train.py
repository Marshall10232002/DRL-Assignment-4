# File: Q2/train.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
import sys

# ─── Setup Path to Import make_dmc_env ───
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

# ─── ENV SETUP ───

def make_env():
    env_name = "cartpole-balance"
    env = make_dmc_env(env_name, seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    return env

env = make_env()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = env.action_space.high[0]

# ─── DDPG NETWORKS ───

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

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))

# ─── REPLAY BUFFER ───

class ReplayBuffer:
    def __init__(self, size=int(1e6)):
        self.buffer = deque(maxlen=size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        obs, act, rew, next_obs, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(act, dtype=torch.float32),
            torch.tensor(rew, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1)
        )

# ─── UTILITY ───

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

# ─── TRAINING ───

def train():
    actor = Actor(obs_dim, act_dim, act_limit)
    critic = Critic(obs_dim, act_dim)
    target_actor = Actor(obs_dim, act_dim, act_limit)
    target_critic = Critic(obs_dim, act_dim)
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    replay_buffer = ReplayBuffer()

    batch_size = 256
    gamma = 0.99
    tau = 0.005

    steps = 500_000
    start_steps = 5_000
    update_after = 5_000
    update_every = 50

    obs, _ = env.reset()
    for t in range(steps):
        if t < start_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = actor(torch.tensor(obs, dtype=torch.float32)).numpy()
                action += 0.1 * np.random.randn(act_dim)
                action = np.clip(action, -act_limit, act_limit)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add((obs, action, reward, next_obs, float(done)))
        obs = next_obs

        if done:
            obs, _ = env.reset()

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample(batch_size)
                o, a, r, o2, d = batch

                # Critic update
                with torch.no_grad():
                    target_q = target_critic(o2, target_actor(o2))
                    target = r + gamma * (1 - d) * target_q

                critic_loss = ((critic(o, a) - target)**2).mean()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                actor_loss = -critic(o, actor(o)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update
                soft_update(actor, target_actor, tau)
                soft_update(critic, target_critic, tau)

        if t % 10000 == 0:
            print(f"Step {t} completed")

    if not os.path.exists('Q2_model'):
        os.makedirs('Q2_model')
    torch.save(actor.state_dict(), 'Q2_model/ddpg_actor.pth')

if __name__ == '__main__':
    train()
