#!/usr/bin/env python3

import os
import random
import math
import warnings
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dmc import make_dmc_env
import wandb

# ──────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning)

# reproducibility & device
SEED        = 100
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_default_dtype(torch.float64)

# hyper-parameters
MAX_EPISODES    = 1_000_000
MAX_STEPS_EP    = 1_000
REPLAY_SIZE     = 1_000_000
BATCH_SZ        = 512
GAMMA           = 0.99
TAU             = 0.005
LR              = 3e-4
RANDOM_STEPS    = 10_000
TRAIN_START     = 10_000

# ──────────────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, obs, act, rew, next_obs):
        self.buffer.append((obs, act, rew, next_obs))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s,a,r,s1 = zip(*batch)
        to_t = lambda x: torch.tensor(np.array(x), device=DEVICE, dtype=torch.float64)
        return to_t(s), to_t(a), to_t(r), to_t(s1)
    def __len__(self):
        return len(self.buffer)

# ──────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim + act_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x).view(-1)

class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.l1 = nn.Linear(obs_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu_head  = nn.Linear(256, act_dim)
        self.sig_head = nn.Linear(256, act_dim)
    def forward(self, s, deterministic=False, return_logp=False):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)

        if deterministic:
            return torch.tanh(mu), None

        log_sigma = torch.clamp(self.sig_head(x), -20.0, 2.0)
        sigma     = torch.exp(log_sigma)
        dist      = Normal(mu, sigma)
        z         = dist.rsample()
        action    = torch.tanh(z)

        logp = None
        if return_logp:
            logp = dist.log_prob(z).sum(1)
            logp -= (2 * (math.log(2) - z - F.softplus(-2*z))).sum(1)
        return action, logp

# ──────────────────────────────────────────────────────────────────
class SACAgent:
    def __init__(self):
        # environment
        self.env = make_dmc_env("humanoid-walk", SEED, flatten=True, use_pixels=False)
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        # networks
        self.actor   = PolicyNet(obs_dim, act_dim).to(DEVICE)
        self.critic1 = QNetwork(obs_dim, act_dim).to(DEVICE)
        self.critic2 = QNetwork(obs_dim, act_dim).to(DEVICE)
        self.target1 = QNetwork(obs_dim, act_dim).to(DEVICE)
        self.target2 = QNetwork(obs_dim, act_dim).to(DEVICE)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        for p in self.target1.parameters(): p.requires_grad = False
        for p in self.target2.parameters(): p.requires_grad = False

        # optimizers and alpha
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=LR)
        self.opt_q1    = torch.optim.Adam(self.critic1.parameters(), lr=LR)
        self.opt_q2    = torch.optim.Adam(self.critic2.parameters(), lr=LR)
        self.log_alpha = torch.tensor(np.log(0.2), device=DEVICE, requires_grad=True)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=LR)
        self.target_entropy = -act_dim

        # replay buffer
        self.replay = ReplayBuffer(REPLAY_SIZE)

        # W&B
        wandb.init(
            project="drl_hw4",
            name=f"sac_humanoid_seed{SEED}",
            config=dict(gamma=GAMMA, tau=TAU, lr=LR, batch=BATCH_SZ)
        )

    def _soft_update(self, src, tgt):
        for sp, tp in zip(src.parameters(), tgt.parameters()):
            tp.data.mul_(1 - TAU)
            tp.data.add_(TAU * sp.data)

    def _learn_step(self):
        s, a, r, s1 = self.replay.sample(BATCH_SZ)
        # current Q-values
        q1_curr = self.critic1(s, a)
        q2_curr = self.critic2(s, a)

        # compute targets
        with torch.no_grad():
            a1, logp1 = self.actor(s1, return_logp=True)
            q1_next = self.target1(s1, a1)
            q2_next = self.target2(s1, a1)
            q_next  = torch.min(q1_next, q2_next)
            q_target = r + GAMMA * (q_next - torch.exp(self.log_alpha) * logp1)

        # update critics
        loss_q1 = F.mse_loss(q1_curr, q_target)
        loss_q2 = F.mse_loss(q2_curr, q_target)
        self.opt_q1.zero_grad(); loss_q1.backward(); self.opt_q1.step()
        self.opt_q2.zero_grad(); loss_q2.backward(); self.opt_q2.step()

        # freeze critics for actor & alpha
        for p in self.critic1.parameters(): p.requires_grad = False
        for p in self.critic2.parameters(): p.requires_grad = False

        # actor loss
        a_pi, logp_pi = self.actor(s, return_logp=True)
        q_pi = torch.min(self.critic1(s, a_pi), self.critic2(s, a_pi))
        loss_actor = (torch.exp(self.log_alpha).detach() * logp_pi - q_pi).mean()
        self.opt_actor.zero_grad(); loss_actor.backward(); self.opt_actor.step()

        # alpha loss
        loss_alpha = (torch.exp(self.log_alpha) *
                      (-logp_pi - self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); loss_alpha.backward(); self.opt_alpha.step()

        # unfreeze critics
        for p in self.critic1.parameters(): p.requires_grad = True
        for p in self.critic2.parameters(): p.requires_grad = True

        # soft‐update targets
        self._soft_update(self.critic1, self.target1)
        self._soft_update(self.critic2, self.target2)


if __name__ == "__main__":
    # instantiate agent
    agent = SACAgent()

    # training loop in main()
    recent_rewards = deque(maxlen=50)
    for ep in range(1, MAX_EPISODES + 1):
        # reset environment
        obs, _ = agent.env.reset(seed=random.randint(0, 1_000_000))
        state = obs
        ep_reward = 0.0
        step = 0
        done = False

        # one episode
        while not done and step < MAX_STEPS_EP:
            if len(agent.replay) < RANDOM_STEPS:
                action = np.random.uniform(-1, 1, agent.env.action_space.shape)
            else:
                with torch.no_grad():
                    action, _ = agent.actor(
                        torch.tensor(state, device=DEVICE).unsqueeze(0),
                        deterministic=False
                    )
                    action = action.cpu().numpy()[0]

            nxt, rew, done, trunc, _ = agent.env.step(action)
            agent.replay.add(state, action, rew, nxt)

            state = nxt
            ep_reward += rew
            step += 1

            if len(agent.replay) >= TRAIN_START:
                agent._learn_step()

        # logging
        recent_rewards.append(ep_reward)
        wandb.log({
            "episode_reward":  ep_reward,
            "avg50":           float(np.mean(recent_rewards)),
            "episode_length":  step,
            "alpha":           float(torch.exp(agent.log_alpha))
        }, step=ep)

        # periodic save
        if ep % 100 == 0:
            fname = f"actor_ep{ep}.pth"
            torch.save(agent.actor.state_dict(), fname)
            print(f"▶ Saved {fname}")

    # final save
    torch.save(agent.actor.state_dict(), "actor_final.pth")
    print("▶ Training complete, final weights saved.")