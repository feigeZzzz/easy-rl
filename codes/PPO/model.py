"""Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:29:24
LastEditor: John
LastEditTime: 2021-04-08 22:36:43
Discription:
Environment:
"""


import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,
                 hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist


class ActorNet(nn.Module):
    def __init__(self, n_states, bound):
        super(ActorNet, self).__init__()
        self.n_states = n_states
        self.bound = bound

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU()
        )

        self.mu_out = nn.Linear(128, 1)
        self.sigma_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer(x))
        mu = self.bound * torch.tanh(self.mu_out(x))
        sigma = F.softplus(self.sigma_out(x))
        dist = torch.distributions.Normal(mu, sigma)
        return dist


class CriticNet(nn.Module):
    def __init__(self, n_states):
        super(CriticNet, self).__init__()
        self.n_states = n_states

        self.layer = nn.Sequential(
            nn.Linear(self.n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        v = self.layer(x)
        return v


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        value = self.critic(state)
        return value
