import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


# === 1. Residual Block ===
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


# === 2. Feature Extractor Used by PPO ===
class CustomMLPExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim=64):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        hidden_dim = 64
        num_blocks = 3

        self.input = nn.Linear(input_dim, hidden_dim)
        self.resblocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(num_blocks)])
        self.output = nn.Linear(hidden_dim, features_dim)

    def forward(self, x):
        x = torch.relu(self.input(x))
        x = self.resblocks(x)
        return self.output(x)
