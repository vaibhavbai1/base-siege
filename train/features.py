"""Spatial board features fused with the nine resource and battle summaries."""
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from env import config as C


class SiegeFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=C.FEATURE_FUSION_DIM)
        self.grid = nn.Sequential(nn.Conv2d(C.N_GRID_LAYERS, 32, 3, stride=2, padding=1), nn.ReLU(),
                                  nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        self.vector = nn.Sequential(nn.Linear(C.VECTOR_DIM, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(64 * 4 * 4 + 32, C.FEATURE_FUSION_DIM), nn.ReLU())

    def forward(self, observations):
        return self.fusion(torch.cat((self.grid(observations["grid"]),
                                      self.vector(observations["vector"])), dim=1))
