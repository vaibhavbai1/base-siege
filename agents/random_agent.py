"""Uniform sampling over currently legal actions."""
import numpy as np


class RandomAgent:
    def __init__(self, seed=0):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.rng = np.random.default_rng(self.seed)

    def act(self, obs, mask):
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise ValueError("No legal action; the episode may have ended")
        return int(self.rng.choice(legal))
