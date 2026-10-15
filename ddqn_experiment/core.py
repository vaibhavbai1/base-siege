"""Double DQN: legal exploration, legal bootstrap actions and float32 replay."""
from copy import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env import config as C
from train.features import SiegeFeatures


@dataclass(frozen=True)
class Settings:
    total_steps: int = 4_194_304
    placement_steps: int = 32_768
    buffer_size: int = 50_000
    batch_size: int = 128
    learning_starts: int = 2_048
    train_freq: int = 4
    target_interval: int = 4_096
    learning_rate: float = 0.0001
    gamma: float = 1.0
    max_grad_norm: float = 10.0
    exploration_steps: int = 524_288
    final_epsilon: float = 0.05
    checkpoint_interval: int = 65_536
    log_interval: int = 4_096

    def validate(self):
        counts = (self.total_steps, self.placement_steps, self.buffer_size, self.batch_size,
                  self.learning_starts, self.train_freq, self.target_interval,
                  self.exploration_steps, self.checkpoint_interval, self.log_interval)
        if any(type(v) is not int or v < 1 for v in counts):
            raise ValueError("All count settings must be positive integers")
        if self.total_steps <= self.placement_steps or self.batch_size > self.buffer_size:
            raise ValueError("Need a joint phase, and a buffer at least as large as a batch")
        if not (0 < self.learning_rate < 1 and 0 <= self.gamma <= 1
                and self.max_grad_norm > 0 and 0 <= self.final_epsilon <= 1):
            raise ValueError("Invalid learning or exploration setting")
        if not all(np.isfinite(v) for v in (self.learning_rate, self.gamma,
                                           self.max_grad_norm, self.final_epsilon)):
            raise ValueError("Settings must be finite")


def epsilon_at(step, settings):
    """Separate placement exploration, then restart exploration for shopping."""
    if step < settings.placement_steps:
        return 1.0 - 0.9 * min(step / settings.placement_steps, 1.0)
    fraction = min((step - settings.placement_steps) / settings.exploration_steps, 1.0)
    return 1.0 + fraction * (settings.final_epsilon - 1.0)


def tensor_observation(obs, device, *, batch=False):
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device)
            if batch else torch.as_tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
            for k, v in obs.items()}


class QNetwork(nn.Module):
    def __init__(self, observation_space):
        super().__init__()
        # Same grid/vector feature design as PPO; a Q head replaces actor/critic heads.
        self.features = SiegeFeatures(observation_space)
        self.head = nn.Sequential(nn.Linear(C.FEATURE_FUSION_DIM, C.POLICY_HEAD_DIM),
                                  nn.ReLU(), nn.Linear(C.POLICY_HEAD_DIM, C.N_ACTIONS))

    def forward(self, obs):
        return self.head(self.features(obs))


def masked_argmax(q, mask):
    mask = torch.as_tensor(mask, dtype=torch.bool, device=q.device)
    if mask.shape != q.shape or not bool(mask.any(dim=-1).all()):
        raise ValueError("Every action-selection row must have a legal action")
    return q.masked_fill(~mask, -torch.inf).argmax(dim=-1)


@torch.no_grad()
def choose_action(network, obs, mask, rng, epsilon, device):
    legal = np.flatnonzero(mask)
    if not len(legal):
        raise ValueError("Cannot act in an ended episode")
    if epsilon > 0 and rng.random() < epsilon:
        return int(rng.choice(legal))
    return int(masked_argmax(network(tensor_observation(obs, device)),
                            np.asarray(mask)[None, :]).item())


def bootstrap_mask(env, terminated, truncated):
    if terminated:
        return np.zeros(C.N_ACTIONS, dtype=bool)
    if truncated:
        # The safety guard hides legal actions. A shallow read-only copy exposes
        # the underlying state's continuation mask without changing the live env.
        # The real 300-tick battle deadline is termination, not truncation.
        probe = copy(env)
        probe.truncated = False
        return probe.action_masks()
    return env.action_masks().copy()


@torch.no_grad()
def double_dqn_targets(rewards, terminated, next_online_q, next_target_q, next_masks, gamma):
    """Online net selects; target net evaluates. Ended rows never take an argmax."""
    targets = rewards.clone()
    active = ~terminated.bool()
    if bool(active.any()):
        actions = masked_argmax(next_online_q[active], next_masks[active])
        values = next_target_q[active].gather(1, actions[:, None]).squeeze(1)
        targets[active] += gamma * values
    return targets


class ReplayBuffer:
    """Ring buffer with explicit next states; no lossy image quantisation."""
    def __init__(self, capacity, observation_space):
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.obs = {k: np.empty((capacity, *s.shape), np.float32)
                    for k, s in observation_space.spaces.items()}
        self.next_obs = {k: np.empty_like(v) for k, v in self.obs.items()}
        self.actions = np.empty(capacity, np.int64)
        self.rewards = np.empty(capacity, np.float32)
        self.terminated = np.empty(capacity, bool)
        self.next_masks = np.empty((capacity, C.N_ACTIONS), bool)

    def add(self, obs, action, reward, next_obs, terminated, next_mask):
        if not terminated and not np.any(next_mask):
            raise ValueError("Nonterminal replay transition has no legal next action")
        i = self.pos
        for k in self.obs:
            self.obs[k][i], self.next_obs[k][i] = obs[k], next_obs[k]
        self.actions[i], self.rewards[i] = action, reward
        self.terminated[i], self.next_masks[i] = terminated, next_mask
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device):
        ix = rng.integers(self.size, size=batch_size)
        return (tensor_observation({k: v[ix] for k, v in self.obs.items()}, device, batch=True),
                torch.as_tensor(self.actions[ix], device=device),
                torch.as_tensor(self.rewards[ix], device=device),
                tensor_observation({k: v[ix] for k, v in self.next_obs.items()}, device, batch=True),
                torch.as_tensor(self.terminated[ix], device=device),
                torch.as_tensor(self.next_masks[ix], device=device))

    def state_dict(self):
        # Only initialized rows enter checkpoints. Arrays remain float32.
        return {"capacity": self.capacity, "pos": self.pos, "size": self.size,
                "obs": {k: v[:self.size].copy() for k, v in self.obs.items()},
                "next_obs": {k: v[:self.size].copy() for k, v in self.next_obs.items()},
                **{k: getattr(self, k)[:self.size].copy() for k in
                   ("actions", "rewards", "terminated", "next_masks")}}

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed")
        self.pos, self.size = state["pos"], state["size"]
        for name in ("obs", "next_obs"):
            for k, v in state[name].items():
                getattr(self, name)[k][:self.size] = v
        for k in ("actions", "rewards", "terminated", "next_masks"):
            getattr(self, k)[:self.size] = state[k]


def update(online, target, optimizer, replay, rng, settings, device):
    obs, actions, rewards, next_obs, terminated, masks = replay.sample(settings.batch_size, rng, device)
    with torch.no_grad():
        expected = double_dqn_targets(rewards, terminated, online(next_obs), target(next_obs),
                                      masks, settings.gamma)
    predicted = online(obs).gather(1, actions[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(predicted, expected)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite DQN loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), settings.max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach())
