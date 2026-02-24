# Listing 9.4 — Evaluation harness with action noise and observation staleness.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any
import random

# from lever_env import LeverEnv, LeverEnvConfig


@dataclass(frozen=True)
class NoiseConfig:
    action_noise_p: float = 0.0  # probability to randomize each agent's action
    obs_stale_p: float = 0.0     # probability to return previous obs for each agent


class NoisyLeverEnv(LeverEnv):
    def __init__(self, cfg: LeverEnvConfig, noise: NoiseConfig, seed: int = 0):
        super().__init__(cfg, seed=seed)
        self.noise = noise
        self._last_obs: Dict[int, Tuple[int, int]] = {}

    def reset(self, seed: int = 0):
        obs, st = super().reset(seed=seed)
        self._last_obs = dict(obs)
        return obs, st

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Tuple[int, int]], float, bool, Dict[str, Any]]:
        actions = dict(actions)
        if self.noise.action_noise_p > 0.0:
            for i in (0, 1):
                if self.rng.random() < self.noise.action_noise_p:
                    actions[i] = self.rng.randrange(4)

        obs, r, done, info = super().step(actions)

        out_obs = dict(obs)
        if self.noise.obs_stale_p > 0.0 and self._last_obs:
            for i in (0, 1):
                if self.rng.random() < self.noise.obs_stale_p:
                    out_obs[i] = self._last_obs.get(i, out_obs[i])

        self._last_obs = dict(obs)
        return out_obs, r, done, info
