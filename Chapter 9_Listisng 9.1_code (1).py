# Listing 9.1 — Minimal two-agent coordination environment (lever task).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional
import random


@dataclass(frozen=True)
class LeverEnvConfig:
    line_length: int = 7
    lever_pos: int = 3
    max_steps: int = 15
    success_reward: float = 1.0
    step_penalty: float = 0.0  # negative values encourage faster completion


class LeverEnv:
    """
    Two-agent environment with partial observability.

    Actions:
      0 = left
      1 = stay
      2 = right
      3 = pull
    """
    LEFT, STAY, RIGHT, PULL = 0, 1, 2, 3

    def __init__(self, cfg: LeverEnvConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.t: int = 0
        self.pos = {0: 0, 1: cfg.line_length - 1}
        self.done: bool = False

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, Tuple[int, int]], Tuple[int, int, int]]:
        if seed is not None:
            self.rng.seed(seed)
        self.t = 0
        # deterministic reset endpoints; randomness can be added via seed if desired
        self.pos = {0: 0, 1: self.cfg.line_length - 1}
        self.done = False
        obs = self._obs()
        return obs, self._state()

    def _state(self) -> Tuple[int, int, int]:
        return (self.pos[0], self.pos[1], self.t)

    def _obs(self) -> Dict[int, Tuple[int, int]]:
        # local observation: (own_position, remaining_steps)
        remaining = max(0, self.cfg.max_steps - self.t)
        return {i: (self.pos[i], remaining) for i in (0, 1)}

    def step(self, actions: Dict[int, int]) -> Tuple[Dict[int, Tuple[int, int]], float, bool, Dict[str, Any]]:
        if self.done:
            return self._obs(), 0.0, True, {"success": False}

        a0 = actions.get(0, self.STAY)
        a1 = actions.get(1, self.STAY)

        # Movement
        self.pos[0] = self._apply_move(self.pos[0], a0)
        self.pos[1] = self._apply_move(self.pos[1], a1)

        # Reward and termination
        reward = self.cfg.step_penalty
        success = False

        if (
            a0 == self.PULL
            and a1 == self.PULL
            and self.pos[0] == self.cfg.lever_pos
            and self.pos[1] == self.cfg.lever_pos
        ):
            reward += self.cfg.success_reward
            success = True
            self.done = True

        self.t += 1
        if self.t >= self.cfg.max_steps:
            self.done = True

        return self._obs(), reward, self.done, {"success": success}

    def _apply_move(self, pos: int, action: int) -> int:
        if action == self.LEFT:
            return max(0, pos - 1)
        if action == self.RIGHT:
            return min(self.cfg.line_length - 1, pos + 1)
        # stay or pull: no movement
        return pos
