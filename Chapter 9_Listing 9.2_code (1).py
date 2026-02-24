# Listing 9.2 — Tabular VDN training with CTDE, target tables, and replay.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import random
from collections import deque


Obs = Tuple[int, int]  # (pos, remaining_steps)
Action = int


@dataclass(frozen=True)
class VDNConfig:
    gamma: float = 0.95
    alpha: float = 0.10
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    replay_capacity: int = 5000
    batch_size: int = 64
    target_sync_interval: int = 200
    min_replay: int = 500


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)

    def push(self, tr) -> None:
        self.buf.append(tr)

    def sample(self, rng: random.Random, batch_size: int):
        n = len(self.buf)
        if n == 0:
            return []
        if batch_size >= n:
            return list(self.buf)
        idxs = rng.sample(range(n), batch_size)
        return [self.buf[i] for i in idxs]

    def __len__(self) -> int:
        return len(self.buf)


class TabularQ:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.q: Dict[Tuple[Obs, Action], float] = {}

    def get(self, obs: Obs, a: Action) -> float:
        return self.q.get((obs, a), 0.0)

    def set(self, obs: Obs, a: Action, v: float) -> None:
        self.q[(obs, a)] = v

    def argmax(self, obs: Obs, rng: random.Random) -> Action:
        best: List[Action] = []
        best_v = None
        for a in range(self.n_actions):
            v = self.get(obs, a)
            if best_v is None or v > best_v:
                best_v = v
                best = [a]
            elif v == best_v:
                best.append(a)
        return rng.choice(best) if best else rng.randrange(self.n_actions)

    def copy_from(self, other: "TabularQ") -> None:
        self.q = dict(other.q)


class VDNTrainer:
    """
    Two-agent VDN with additive value decomposition:
        Q_tot(o1,o2,a1,a2) = Q1(o1,a1) + Q2(o2,a2)
    Training uses global reward and joint maximization over next actions (CTDE).
    Execution uses local epsilon-greedy on Q_i.
    """
    def __init__(self, n_actions: int, cfg: VDNConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.q1 = TabularQ(n_actions)
        self.q2 = TabularQ(n_actions)
        self.tgt1 = TabularQ(n_actions)
        self.tgt2 = TabularQ(n_actions)
        self.tgt1.copy_from(self.q1)
        self.tgt2.copy_from(self.q2)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.step_count = 0

    def epsilon(self) -> float:
        # Linear decay schedule over epsilon_decay_steps
        s = min(self.step_count, self.cfg.epsilon_decay_steps)
        frac = s / max(1, self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def act(self, obs1: Obs, obs2: Obs) -> Tuple[Action, Action]:
        eps = self.epsilon()
        if self.rng.random() < eps:
            a1 = self.rng.randrange(self.q1.n_actions)
        else:
            a1 = self.q1.argmax(obs1, self.rng)

        if self.rng.random() < eps:
            a2 = self.rng.randrange(self.q2.n_actions)
        else:
            a2 = self.q2.argmax(obs2, self.rng)
        return a1, a2

    def push_transition(
        self,
        obs1: Obs,
        obs2: Obs,
        a1: Action,
        a2: Action,
        r: float,
        nobs1: Obs,
        nobs2: Obs,
        done: bool,
    ):
        self.replay.push((obs1, obs2, a1, a2, r, nobs1, nobs2, done))

    def train_step(self) -> None:
        self.step_count += 1
        if len(self.replay) < self.cfg.min_replay:
            if self.step_count % self.cfg.target_sync_interval == 0:
                self._sync_targets()
            return

        batch = self.replay.sample(self.rng, self.cfg.batch_size)

        for (o1, o2, a1, a2, r, no1, no2, done) in batch:
            # Current joint estimate
            q_tot = self.q1.get(o1, a1) + self.q2.get(o2, a2)

            # Next joint max using target tables (CTDE: joint maximization)
            if done:
                target = r
            else:
                best_next = None
                for na1 in range(self.q1.n_actions):
                    for na2 in range(self.q2.n_actions):
                        v = self.tgt1.get(no1, na1) + self.tgt2.get(no2, na2)
                        if best_next is None or v > best_next:
                            best_next = v
                target = r + self.cfg.gamma * (best_next if best_next is not None else 0.0)

            td = target - q_tot

            # Shared TD error reflects the additive credit assignment assumption.
            self.q1.set(o1, a1, self.q1.get(o1, a1) + self.cfg.alpha * td)
            self.q2.set(o2, a2, self.q2.get(o2, a2) + self.cfg.alpha * td)

        if self.step_count % self.cfg.target_sync_interval == 0:
            self._sync_targets()

    def _sync_targets(self) -> None:
        self.tgt1.copy_from(self.q1)
        self.tgt2.copy_from(self.q2)
