# Listing 7.3 — Mitigation mechanisms: token bucket, backoff, ordered acquisition, drift hysteresis.

from __future__ import annotations

import time
import random
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence


class TokenBucket:
    """
    Bounds the rate of an action such as reassignment or re-auctioning.
    """
    def __init__(self, rate_per_s: float, burst: float):
        self.rate = max(0.0, rate_per_s)
        self.burst = max(0.0, burst)
        self.tokens = self.burst
        self.last = time.perf_counter()

    def allow(self, cost: float = 1.0) -> bool:
        now = time.perf_counter()
        dt = max(0.0, now - self.last)
        self.last = now
        self.tokens = min(self.burst, self.tokens + dt * self.rate)
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


class Backoff:
    """
    Exponential backoff with jitter for retries.
    """
    def __init__(self, base_s: float = 0.01, max_s: float = 1.0, jitter: float = 0.2, seed: int = 0):
        self.base_s = max(1e-6, base_s)
        self.max_s = max(self.base_s, max_s)
        self.jitter = max(0.0, jitter)
        self.rng = random.Random(seed)

    def delay(self, attempt: int) -> float:
        d = min(self.max_s, self.base_s * (2 ** max(0, attempt)))
        if self.jitter > 0:
            d *= (1.0 + self.rng.uniform(-self.jitter, self.jitter))
        return max(0.0, d)


def ordered_resources(resources: Sequence[str]) -> Sequence[str]:
    """
    Enforces a deterministic total order for acquisition to prevent cycles.
    """
    return tuple(sorted(resources))


@dataclass
class DriftPolicy:
    """
    Hysteresis policy for adapting a parameter based on repeated drift alerts.
    """
    min_interval_s: float = 10.0         # minimum time between updates
    required_consecutive: int = 3         # drift alerts required
    factor_up: float = 1.25               # multiplicative increase
    factor_down: float = 0.90             # multiplicative decrease
    clamp_min: float = 0.001
    clamp_max: float = 5.0


class DriftController:
    def __init__(self, initial: float, policy: DriftPolicy):
        self.value = initial
        self.policy = policy
        self._last_update_at = 0.0
        self._consecutive = 0

    def on_drift_alert(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.perf_counter()
        self._consecutive += 1
        if self._consecutive < self.policy.required_consecutive:
            return self.value
        if (now - self._last_update_at) < self.policy.min_interval_s:
            return self.value
        self._last_update_at = now
        self._consecutive = 0
        self.value = min(self.policy.clamp_max, max(self.policy.clamp_min, self.value * self.policy.factor_up))
        return self.value

    def on_stable_interval(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.perf_counter()
        # Slow decay toward lower values when stability persists.
        if (now - self._last_update_at) < self.policy.min_interval_s:
            return self.value
        self._consecutive = 0
        self.value = min(self.policy.clamp_max, max(self.policy.clamp_min, self.value * self.policy.factor_down))
        self._last_update_at = now
        return self.value
