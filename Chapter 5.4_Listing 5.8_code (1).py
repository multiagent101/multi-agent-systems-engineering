# Listing 5.8 — Recovery meter: rolling throughput and coordination continuity.

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class RecoveryConfig:
    rate_window_s: float = 1.0
    baseline_window_s: float = 2.0
    recovery_fraction: float = 0.9
    consecutive_windows: int = 3


class RecoveryMeter:
    def __init__(self, cfg: RecoveryConfig):
        self.cfg = cfg
        self._completions: Deque[float] = deque()
        self._assignments: Deque[float] = deque()
        self._fault_at: Optional[float] = None
        self._baseline_rate: Optional[float] = None
        self._recovered_at: Optional[float] = None
        self._ok_windows: int = 0

    def record_assignment(self, t: float) -> None:
        self._assignments.append(t)

    def record_completion(self, t: float) -> None:
        self._completions.append(t)

    def mark_fault(self, t: float) -> None:
        self._fault_at = t
        self._baseline_rate = None
        self._recovered_at = None
        self._ok_windows = 0

    def _rate_over(self, events: Deque[float], now: float, window_s: float) -> float:
        cutoff = now - window_s
        while events and events[0] < cutoff:
            events.popleft()
        return len(events) / max(1e-9, window_s)

    def update(self, now: Optional[float] = None) -> Optional[float]:
        if now is None:
            now = time.perf_counter()
        if self._fault_at is None:
            return None

        # Establish baseline from pre-fault completions once.
        if self._baseline_rate is None:
            baseline_now = self._fault_at
            cutoff = baseline_now - self.cfg.baseline_window_s
            baseline_count = 0
            for t in self._completions:
                if cutoff <= t <= baseline_now:
                    baseline_count += 1
            self._baseline_rate = baseline_count / max(1e-9, self.cfg.baseline_window_s)

        if now <= self._fault_at:
            return None

        rate = self._rate_over(self._completions, now, self.cfg.rate_window_s)
        target = self.cfg.recovery_fraction * (self._baseline_rate or 0.0)

        if rate >= target:
            self._ok_windows += 1
            if self._ok_windows >= self.cfg.consecutive_windows and self._recovered_at is None:
                self._recovered_at = now
        else:
            self._ok_windows = 0

        return self._recovered_at
