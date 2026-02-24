from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Set, Tuple, List


@dataclass(frozen=True)
class QueueSample:
    t: float
    name: str
    value: int


@dataclass(frozen=True)
class ResourceEvent:
    t: float
    agent_id: str
    kind: str          # "acquire" | "release" | "wait"
    resource_id: str


@dataclass(frozen=True)
class MetricSample:
    t: float
    name: str
    value: float


class OscillationDetector:
    """
    Flags sustained oscillation in an integer-valued signal using sign changes
    and coefficient-of-variation thresholds over a rolling window.
    """
    def __init__(self, window_s: float = 5.0, min_samples: int = 30,
                 min_sign_changes: int = 10, min_cv: float = 0.25):
        self.window_s = window_s
        self.min_samples = min_samples
        self.min_sign_changes = min_sign_changes
        self.min_cv = min_cv
        self._series: Dict[str, Deque[Tuple[float, int]]] = defaultdict(deque)

    def add(self, sample: QueueSample) -> Optional[str]:
        s = self._series[sample.name]
        s.append((sample.t, sample.value))
        cutoff = sample.t - self.window_s
        while s and s[0][0] < cutoff:
            s.popleft()

        if len(s) < self.min_samples:
            return None

        vals = [v for _, v in s]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        cv = (var ** 0.5) / max(1.0, mean)

        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
        signs = [0 if d == 0 else (1 if d > 0 else -1) for d in diffs]
        sign_changes = 0
        last = 0
        for sgn in signs:
            if sgn == 0:
                continue
            if last != 0 and sgn != last:
                sign_changes += 1
            last = sgn

        if sign_changes >= self.min_sign_changes and cv >= self.min_cv:
            return f"oscillation suspected: name={sample.name} sign_changes={sign_changes} cv={cv:.3f}"
        return None


class DeadlockDetector:
    """
    Maintains a wait-for graph from resource events and detects cycles that persist.
    Resources are assumed exclusive for the purpose of cycle detection.
    """
    def __init__(self, persist_s: float = 2.0):
        self.persist_s = persist_s
        self._holder: Dict[str, str] = {}  # resource_id -> agent_id
        self._waiting: Dict[str, str] = {} # agent_id -> resource_id
        self._first_seen_cycle_at: Optional[float] = None
        self._last_cycle: Optional[Tuple[str, ...]] = None

    def add(self, ev: ResourceEvent) -> Optional[str]:
        if ev.kind == "acquire":
            self._holder[ev.resource_id] = ev.agent_id
            if self._waiting.get(ev.agent_id) == ev.resource_id:
                self._waiting.pop(ev.agent_id, None)
        elif ev.kind == "release":
            if self._holder.get(ev.resource_id) == ev.agent_id:
                self._holder.pop(ev.resource_id, None)
        elif ev.kind == "wait":
            self._waiting[ev.agent_id] = ev.resource_id

        cycle = self._detect_cycle()
        now = ev.t
        if cycle is None:
            self._first_seen_cycle_at = None
            self._last_cycle = None
            return None

        if self._last_cycle != cycle:
            self._last_cycle = cycle
            self._first_seen_cycle_at = now
            return None

        if self._first_seen_cycle_at is not None and (now - self._first_seen_cycle_at) >= self.persist_s:
            cyc = "->".join(list(cycle) + [cycle[0]])
            return f"deadlock suspected: cycle={cyc} persist_s={(now - self._first_seen_cycle_at):.3f}"
        return None

    def _detect_cycle(self) -> Optional[Tuple[str, ...]]:
        # Build wait-for edges: agent -> holder(agent) for the resource it waits on.
        edges: Dict[str, str] = {}
        for a, r in self._waiting.items():
            h = self._holder.get(r)
            if h is not None and h != a:
                edges[a] = h

        # Detect any cycle by following pointers.
        visited_global: Set[str] = set()
        for start in edges.keys():
            if start in visited_global:
                continue
            path: List[str] = []
            seen_local: Dict[str, int] = {}
            cur = start
            while cur in edges:
                if cur in seen_local:
                    i = seen_local[cur]
                    cyc_nodes = path[i:]
                    if not cyc_nodes:
                        return None
                    # Canonicalize rotation for stable persistence checks.
                    n = len(cyc_nodes)
                    rots = [tuple(cyc_nodes[j:] + cyc_nodes[:j]) for j in range(n)]
                    return min(rots)
                seen_local[cur] = len(path)
                path.append(cur)
                visited_global.add(cur)
                cur = edges[cur]
        return None


class DriftDetector:
    """
    Detects mean/variance drift using two rolling windows.
    Intended for signals such as service time, message delay, bid values.
    """
    def __init__(self, baseline_s: float = 10.0, recent_s: float = 3.0,
                 min_samples: int = 50, mean_shift_sigma: float = 3.0):
        self.baseline_s = baseline_s
        self.recent_s = recent_s
        self.min_samples = min_samples
        self.mean_shift_sigma = mean_shift_sigma
        self._series: Dict[str, Deque[Tuple[float, float]]] = defaultdict(deque)

    def add(self, sample: MetricSample) -> Optional[str]:
        s = self._series[sample.name]
        s.append((sample.t, sample.value))
        cutoff = sample.t - self.baseline_s
        while s and s[0][0] < cutoff:
            s.popleft()

        # Split series into baseline and recent windows.
        recent_cutoff = sample.t - self.recent_s
        baseline_vals = [v for t, v in s if t < recent_cutoff]
        recent_vals = [v for t, v in s if t >= recent_cutoff]

        if len(baseline_vals) < self.min_samples or len(recent_vals) < max(10, self.min_samples // 5):
            return None

        b_mean = sum(baseline_vals) / len(baseline_vals)
        b_var = sum((v - b_mean) ** 2 for v in baseline_vals) / len(baseline_vals)
        b_std = max(1e-9, b_var ** 0.5)
        r_mean = sum(recent_vals) / len(recent_vals)

        z = abs(r_mean - b_mean) / b_std
        if z >= self.mean_shift_sigma:
            return f"drift suspected: name={sample.name} baseline_mean={b_mean:.6f} recent_mean={r_mean:.6f} z={z:.2f}"
        return None
