# Listing 6.2 — Inconsistency injection wrapper for SharedState.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional
import random


@dataclass
class InconsistencyPlan:
    stale_read_probability: float = 0.0
    drop_write_probability: float = 0.0
    duplicate_write_probability: float = 0.0
    max_stale_versions: int = 1


class FaultySharedState:
    def __init__(self, inner, rng: random.Random, plan: InconsistencyPlan):
        self.inner = inner
        self.rng = rng
        self.plan = plan

    def read(self, key: str, *, at_seq: Optional[int] = None):
        rr = self.inner.read(key, at_seq=at_seq)
        if not rr.versions:
            return rr

        if (
            self.plan.stale_read_probability > 0.0
            and self.rng.random() < self.plan.stale_read_probability
        ):
            max_seq = max(vv.version.seq for vv in rr.versions)
            stale_seq = max(0, max_seq - self.plan.max_stale_versions)
            return self.inner.read(key, at_seq=stale_seq)

        return rr

    def put(
        self,
        key: str,
        value: Any,
        *,
        node_id: str,
        expected_version=None,
        base_vc=None,
    ):
        if (
            self.plan.drop_write_probability > 0.0
            and self.rng.random() < self.plan.drop_write_probability
        ):
            rr = self.inner.read(key)
            if rr.versions:
                return max(rr.versions, key=lambda vv: vv.version.seq).version
            raise ValueError("drop_write on missing key")

        ver = self.inner.put(
            key,
            value,
            node_id=node_id,
            expected_version=expected_version,
            base_vc=base_vc,
        )

        if (
            self.plan.duplicate_write_probability > 0.0
            and self.rng.random() < self.plan.duplicate_write_probability
        ):
            try:
                self.inner.put(
                    key,
                    value,
                    node_id=node_id,
                    expected_version=None,
                    base_vc=base_vc,
                )
            except Exception:
                pass

        return ver

    def resolve(self, key: str, resolver: Callable, *, node_id: str):
        return self.inner.resolve(key, resolver, node_id=node_id)
