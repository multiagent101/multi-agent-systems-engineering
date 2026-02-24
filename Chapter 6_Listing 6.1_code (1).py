# Listing 6.1 — Versioned shared-state layer with version chains + multi-value register semantics.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar
import threading

T = TypeVar("T")


@dataclass(frozen=True)
class VectorClock:
    """
    Minimal vector clock for concurrency detection.
    """
    clock: Tuple[Tuple[str, int], ...]  # sorted items for stable hashing/ordering

    @staticmethod
    def empty() -> "VectorClock":
        return VectorClock(tuple())

    def to_dict(self) -> Dict[str, int]:
        return dict(self.clock)

    @staticmethod
    def from_dict(d: Dict[str, int]) -> "VectorClock":
        return VectorClock(tuple(sorted(d.items())))

    def tick(self, node_id: str) -> "VectorClock":
        d = self.to_dict()
        d[node_id] = d.get(node_id, 0) + 1
        return VectorClock.from_dict(d)

    def merge(self, other: "VectorClock") -> "VectorClock":
        a = self.to_dict()
        b = other.to_dict()
        out: Dict[str, int] = {}
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            out[k] = max(a.get(k, 0), b.get(k, 0))
        return VectorClock.from_dict(out)

    def compare(self, other: "VectorClock") -> str:
        """
        Returns one of: "equal", "before", "after", "concurrent".
        """
        a = self.to_dict()
        b = other.to_dict()
        keys = set(a.keys()) | set(b.keys())
        le = True
        ge = True
        eq = True
        for k in keys:
            av = a.get(k, 0)
            bv = b.get(k, 0)
            if av != bv:
                eq = False
            if av > bv:
                le = False
            if av < bv:
                ge = False
        if eq:
            return "equal"
        if le:
            return "before"
        if ge:
            return "after"
        return "concurrent"


@dataclass(frozen=True)
class Version:
    """
    Storage-visible version.
    seq provides a total order when assigned by an authority or committed log.
    vc provides concurrency metadata for eventual-concurrency modes.
    """
    seq: int
    vc: VectorClock


@dataclass(frozen=True)
class ValueVersion(Generic[T]):
    version: Version
    value: T


@dataclass(frozen=True)
class ReadResult(Generic[T]):
    """
    versions contains all versions visible at the chosen snapshot boundary.
    Concurrent versions (under vector-clock comparison) represent siblings.
    """
    versions: Tuple[ValueVersion[T], ...]


class SharedState(Generic[T]):
    def read(self, key: str, *, at_seq: Optional[int] = None) -> ReadResult[T]:
        raise NotImplementedError

    def put(
        self,
        key: str,
        value: T,
        *,
        node_id: str,
        expected_version: Optional[Version] = None,
        base_vc: Optional[VectorClock] = None,
    ) -> Version:
        raise NotImplementedError

    def resolve(
        self,
        key: str,
        resolver: Callable[[Tuple[ValueVersion[T], ...]], ValueVersion[T]],
        *,
        node_id: str,
    ) -> Version:
        raise NotImplementedError


class InMemoryMVRegister(SharedState[T]):
    """
    In-memory store with per-key version chains and multi-value register semantics.

    - seq is assigned monotonically by this store in single-authority mode.
    - vc tracks causality when concurrent writes are permitted.
    - Historical versions are retained to support snapshot reads; explicit
      garbage collection would be required in long-running systems.

    In log-ordered mode, apply_committed() must be used exclusively for writes.
    """
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq: int = 0
        self._store: Dict[str, List[ValueVersion[T]]] = {}

    def read(self, key: str, *, at_seq: Optional[int] = None) -> ReadResult[T]:
        with self._lock:
            chain = self._store.get(key, [])
            if at_seq is None:
                return ReadResult(tuple(chain))
            filt = [vv for vv in chain if vv.version.seq <= at_seq]
            return ReadResult(tuple(filt))

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def put(
        self,
        key: str,
        value: T,
        *,
        node_id: str,
        expected_version: Optional[Version] = None,
        base_vc: Optional[VectorClock] = None,
    ) -> Version:
        """
        If expected_version is provided, enforces CAS against that exact version.
        base_vc must reflect the causal context observed by the caller when
        concurrent-write semantics are required.
        """
        with self._lock:
            chain = self._store.get(key, [])

            if expected_version is not None:
                if not any(vv.version == expected_version for vv in chain):
                    raise ValueError(f"CAS failed for key={key}: expected_version not found")

            vc = (base_vc or VectorClock.empty()).tick(node_id)
            ver = Version(seq=self._next_seq(), vc=vc)
            new_vv = ValueVersion(version=ver, value=value)

            chain = list(chain)
            chain.append(new_vv)
            self._store[key] = chain
            return ver

    def resolve(
        self,
        key: str,
        resolver: Callable[[Tuple[ValueVersion[T], ...]], ValueVersion[T]],
        *,
        node_id: str,
    ) -> Version:
        """
        Resolves visible concurrent versions by appending a new version that
        is causally after the versions supplied to the resolver.

        Convergence requires that resolution observe all relevant concurrent
        versions (e.g., after anti-entropy) and that resolver be deterministic.
        """
        with self._lock:
            chain = self._store.get(key, [])
            if not chain:
                raise KeyError(f"resolve on missing key={key}")

            siblings = tuple(chain)
            chosen = resolver(siblings)

            merged = VectorClock.empty()
            for vv in siblings:
                merged = merged.merge(vv.version.vc)
            vc = merged.tick(node_id)
            ver = Version(seq=self._next_seq(), vc=vc)

            chain = list(chain)
            chain.append(ValueVersion(version=ver, value=chosen.value))
            self._store[key] = chain
            return ver

    def apply_committed(self, seq: int, key: str, value: T, *, vc: VectorClock) -> None:
        """
        Applies a committed write with externally assigned seq.

        seq must be strictly increasing relative to previously applied values.
        In this mode, no local writes outside the committed log are permitted.
        """
        with self._lock:
            if seq <= self._seq:
                raise ValueError("apply_committed requires strictly increasing seq")
            self._seq = seq
            ver = Version(seq=seq, vc=vc)
            chain = self._store.get(key, [])
            chain = list(chain)
            chain.append(ValueVersion(version=ver, value=value))
            self._store[key] = chain
