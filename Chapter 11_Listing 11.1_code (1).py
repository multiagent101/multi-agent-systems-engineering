# Listing 11.1 — Resilience primitives: retry/backoff, circuit breaker, bulkhead, idempotency store, lease manager, outbox.

from __future__ import annotations

import random
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.02
    max_delay_s: float = 1.0
    jitter: float = 0.2


def call_with_retry(
    fn: Callable[[], Any],
    *,
    policy: RetryPolicy,
    rng: random.Random,
    retry_if: Callable[[Exception], bool],
    on_attempt: Optional[Callable[[], None]] = None,
) -> Any:
    attempt = 0
    while True:
        try:
            if on_attempt is not None:
                on_attempt()
            return fn()
        except Exception as e:
            attempt += 1
            if attempt >= policy.max_attempts or not retry_if(e):
                raise
            delay = min(policy.max_delay_s, policy.base_delay_s * (2 ** (attempt - 1)))
            if policy.jitter > 0:
                delay *= (1.0 + rng.uniform(-policy.jitter, policy.jitter))
            time.sleep(max(0.0, delay))


class CircuitBreaker:
    """
    Threshold-based circuit breaker with time-based open interval and half-open probes.
    Trips after a bounded number of recorded failures and closes after a bounded number
    of successful probes in half-open.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        open_interval_s: float = 2.0,
        half_open_max_calls: int = 2,
        half_open_success_threshold: int = 2,
    ):
        self.failure_threshold = max(1, failure_threshold)
        self.open_interval_s = max(0.1, open_interval_s)
        self.half_open_max_calls = max(1, half_open_max_calls)
        self.half_open_success_threshold = max(1, half_open_success_threshold)

        self._lock = threading.Lock()
        self._state = "closed"  # closed | open | half_open
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._half_open_successes = 0

    def allow(self) -> bool:
        with self._lock:
            now = time.perf_counter()
            if self._state == "closed":
                return True
            if self._state == "open":
                if (now - self._opened_at) >= self.open_interval_s:
                    self._state = "half_open"
                    self._half_open_calls = 0
                    self._half_open_successes = 0
                    return True
                return False
            if self._state == "half_open":
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
        return False

    def record_success(self) -> None:
        with self._lock:
            if self._state == "half_open":
                self._half_open_successes += 1
                if self._half_open_successes >= self.half_open_success_threshold:
                    self._state = "closed"
                    self._failures = 0
                return
            self._failures = 0
            if self._state == "open":
                self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state == "half_open":
                self._state = "open"
                self._opened_at = time.perf_counter()
                return
            if self._state == "closed" and self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = time.perf_counter()

    def state(self) -> str:
        with self._lock:
            return self._state


class Bulkhead:
    """
    Concurrency limiter for a dependency.
    """

    def __init__(self, capacity: int):
        self.sem = threading.Semaphore(max(1, capacity))

    def acquire(self, timeout_s: Optional[float] = None) -> bool:
        if timeout_s is None:
            return self.sem.acquire()
        return self.sem.acquire(timeout=timeout_s)

    def release(self) -> None:
        self.sem.release()


class IdempotencyStore:
    """
    Thread-safe idempotency store. Ensures only one compute executes per key and
    that duplicates observe the stored result (or stored exception) without re-executing.
    In production this is backed by durable state.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._seen: Dict[str, Any] = {}
        self._inflight: Dict[str, threading.Event] = {}

    def get_or_put(self, key: str, compute: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._seen:
                v = self._seen[key]
                if isinstance(v, Exception):
                    raise v
                return v
            ev = self._inflight.get(key)
            if ev is None:
                ev = threading.Event()
                self._inflight[key] = ev
                leader = True
            else:
                leader = False

        if not leader:
            ev.wait()
            with self._lock:
                v = self._seen.get(key)
                if isinstance(v, Exception):
                    raise v
                return v

        try:
            result = compute()
        except Exception as e:
            with self._lock:
                self._seen[key] = e
                self._inflight.pop(key, None)
                ev.set()
            raise
        else:
            with self._lock:
                self._seen.setdefault(key, result)
                self._inflight.pop(key, None)
                ev.set()
                return self._seen[key]


@dataclass(frozen=True)
class Lease:
    holder_id: str
    token: str  # versioned token
    expires_at: float


class LeaseManager:
    """
    Lease manager using CAS-like semantics on an underlying key-value store.
    The store is modeled as a dict with a lock for this reference implementation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._leases: Dict[str, Lease] = {}

    def acquire(self, lease_id: str, holder_id: str, ttl_s: float) -> Lease:
        now = time.perf_counter()
        token = f"{holder_id}:{secrets.token_hex(8)}"
        new_lease = Lease(holder_id=holder_id, token=token, expires_at=now + max(0.01, ttl_s))
        with self._lock:
            cur = self._leases.get(lease_id)
            if cur is None or cur.expires_at <= now:
                self._leases[lease_id] = new_lease
                return new_lease
            raise RuntimeError("lease_unavailable")

    def renew(self, lease_id: str, token: str, ttl_s: float) -> Lease:
        now = time.perf_counter()
        with self._lock:
            cur = self._leases.get(lease_id)
            if cur is None:
                raise RuntimeError("lease_missing")
            if cur.token != token:
                raise RuntimeError("lease_token_mismatch")
            if cur.expires_at <= now:
                raise RuntimeError("lease_expired")
            new_lease = Lease(holder_id=cur.holder_id, token=cur.token, expires_at=now + max(0.01, ttl_s))
            self._leases[lease_id] = new_lease
            return new_lease

    def validate(self, lease_id: str, token: str) -> bool:
        now = time.perf_counter()
        with self._lock:
            cur = self._leases.get(lease_id)
            return cur is not None and cur.token == token and cur.expires_at > now

    def release(self, lease_id: str, token: str) -> None:
        with self._lock:
            cur = self._leases.get(lease_id)
            if cur is not None and cur.token == token:
                self._leases.pop(lease_id, None)


class Outbox:
    """
    Minimal outbox for durable-intent message emission.
    In production, outbox records live in the same durable transaction as state updates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending: Dict[str, Any] = {}  # msg_id -> payload
        self._sent: Dict[str, float] = {}  # msg_id -> sent_at

    def put(self, msg_id: str, payload: Any) -> None:
        with self._lock:
            if msg_id in self._pending or msg_id in self._sent:
                return
            self._pending[msg_id] = payload

    def get_batch(self, max_items: int = 100) -> Dict[str, Any]:
        with self._lock:
            out: Dict[str, Any] = {}
            for i, (mid, p) in enumerate(self._pending.items()):
                if i >= max_items:
                    break
                out[mid] = p
            return out

    def mark_sent(self, msg_id: str) -> None:
        with self._lock:
            if msg_id in self._pending:
                self._pending.pop(msg_id, None)
                self._sent[msg_id] = time.perf_counter()
