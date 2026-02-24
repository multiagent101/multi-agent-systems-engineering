# Listing 11.2 — Failure simulation experiment: tool degradation with and without resilience patterns.

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

# Reuse primitives from Listing 11.1: RetryPolicy, call_with_retry, CircuitBreaker, Bulkhead, IdempotencyStore, LeaseManager.
from fixed_code import (  # type: ignore
    RetryPolicy,
    call_with_retry,
    CircuitBreaker,
    Bulkhead,
    IdempotencyStore,
    LeaseManager,
)


class ToolError(RuntimeError):
    pass


class BreakerOpen(RuntimeError):
    pass


class BulkheadFull(RuntimeError):
    pass


class LeaseInvalid(RuntimeError):
    pass


class LeaseUnavailable(RuntimeError):
    pass


@dataclass
class ToolProfile:
    base_latency_s: float = 0.01
    failure_p: float = 0.01
    slow_factor: float = 1.0


class FlakyTool:
    def __init__(self, rng: random.Random, profile: ToolProfile):
        self.rng = rng
        self.profile = profile
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0

    def set_profile(self, profile: ToolProfile) -> None:
        with self._lock:
            self.profile = profile

    def call(self, payload: str) -> str:
        with self._lock:
            self.calls += 1
            p = self.profile
        # Simulated latency
        time.sleep(p.base_latency_s * p.slow_factor)
        # Simulated failure (guard RNG access under lock)
        with self._lock:
            r = self.rng.random()
        if r < p.failure_p:
            with self._lock:
                self.failures += 1
            raise ToolError("tool_error")
        return f"ok:{payload}"


@dataclass
class ExperimentResult:
    completed: int
    attempts: int
    tool_calls: int
    tool_failures: int
    duration_s: float


class WorkerWithResilience:
    def __init__(
        self,
        worker_id: str,
        tool: FlakyTool,
        rng: random.Random,
        *,
        breaker: Optional[CircuitBreaker] = None,
        bulkhead: Optional[Bulkhead] = None,
        retry: Optional[RetryPolicy] = None,
        idempotency: Optional[IdempotencyStore] = None,
        leases: Optional[LeaseManager] = None,
        lease_ttl_s: float = 0.2,
    ):
        self.worker_id = worker_id
        self.tool = tool
        self.rng = rng
        self.breaker = breaker
        self.bulkhead = bulkhead
        self.retry = retry
        self.idempotency = idempotency
        self.leases = leases
        self.lease_ttl_s = lease_ttl_s

        self.completed = 0
        self.attempts = 0  # dependency call attempts (including retries)

    def execute_task(self, task_id: int) -> None:
        lease_token = None
        lease_id = f"task:{task_id}"
        if self.leases is not None:
            try:
                lease = self.leases.acquire(lease_id, self.worker_id, self.lease_ttl_s)
                lease_token = lease.token
            except RuntimeError as e:
                if str(e) == "lease_unavailable":
                    raise LeaseUnavailable("lease_unavailable")
                raise

        def do_call() -> str:
            if self.leases is not None:
                if lease_token is None or not self.leases.validate(lease_id, lease_token):
                    raise LeaseInvalid("lease_invalid")

            if self.breaker is not None and not self.breaker.allow():
                raise BreakerOpen("breaker_open")

            acquired = False
            if self.bulkhead is not None:
                acquired = self.bulkhead.acquire(timeout_s=0.05)
                if not acquired:
                    raise BulkheadFull("bulkhead_full")

            try:
                self.attempts += 1
                res = self.tool.call(str(task_id))
                if self.leases is not None:
                    if not self.leases.validate(lease_id, lease_token):
                        raise LeaseInvalid("lease_invalid")
                if self.breaker is not None:
                    self.breaker.record_success()
                return res
            except ToolError:
                if self.breaker is not None:
                    self.breaker.record_failure()
                raise
            finally:
                if self.bulkhead is not None and acquired:
                    self.bulkhead.release()

        def compute() -> str:
            if self.retry is None:
                return do_call()
            return call_with_retry(
                do_call,
                policy=self.retry,
                rng=self.rng,
                retry_if=lambda e: isinstance(e, (ToolError, BulkheadFull)),
            )

        try:
            if self.idempotency is not None:
                _ = self.idempotency.get_or_put(f"task_result:{task_id}", compute)
            else:
                _ = compute()
            self.completed += 1
        finally:
            if self.leases is not None and lease_token is not None:
                self.leases.release(lease_id, lease_token)


def run_experiment(
    *,
    seed: int,
    tasks: int,
    concurrency: int,
    inject_at_s: float,
    degraded_profile: ToolProfile,
    resilient: bool,
) -> ExperimentResult:
    rng = random.Random(seed)
    tool = FlakyTool(random.Random(seed + 1), ToolProfile())

    if resilient:
        idemp = IdempotencyStore()
        leases = LeaseManager()
        breaker = CircuitBreaker(
            failure_threshold=4,
            open_interval_s=1.0,
            half_open_max_calls=2,
            half_open_success_threshold=2,
        )
        bulkhead = Bulkhead(capacity=max(1, concurrency // 2))
        retry = RetryPolicy(max_attempts=4, base_delay_s=0.02, max_delay_s=0.3, jitter=0.25)
    else:
        idemp = None
        leases = None
        breaker = None
        bulkhead = None
        # Poorly bounded retries: many attempts, no jitter, tight max delay.
        retry = RetryPolicy(max_attempts=50, base_delay_s=0.0, max_delay_s=0.01, jitter=0.0)

    workers = [
        WorkerWithResilience(
            worker_id=f"w{i}",
            tool=tool,
            rng=random.Random(seed + 100 + i),
            breaker=breaker,
            bulkhead=bulkhead,
            retry=retry,
            idempotency=idemp,
            leases=leases,
        )
        for i in range(concurrency)
    ]

    start = time.perf_counter()
    injected = False

    def worker_thread(w: WorkerWithResilience, ids: List[int]) -> None:
        for tid in ids:
            try:
                w.execute_task(tid)
            except LeaseUnavailable:
                pass
            except BreakerOpen:
                pass
            except LeaseInvalid:
                pass
            except Exception:
                pass

    # Submit duplicates to exercise at-least-once pressure.
    submitted: List[int] = []
    for tid in range(tasks):
        submitted.append(tid)
        submitted.append(tid)
    rng.shuffle(submitted)

    buckets: List[List[int]] = [[] for _ in range(concurrency)]
    for i, tid in enumerate(submitted):
        buckets[i % concurrency].append(tid)

    threads: List[threading.Thread] = []
    for i, w in enumerate(workers):
        th = threading.Thread(target=worker_thread, args=(w, buckets[i]), daemon=True)
        threads.append(th)

    for th in threads:
        th.start()

    while any(th.is_alive() for th in threads):
        now = time.perf_counter()
        if (not injected) and (now - start) >= inject_at_s:
            tool.set_profile(degraded_profile)
            injected = True
        time.sleep(0.01)

    for th in threads:
        th.join()

    end = time.perf_counter()
    completed = sum(w.completed for w in workers)
    attempts = sum(w.attempts for w in workers)
    return ExperimentResult(
        completed=completed,
        attempts=attempts,
        tool_calls=tool.calls,
        tool_failures=tool.failures,
        duration_s=end - start,
    )
