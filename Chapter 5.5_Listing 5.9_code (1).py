# Listing 5.9 — Load testing runner: saturation sweep, contention sampling, and sharded centralized experiment.

from __future__ import annotations

import asyncio
import contextlib
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Callable

# Reuse types from Section 5.2 module (Task, SubmitTask, ClientDone, etc.).
# from latency_bench import Task, SubmitTask, ClientDone, Network, Worker, CentralizedEngine, MarketEngine, PeerConsensusEngine, PeerFollower, Client, BenchConfig, run_once


@dataclass
class ContentionSample:
    t_s: float
    queues: Dict[str, int]
    state: Dict[str, float]


class ContentionSampler:
    """
    Periodically samples queue depths and optional scalar state metrics.
    """
    def __init__(self, interval_s: float = 0.05):
        self.interval_s = max(0.01, interval_s)
        self.samples: List[ContentionSample] = []
        self._running = False

    async def run(
        self,
        start_t: float,
        queues: Dict[str, "asyncio.Queue[Any]"],
        state_fns: Dict[str, Callable[[], float]],
    ) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_s)
            now = time.perf_counter()
            q = {name: qobj.qsize() for name, qobj in queues.items()}
            s = {name: float(fn()) for name, fn in state_fns.items()}
            self.samples.append(ContentionSample(t_s=now - start_t, queues=q, state=s))

    def stop(self) -> None:
        self._running = False


class ShardedCentralRouter:
    """
    Routes SubmitTask messages to one of K independent coordinators by task_id modulo.
    This models coordination-scope scaling for centralized orchestration.
    """
    def __init__(self, net: Any, coordinator_ids: List[str], k: int):
        self.net = net
        self.engine_id = "router"
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(self.engine_id, self.inbox)
        self.coordinator_ids = coordinator_ids
        self.k = max(1, k)
        self.alive = True

    def _route(self, task_id: int) -> str:
        return self.coordinator_ids[task_id % self.k]

    async def run(self) -> None:
        from latency_bench import SubmitTask
        while self.alive:
            _, msg = await self.inbox.get()
            if isinstance(msg, SubmitTask):
                dst = self._route(msg.task.task_id)
                await self.net.send(self.engine_id, dst, msg)


class ShardNetAdapter:
    """
    Presents a per-shard view of a shared transport by aliasing the controller id "ctrl"
    to a shard-specific endpoint id (e.g., "ctrl0"), allowing reuse of components that
    hardcode "ctrl" for coordinator communication.
    """
    def __init__(self, net: Any, ctrl_actual: str):
        self._net = net
        self._ctrl_actual = ctrl_actual

    def register(self, endpoint_id: str, inbox: "asyncio.Queue[Any]") -> None:
        if endpoint_id == "ctrl":
            self._net.register(self._ctrl_actual, inbox)
        else:
            self._net.register(endpoint_id, inbox)

    async def send(self, src: str, dst: str, msg: Any) -> None:
        if src == "ctrl":
            src = self._ctrl_actual
        if dst == "ctrl":
            dst = self._ctrl_actual
        await self._net.send(src, dst, msg)


@dataclass
class LoadPoint:
    arrival_rate: float
    submitted: int
    completed: int
    incomplete: int
    throughput_tps: float
    coord_p95_ms: float
    e2e_p95_ms: float


async def saturation_sweep(
    run_fn,
    arrival_rates: List[float],
    *,
    max_points: Optional[int] = None,
) -> List[LoadPoint]:
    points: List[LoadPoint] = []
    for i, lam in enumerate(arrival_rates):
        if max_points is not None and i >= max_points:
            break
        res = await run_fn(lam)
        point = res[0] if isinstance(res, tuple) else res
        points.append(point)
    return points


async def run_profiled_sharded_centralized(
    *,
    seed: int,
    duration_s: float,
    arrival_rate: float,
    n_workers: int,
    n_shards: int,
    base_delay_s: float,
    jitter_s: float,
    service_rate: float,
    overhead_s: float,
    sample_interval_s: float = 0.05,
) -> Tuple[LoadPoint, List[ContentionSample]]:
    # The implementation mirrors the Section 5.2 runner but adds routing + sampling.
    from latency_bench import (
        Task, SubmitTask,
        Network, Worker, CentralizedEngine, Client,
        TransportCounters, LatSummary,
    )

    rng = random.Random(seed)
    counters = TransportCounters()

    coord_lat_s: List[float] = []

    def on_deliver(src: str, dst: str, msg: Any, delivered_at: float) -> None:
        if dst.startswith("w") and msg.__class__.__name__ == "AssignTask":
            coord_lat_s.append(delivered_at - msg.task.created_at)

    net = Network(rng=rng, base_delay_s=base_delay_s, jitter_s=jitter_s, counters=counters, on_deliver=on_deliver)

    worker_ids = [f"w{i}" for i in range(n_workers)]

    k = max(1, min(n_shards, n_workers))
    coord_ids = [f"ctrl{i}" for i in range(k)]

    # Partition workers into disjoint pools; router partitions task submissions consistently.
    shard_workers: List[List[str]] = []
    for shard in range(k):
        shard_workers.append([wid for idx, wid in enumerate(worker_ids) if idx % k == shard])

    # Create per-shard engines and per-shard worker instances, each bound to a shard adapter.
    coordinators: List[CentralizedEngine] = []
    workers: List[Worker] = []
    for shard in range(k):
        shard_net = ShardNetAdapter(net, ctrl_actual=coord_ids[shard])
        c = CentralizedEngine(shard_net, shard_workers[shard])
        coordinators.append(c)
        for i, wid in enumerate(shard_workers[shard]):
            w = Worker(wid, shard_net, random.Random(seed + 1000 + i), service_rate, overhead_s)
            workers.append(w)

    client = Client("client", net)
    router = ShardedCentralRouter(net, coord_ids, k=k)

    # Sampling: coordinator/worker/client/router inbox depth.
    queues: Dict[str, asyncio.Queue[Any]] = {"client": client.inbox, "router": router.inbox}
    for shard, c in enumerate(coordinators):
        queues[coord_ids[shard]] = c.inbox
    for w in workers:
        queues[w.worker_id] = w.inbox

    # State sampling: worker backlog_units as a coarse execution-pressure proxy.
    state_fns = {f"backlog_{w.worker_id}": (lambda ww=w: ww.backlog_units) for w in workers}

    sampler = ContentionSampler(sample_interval_s)

    tasks: List[asyncio.Task[Any]] = []
    start = time.perf_counter()

    tasks.append(asyncio.create_task(client.run()))
    tasks.append(asyncio.create_task(router.run()))
    tasks.extend(asyncio.create_task(c.run()) for c in coordinators)
    tasks.extend(asyncio.create_task(w.run()) for w in workers)
    tasks.append(asyncio.create_task(sampler.run(start, queues, state_fns)))

    submitted = 0
    next_t = start

    # Exponential inter-arrival scheduling using absolute deadlines to avoid delay-coupling.
    while True:
        now = time.perf_counter()
        if now - start >= duration_s:
            break
        if now < next_t:
            await asyncio.sleep(next_t - now)
            continue

        now = time.perf_counter()
        size = max(0.1, rng.lognormvariate(math.log(2.0), 0.6))  # median size is 2.0 under this parameterization
        task = Task(task_id=submitted, size=size, payload_bytes=256, created_at=now)
        client.submitted_at[task.task_id] = now
        asyncio.create_task(net.send("client", "router", SubmitTask(client.client_id, task)))
        submitted += 1
        next_t = now + rng.expovariate(arrival_rate)

    drain_deadline = time.perf_counter() + 10.0
    while len(client.done_at) < len(client.submitted_at) and time.perf_counter() < drain_deadline:
        await asyncio.sleep(0.01)

    end = time.perf_counter()

    # Compute e2e latencies for completed tasks only (incomplete tasks are right-censored).
    e2e_lat_s = [
        client.done_at[tid] - t0
        for tid, t0 in client.submitted_at.items()
        if tid in client.done_at
    ]

    sampler.stop()
    client.alive = False
    router.alive = False
    for c in coordinators:
        c.alive = False
    for w in workers:
        w.alive = False

    for t in tasks:
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*tasks, return_exceptions=True)

    completed = len(client.done_at)
    incomplete = max(0, submitted - completed)
    duration = max(1e-9, end - start)

    coord = LatSummary.from_samples(coord_lat_s)
    e2e = LatSummary.from_samples(e2e_lat_s)

    point = LoadPoint(
        arrival_rate=arrival_rate,
        submitted=submitted,
        completed=completed,
        incomplete=incomplete,
        throughput_tps=completed / duration,
        coord_p95_ms=coord.p95_ms,
        e2e_p95_ms=e2e.p95_ms,
    )
    return point, sampler.samples
