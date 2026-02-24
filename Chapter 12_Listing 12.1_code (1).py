# Listing 12.1 — Benchmark harness runner producing normalized throughput/latency/overhead artifacts.

from __future__ import annotations

import asyncio
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

# Expected module from Chapter 5.2 and 5.3 in a repository layout.
# - latency_bench.py provides Task, SubmitTask, AssignTask, Award, ClientDone, Network, Worker, and engines.
# - probe.py provides TransportProbe and BandwidthLogger if per-link and overhead-only accounting is required.
from latency_bench import (
    Task,
    SubmitTask,
    AssignTask,
    Award,
    ClientDone,
    Network,
    Worker,
    CentralizedEngine,
    MarketEngine,
    PeerConsensusEngine,
    PeerFollower,
    Client,
    TransportCounters,
    LatSummary,
)


@dataclass(frozen=True)
class HarnessConfig:
    arch: str  # "centralized" | "peer" | "market"
    seed: int
    warmup_s: float
    measure_s: float
    drain_cap_s: float

    arrival_rate: float
    n_workers: int
    service_rate: float
    overhead_s: float

    base_delay_s: float
    jitter_s: float

    bid_deadline_s: float = 0.02
    n_followers: int = 4


@dataclass
class RunArtifact:
    config: Dict[str, Any]
    started_at: float
    ended_at: float
    submitted_measure: int
    completed_measure: int
    incomplete_measure: int
    goodput_tps: float
    coord: Dict[str, Any]
    e2e: Dict[str, Any]
    transport_measure: Dict[str, Any]
    overhead_per_task: Dict[str, Any]
    control_plane: Dict[str, Any]


def _percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = int(math.ceil(p * len(ys))) - 1
    i = max(0, min(i, len(ys) - 1))
    return ys[i]


def _lat_stats(samples_s: List[float]) -> Dict[str, Any]:
    if not samples_s:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    mean = statistics.fmean(samples_s)
    return {
        "count": len(samples_s),
        "mean_ms": mean * 1000.0,
        "p50_ms": _percentile(samples_s, 0.50) * 1000.0,
        "p95_ms": _percentile(samples_s, 0.95) * 1000.0,
        "p99_ms": _percentile(samples_s, 0.99) * 1000.0,
        "max_ms": max(samples_s) * 1000.0,
    }


async def run_harness(cfg: HarnessConfig) -> RunArtifact:
    rng = random.Random(cfg.seed)
    counters = TransportCounters()

    # Measurement gates
    t_start = time.perf_counter()
    t_warm_end = t_start + max(0.0, cfg.warmup_s)
    t_meas_end = t_warm_end + max(0.0, cfg.measure_s)
    t_drain_end = t_meas_end + max(0.0, cfg.drain_cap_s)

    # Snapshot transport counters for measurement-window attribution.
    warm_msgs = 0
    warm_bytes = 0
    meas_end_msgs = 0
    meas_end_bytes = 0

    # Latency samples restricted to measurement submissions.
    coord_lat_s: List[float] = []
    e2e_lat_s: List[float] = []

    # Track submissions during measurement only.
    submitted_measure = 0
    submitted_at: Dict[int, float] = {}
    done_at: Dict[int, float] = {}

    # Control-plane decision counts during measurement (binding decisions observed at delivery).
    binding_decisions = 0

    def on_deliver(src: str, dst: str, msg: Any, delivered_at: float) -> None:
        nonlocal binding_decisions
        # Binding assignment events.
        if isinstance(msg, AssignTask) and dst.startswith("w"):
            if msg.task.task_id in submitted_at:
                coord_lat_s.append(delivered_at - submitted_at[msg.task.task_id])
                binding_decisions += 1
        if isinstance(msg, Award) and dst.startswith("w") and msg.winner_id == dst:
            if msg.task.task_id in submitted_at:
                coord_lat_s.append(delivered_at - submitted_at[msg.task.task_id])
                binding_decisions += 1
        # Accepted completion event (observable proxy: client confirmation delivered).
        if isinstance(msg, ClientDone) and dst == "client":
            if msg.task_id in submitted_at:
                if msg.task_id not in done_at:
                    done_at[msg.task_id] = delivered_at
                    e2e_lat_s.append(delivered_at - submitted_at[msg.task_id])

    net = Network(
        rng=rng,
        base_delay_s=cfg.base_delay_s,
        jitter_s=cfg.jitter_s,
        counters=counters,
        on_deliver=on_deliver,
    )

    worker_ids = [f"w{i}" for i in range(cfg.n_workers)]
    workers = [
        Worker(wid, net, random.Random(cfg.seed + 1000 + i), cfg.service_rate, cfg.overhead_s)
        for i, wid in enumerate(worker_ids)
    ]
    client = Client("client", net)

    engine_tasks: List[asyncio.Task[Any]] = []
    followers: List[PeerFollower] = []

    if cfg.arch == "centralized":
        engine = CentralizedEngine(net, worker_ids)
        engine_id = "ctrl"
        engine_tasks.append(asyncio.create_task(engine.run()))
    elif cfg.arch == "market":
        engine = MarketEngine(net, worker_ids, cfg.bid_deadline_s)
        engine_id = "market"
        engine_tasks.append(asyncio.create_task(engine.run()))
    elif cfg.arch == "peer":
        follower_ids = [f"p{i}" for i in range(cfg.n_followers)]
        followers = [PeerFollower(fid, net, epoch=1) for fid in follower_ids]
        for f in followers:
            engine_tasks.append(asyncio.create_task(f.run()))
        engine = PeerConsensusEngine(net, worker_ids, follower_ids)
        engine_id = "leader"
        engine_tasks.append(asyncio.create_task(engine.run()))
    else:
        raise ValueError(f"unknown arch={cfg.arch}")

    tasks: List[asyncio.Task[Any]] = [asyncio.create_task(client.run())]
    tasks.extend(asyncio.create_task(w.run()) for w in workers)
    tasks.extend(engine_tasks)

    async def send_submit(task: Task) -> None:
        await net.send("client", engine_id, SubmitTask("client", task))

    # Warm-up: generate tasks but do not record them in measurement maps.
    task_id = 0
    next_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now >= t_warm_end:
            break
        if now < next_t:
            await asyncio.sleep(min(0.005, next_t - now))
            continue
        size = max(0.1, rng.lognormvariate(math.log(2.0), 0.6))  # median ~ 2.0
        task = Task(task_id=task_id, size=size, payload_bytes=256, created_at=now)
        asyncio.create_task(send_submit(task))
        task_id += 1
        next_t += rng.expovariate(cfg.arrival_rate)

    warm_msgs = counters.messages
    warm_bytes = counters.bytes

    # Measurement: record submissions for normalization and latency.
    next_t = time.perf_counter()
    while True:
        now = time.perf_counter()
        if now >= t_meas_end:
            break
        if now < next_t:
            await asyncio.sleep(min(0.005, next_t - now))
            continue
        size = max(0.1, rng.lognormvariate(math.log(2.0), 0.6))  # median ~ 2.0
        task = Task(task_id=task_id, size=size, payload_bytes=256, created_at=now)
        submitted_at[task.task_id] = now
        submitted_measure += 1
        asyncio.create_task(send_submit(task))
        task_id += 1
        next_t += rng.expovariate(cfg.arrival_rate)

    meas_end_msgs = counters.messages
    meas_end_bytes = counters.bytes

    # Drain until all measurement tasks complete or drain cap expires.
    while time.perf_counter() < t_drain_end and len(done_at) < len(submitted_at):
        await asyncio.sleep(0.01)

    t_end = time.perf_counter()

    # Stop components.
    client.alive = False
    for w in workers:
        w.alive = False
    engine.alive = False
    for f in followers:
        f.alive = False

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    completed_measure = len(done_at)
    incomplete_measure = max(0, submitted_measure - completed_measure)

    dur = max(1e-9, (t_meas_end - t_warm_end))
    goodput_tps = completed_measure / dur

    # Measurement-attributable transport deltas (warm-up excluded; drain included).
    delta_msgs = max(0, counters.messages - warm_msgs)
    delta_bytes = max(0, counters.bytes - warm_bytes)

    msgs_per_task = delta_msgs / max(1, completed_measure)
    bytes_per_task = delta_bytes / max(1, completed_measure)

    control_plane_tps = binding_decisions / dur

    art = RunArtifact(
        config=asdict(cfg),
        started_at=t_start,
        ended_at=t_end,
        submitted_measure=submitted_measure,
        completed_measure=completed_measure,
        incomplete_measure=incomplete_measure,
        goodput_tps=goodput_tps,
        coord=_lat_stats(coord_lat_s),
        e2e=_lat_stats(e2e_lat_s),
        transport_measure={"messages": delta_msgs, "bytes": delta_bytes},
        overhead_per_task={"messages_per_task": msgs_per_task, "bytes_per_task": bytes_per_task},
        control_plane={"binding_decisions": binding_decisions, "binding_tps": control_plane_tps},
    )
    return art


def write_artifact(artifact: RunArtifact, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(artifact), f, indent=2, ensure_ascii=False)
