# Listing 5.2 — Latency benchmark harness and scaling sweep.
# Standard library only. In-process simulation using asyncio and an instrumented transport.

from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import pickle
import random
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# -----------------------------
# Metrics utilities
# -----------------------------

def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    i = int(math.ceil(p * len(ys))) - 1
    i = max(0, min(i, len(ys) - 1))
    return ys[i]


@dataclass
class LatSummary:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    @staticmethod
    def from_samples(samples_s: List[float]) -> "LatSummary":
        if not samples_s:
            return LatSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        mean = statistics.fmean(samples_s)
        return LatSummary(
            count=len(samples_s),
            mean_ms=mean * 1000.0,
            p50_ms=percentile(samples_s, 0.50) * 1000.0,
            p95_ms=percentile(samples_s, 0.95) * 1000.0,
            p99_ms=percentile(samples_s, 0.99) * 1000.0,
            max_ms=max(samples_s) * 1000.0,
        )


@dataclass
class TransportCounters:
    messages: int = 0
    bytes: int = 0

    def add(self, msg: Any) -> None:
        self.messages += 1
        try:
            self.bytes += len(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            self.bytes += len(repr(msg).encode("utf-8"))


# -----------------------------
# Workload and message schema
# -----------------------------

@dataclass(frozen=True)
class Task:
    task_id: int
    size: float
    payload_bytes: int
    created_at: float  # monotonic submission timestamp


@dataclass(frozen=True)
class SubmitTask:
    client_id: str
    task: Task


@dataclass(frozen=True)
class AssignTask:
    task: Task
    client_id: str
    assigned_at: float


@dataclass(frozen=True)
class TaskDone:
    task_id: int
    worker_id: str
    done_at: float


@dataclass(frozen=True)
class ClientDone:
    task_id: int
    done_at: float


# Market-specific
@dataclass(frozen=True)
class BidRequest:
    task: Task
    request_id: int
    deadline_at: float


@dataclass(frozen=True)
class BidResponse:
    request_id: int
    worker_id: str
    bid_value: float


@dataclass(frozen=True)
class Award:
    task: Task
    client_id: str
    winner_id: str
    payment: float
    awarded_at: float


# Peer-consensus-specific (stable-leader, majority-ack replication)
@dataclass(frozen=True)
class LogAppend:
    leader_epoch: int
    index: int
    entry: Any  # ("assign", task_id, client_id, worker_id) or ("complete", task_id)


@dataclass(frozen=True)
class LogAck:
    leader_epoch: int
    index: int
    ok: bool
    follower_id: str


# -----------------------------
# Instrumented transport
# -----------------------------

class Network:
    """
    Delay-injecting message transport with a delivery hook used as an internal probe.
    The hook runs at delivery time, before the destination reads the message.
    """
    def __init__(
        self,
        rng: random.Random,
        base_delay_s: float,
        jitter_s: float,
        counters: TransportCounters,
        on_deliver: Optional[Callable[[str, str, Any, float], None]] = None,
    ):
        self._rng = rng
        self._base = base_delay_s
        self._jit = jitter_s
        self._counters = counters
        self._on_deliver = on_deliver
        self._endpoints: Dict[str, asyncio.Queue[Any]] = {}

    def register(self, endpoint: str, inbox: asyncio.Queue[Any]) -> None:
        self._endpoints[endpoint] = inbox

    async def send(self, src: str, dst: str, msg: Any) -> None:
        if dst not in self._endpoints:
            return
        self._counters.add(msg)
        delay = max(0.0, self._base + self._rng.uniform(-self._jit, self._jit))
        if delay:
            await asyncio.sleep(delay)
        delivered_at = time.perf_counter()
        if self._on_deliver is not None:
            self._on_deliver(src, dst, msg, delivered_at)
        await self._endpoints[dst].put((src, msg))


# -----------------------------
# Worker agents
# -----------------------------

class Worker:
    """
    Executor agent with local backlog state and a minimal interface: bid, execute.
    """
    def __init__(self, worker_id: str, net: Network, rng: random.Random, service_rate: float, overhead_s: float):
        self.worker_id = worker_id
        self.net = net
        self.rng = rng
        self.service_rate = max(1e-9, service_rate)
        self.overhead_s = overhead_s
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(worker_id, self.inbox)
        self.alive = True
        self.backlog_units: float = 0.0

    def estimate_cost(self, task: Task) -> float:
        load_factor = 1.0 + 0.10 * math.log1p(self.backlog_units)
        noise = self.rng.uniform(0.98, 1.02)
        return task.size * load_factor * noise

    async def run(self) -> None:
        while self.alive:
            src, msg = await self.inbox.get()
            if isinstance(msg, BidRequest):
                resp = BidResponse(msg.request_id, self.worker_id, self.estimate_cost(msg.task))
                await self.net.send(self.worker_id, "market", resp)
            elif isinstance(msg, AssignTask):
                await self._execute(msg.task, controller_id=src)
            elif isinstance(msg, Award):
                if msg.winner_id == self.worker_id:
                    await self._execute(msg.task, controller_id=src)

    async def _execute(self, task: Task, controller_id: str) -> None:
        self.backlog_units += task.size
        exec_s = self.overhead_s + (task.size / self.service_rate)
        exec_s = max(0.0, exec_s * (1.0 + self.rng.uniform(-0.01, 0.01)))
        await asyncio.sleep(exec_s)
        self.backlog_units = max(0.0, self.backlog_units - task.size)
        done = TaskDone(task.task_id, self.worker_id, time.perf_counter())
        await self.net.send(self.worker_id, controller_id, done)


# -----------------------------
# Control-plane engines
# -----------------------------

class CentralizedEngine:
    """
    Coordinator–worker control plane. Binding assignment is AssignTask delivery.
    Completion confirmation is emitted immediately upon TaskDone receipt.
    """
    def __init__(self, net: Network, worker_ids: List[str]):
        self.net = net
        self.engine_id = "ctrl"
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(self.engine_id, self.inbox)
        self.worker_ids = worker_ids
        self._rr = 0
        self._task_to_client: Dict[int, str] = {}
        self.alive = True

    def _pick_worker(self) -> str:
        wid = self.worker_ids[self._rr % len(self.worker_ids)]
        self._rr += 1
        return wid

    async def run(self) -> None:
        while self.alive:
            _, msg = await self.inbox.get()
            if isinstance(msg, SubmitTask):
                wid = self._pick_worker()
                self._task_to_client[msg.task.task_id] = msg.client_id
                assign = AssignTask(msg.task, msg.client_id, time.perf_counter())
                await self.net.send(self.engine_id, wid, assign)
            elif isinstance(msg, TaskDone):
                cid = self._task_to_client.get(msg.task_id)
                if cid is not None:
                    await self.net.send(self.engine_id, cid, ClientDone(msg.task_id, time.perf_counter()))
                    self._task_to_client.pop(msg.task_id, None)


class MarketEngine:
    """
    Sealed-bid reverse auction. Binding assignment is Award delivery to winner.
    Completion confirmation is emitted upon TaskDone receipt from winner.
    """
    def __init__(self, net: Network, worker_ids: List[str], bid_deadline_s: float):
        self.net = net
        self.engine_id = "market"
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(self.engine_id, self.inbox)
        self.worker_ids = worker_ids
        self.bid_deadline_s = bid_deadline_s
        self._req_id = 0
        self._pending: Dict[int, Dict[str, float]] = {}
        self._task_to_client: Dict[int, str] = {}
        self.alive = True

    async def run(self) -> None:
        while self.alive:
            _, msg = await self.inbox.get()
            if isinstance(msg, SubmitTask):
                asyncio.create_task(self._auction(msg.task, msg.client_id))
            elif isinstance(msg, BidResponse):
                bids = self._pending.get(msg.request_id)
                if bids is not None:
                    bids[msg.worker_id] = msg.bid_value
            elif isinstance(msg, TaskDone):
                cid = self._task_to_client.get(msg.task_id)
                if cid is not None:
                    await self.net.send(self.engine_id, cid, ClientDone(msg.task_id, time.perf_counter()))
                    self._task_to_client.pop(msg.task_id, None)

    async def _auction(self, task: Task, client_id: str) -> None:
        self._req_id += 1
        req_id = self._req_id
        self._pending[req_id] = {}
        deadline_at = time.perf_counter() + self.bid_deadline_s
        req = BidRequest(task, req_id, deadline_at)

        send_tasks = [
            asyncio.create_task(self.net.send(self.engine_id, wid, req))
            for wid in self.worker_ids
        ]
        await asyncio.gather(*send_tasks)

        while time.perf_counter() < deadline_at:
            await asyncio.sleep(0.001)

        bids = self._pending.pop(req_id, {})
        if not bids:
            return

        winner_id, _ = min(bids.items(), key=lambda kv: kv[1])
        sorted_vals = sorted(bids.values())
        payment = sorted_vals[1] if len(sorted_vals) >= 2 else sorted_vals[0]

        self._task_to_client[task.task_id] = client_id
        award = Award(task, client_id, winner_id, payment, time.perf_counter())
        await self.net.send(self.engine_id, winner_id, award)


class PeerFollower:
    """
    Follower replica for stable-leader replication. Entry acceptance enforces log contiguity.
    """
    def __init__(self, node_id: str, net: Network, epoch: int):
        self.node_id = node_id
        self.net = net
        self.epoch = epoch
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(node_id, self.inbox)
        self.log: List[Any] = []
        self.alive = True

    async def run(self) -> None:
        while self.alive:
            src, msg = await self.inbox.get()
            if isinstance(msg, LogAppend):
                ok = False
                if msg.leader_epoch == self.epoch and msg.index == len(self.log):
                    self.log.append(msg.entry)
                    ok = True
                await self.net.send(self.node_id, src, LogAck(msg.leader_epoch, msg.index, ok, self.node_id))


class PeerConsensusEngine:
    """
    Leader-driven majority-ack replication. Binding assignment is dispatch following quorum acknowledgment.
    Completion confirmation is emitted only after majority acknowledgment of a completion entry.
    """
    def __init__(self, net: Network, worker_ids: List[str], follower_ids: List[str]):
        self.net = net
        self.engine_id = "leader"
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(self.engine_id, self.inbox)

        self.worker_ids = worker_ids
        self.follower_ids = follower_ids
        self.epoch = 1

        self.log: List[Any] = []
        self._acks: Dict[int, set[str]] = {}
        self._wait: Dict[int, asyncio.Event] = {}

        self._rr = 0
        self._task_to_client: Dict[int, str] = {}
        self.alive = True

    def _majority(self) -> int:
        total = 1 + len(self.follower_ids)
        return total // 2 + 1

    def _pick_worker(self) -> str:
        wid = self.worker_ids[self._rr % len(self.worker_ids)]
        self._rr += 1
        return wid

    async def run(self) -> None:
        while self.alive:
            _, msg = await self.inbox.get()
            if isinstance(msg, SubmitTask):
                await self._commit_assign(msg.task, msg.client_id)
            elif isinstance(msg, LogAck):
                if msg.leader_epoch != self.epoch or not msg.ok:
                    continue
                s = self._acks.get(msg.index)
                if s is None:
                    continue
                s.add(msg.follower_id)
                if len(s) + 1 >= self._majority():
                    ev = self._wait.get(msg.index)
                    if ev is not None and not ev.is_set():
                        ev.set()
            elif isinstance(msg, TaskDone):
                cid = self._task_to_client.get(msg.task_id)
                if cid is not None:
                    await self._commit_complete(msg.task_id, cid)

    async def _replicate_and_commit(self, index: int, entry: Any) -> None:
        self._acks[index] = set()
        self._wait[index] = asyncio.Event()
        la = LogAppend(self.epoch, index, entry)
        send_tasks = [
            asyncio.create_task(self.net.send(self.engine_id, fid, la))
            for fid in self.follower_ids
        ]
        await asyncio.gather(*send_tasks)
        await self._wait[index].wait()
        self._acks.pop(index, None)
        self._wait.pop(index, None)

    async def _commit_assign(self, task: Task, client_id: str) -> None:
        wid = self._pick_worker()
        entry = ("assign", task.task_id, client_id, wid)
        idx = len(self.log)
        self.log.append(entry)
        await self._replicate_and_commit(idx, entry)
        self._task_to_client[task.task_id] = client_id
        await self.net.send(self.engine_id, wid, AssignTask(task, client_id, time.perf_counter()))

    async def _commit_complete(self, task_id: int, client_id: str) -> None:
        entry = ("complete", task_id)
        idx = len(self.log)
        self.log.append(entry)
        await self._replicate_and_commit(idx, entry)
        await self.net.send(self.engine_id, client_id, ClientDone(task_id, time.perf_counter()))
        self._task_to_client.pop(task_id, None)


# -----------------------------
# Client
# -----------------------------

class Client:
    def __init__(self, client_id: str, net: Network):
        self.client_id = client_id
        self.net = net
        self.inbox: asyncio.Queue[Any] = asyncio.Queue()
        self.net.register(client_id, self.inbox)
        self.submitted_at: Dict[int, float] = {}
        self.done_at: Dict[int, float] = {}
        self.alive = True

    async def run(self) -> None:
        while self.alive:
            _, msg = await self.inbox.get()
            if isinstance(msg, ClientDone):
                self.done_at[msg.task_id] = time.perf_counter()


# -----------------------------
# Benchmark runner and scaling sweep
# -----------------------------

@dataclass
class BenchConfig:
    arch: str  # "centralized" | "peer" | "market"
    seed: int
    n_workers: int
    arrival_rate: float
    duration_s: float

    mean_size: float = 2.0
    size_sigma: float = 0.6
    payload_bytes: int = 256

    service_rate: float = 5.0
    overhead_s: float = 0.002

    base_net_delay_s: float = 0.002
    net_jitter_s: float = 0.003

    # market
    bid_deadline_s: float = 0.02

    # peer
    n_followers: int = 4


@dataclass
class BenchResult:
    submitted: int
    completed: int
    throughput_tps: float
    coordination: LatSummary
    end_to_end: LatSummary
    transport: TransportCounters


async def run_once(cfg: BenchConfig) -> BenchResult:
    rng = random.Random(cfg.seed)
    counters = TransportCounters()

    coordination_lat_s: List[float] = []

    def on_deliver(src: str, dst: str, msg: Any, delivered_at: float) -> None:
        if dst.startswith("w") and isinstance(msg, AssignTask):
            coordination_lat_s.append(delivered_at - msg.task.created_at)
        if dst.startswith("w") and isinstance(msg, Award) and msg.winner_id == dst:
            coordination_lat_s.append(delivered_at - msg.task.created_at)

    net = Network(
        rng=rng,
        base_delay_s=cfg.base_net_delay_s,
        jitter_s=cfg.net_jitter_s,
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
        engine_tasks.append(asyncio.create_task(engine.run()))
        engine_id = "ctrl"
    elif cfg.arch == "market":
        engine = MarketEngine(net, worker_ids, cfg.bid_deadline_s)
        engine_tasks.append(asyncio.create_task(engine.run()))
        engine_id = "market"
    elif cfg.arch == "peer":
        follower_ids = [f"p{i}" for i in range(cfg.n_followers)]
        followers = [PeerFollower(fid, net, epoch=1) for fid in follower_ids]
        for f in followers:
            engine_tasks.append(asyncio.create_task(f.run()))
        engine = PeerConsensusEngine(net, worker_ids, follower_ids)
        engine_tasks.append(asyncio.create_task(engine.run()))
        engine_id = "leader"
    else:
        raise ValueError(f"unknown arch={cfg.arch}")

    tasks: List[asyncio.Task[Any]] = [asyncio.create_task(client.run())]
    tasks.extend(asyncio.create_task(w.run()) for w in workers)
    tasks.extend(engine_tasks)

    start = time.perf_counter()
    submitted = 0

    while time.perf_counter() - start < cfg.duration_s:
        now = time.perf_counter()
        size = max(0.1, rng.lognormvariate(math.log(cfg.mean_size), cfg.size_sigma))
        task = Task(task_id=submitted, size=size, payload_bytes=cfg.payload_bytes, created_at=now)
        client.submitted_at[task.task_id] = now
        await net.send("client", engine_id, SubmitTask(client.client_id, task))
        submitted += 1
        await asyncio.sleep(rng.expovariate(cfg.arrival_rate))

    drain_deadline = time.perf_counter() + 10.0
    while len(client.done_at) < len(client.submitted_at) and time.perf_counter() < drain_deadline:
        await asyncio.sleep(0.01)

    end = time.perf_counter()

    e2e_lat_s = [
        client.done_at[tid] - t0
        for tid, t0 in client.submitted_at.items()
        if tid in client.done_at
    ]

    client.alive = False
    for w in workers:
        w.alive = False
    engine.alive = False
    for f in followers:
        f.alive = False

    for t in tasks:
        t.cancel()
    with contextlib.suppress(Exception):
        await asyncio.gather(*tasks, return_exceptions=True)

    completed = len(client.done_at)
    duration = max(1e-9, end - start)

    return BenchResult(
        submitted=submitted,
        completed=completed,
        throughput_tps=completed / duration,
        coordination=LatSummary.from_samples(coordination_lat_s),
        end_to_end=LatSummary.from_samples(e2e_lat_s),
        transport=counters,
    )


async def scaling_sweep(cfg: BenchConfig, worker_counts: List[int]) -> None:
    print(
        "arch,n_workers,arrival_rate,submitted,completed,throughput_tps,"
        "coord_p50_ms,coord_p95_ms,coord_p99_ms,e2e_p50_ms,e2e_p95_ms,e2e_p99_ms"
    )
    for n in worker_counts:
        run_cfg = BenchConfig(**{**cfg.__dict__, "n_workers": n})
        r = await run_once(run_cfg)
        print(
            f"{run_cfg.arch},{n},{run_cfg.arrival_rate:.6f},{r.submitted},{r.completed},{r.throughput_tps:.6f},"
            f"{r.coordination.p50_ms:.3f},{r.coordination.p95_ms:.3f},{r.coordination.p99_ms:.3f},"
            f"{r.end_to_end.p50_ms:.3f},{r.end_to_end.p95_ms:.3f},{r.end_to_end.p99_ms:.3f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["centralized", "peer", "market"], required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--arrival-rate", type=float, default=50.0)
    ap.add_argument("--duration-s", type=float, default=5.0)
    ap.add_argument("--workers", type=str, default="2,4,8,16")
    ap.add_argument("--base-delay-s", type=float, default=0.002)
    ap.add_argument("--jitter-s", type=float, default=0.003)
    ap.add_argument("--service-rate", type=float, default=5.0)
    ap.add_argument("--overhead-s", type=float, default=0.002)
    ap.add_argument("--bid-deadline-s", type=float, default=0.02)
    ap.add_argument("--followers", type=int, default=4)
    args = ap.parse_args()

    cfg = BenchConfig(
        arch=args.arch,
        seed=args.seed,
        n_workers=2,
        arrival_rate=args.arrival_rate,
        duration_s=args.duration_s,
        base_net_delay_s=args.base_delay_s,
        net_jitter_s=args.jitter_s,
        service_rate=args.service_rate,
        overhead_s=args.overhead_s,
        bid_deadline_s=args.bid_deadline_s,
        n_followers=args.followers,
    )
    worker_counts = [int(x.strip()) for x in args.workers.split(",") if x.strip()]
    asyncio.run(scaling_sweep(cfg, worker_counts))


if __name__ == "__main__":
    main()
