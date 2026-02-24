# Listing 13.1 — Core reference system (coordinator + workers + shared state + instrumented transport).

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import secrets
import time
import threading
import queue
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional, Tuple, List


# -------------------------
# Tracing (Chapter 10 model)
# -------------------------

def _hex(nbytes: int) -> str:
    return secrets.token_hex(nbytes)

_trace_ctx: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar("trace_ctx", default=None)

@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None

    @staticmethod
    def new_root() -> "TraceContext":
        return TraceContext(trace_id=_hex(16), span_id=_hex(8), parent_span_id=None)

    def child(self) -> "TraceContext":
        return TraceContext(trace_id=self.trace_id, span_id=_hex(8), parent_span_id=self.span_id)

@dataclass
class SpanRecord:
    t_start: float
    t_end: float
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    status: str
    attributes: Dict[str, Any]

class AsyncJsonlWriter:
    """
    Best-effort JSONL writer. Records are buffered and written by a background thread
    to avoid blocking the event loop on per-event file I/O.
    """
    def __init__(self, path: str, *, max_queue: int = 10000, flush_every: int = 200):
        self.path = path
        self.flush_every = max(1, flush_every)
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def write_line(self, line: str) -> None:
        try:
            self._q.put_nowait(line)
        except queue.Full:
            pass

    def close(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._thr.join(timeout=1.0)
        except Exception:
            pass

    def _run(self) -> None:
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                n = 0
                while not self._stop.is_set() or not self._q.empty():
                    try:
                        line = self._q.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    f.write(line + "\n")
                    n += 1
                    if n % self.flush_every == 0:
                        f.flush()
                f.flush()
        except Exception:
            return

class JsonlSpanExporter:
    def __init__(self, path: str):
        self._w = AsyncJsonlWriter(path)

    def emit(self, rec: SpanRecord) -> None:
        try:
            self._w.write_line(json.dumps(asdict(rec), separators=(",", ":"), ensure_ascii=False))
        except Exception:
            return

class Tracer:
    def __init__(self, exporter: JsonlSpanExporter):
        self.exporter = exporter

    @contextlib.contextmanager
    def use(self, ctx: TraceContext):
        token = _trace_ctx.set(ctx)
        try:
            yield
        finally:
            _trace_ctx.reset(token)

    @contextlib.contextmanager
    def span(self, name: str, *, kind: str = "internal", attributes: Optional[Dict[str, Any]] = None):
        parent = _trace_ctx.get()
        ctx = parent.child() if parent else TraceContext.new_root()
        token = _trace_ctx.set(ctx)

        t0 = time.perf_counter()
        status = "ok"
        attrs = dict(attributes or {})
        try:
            yield ctx
        except Exception as e:
            status = "error"
            attrs["exception.type"] = type(e).__name__
            attrs["exception.message"] = str(e)
            raise
        finally:
            t1 = time.perf_counter()
            _trace_ctx.reset(token)
            self.exporter.emit(
                SpanRecord(
                    t_start=t0, t_end=t1,
                    trace_id=ctx.trace_id, span_id=ctx.span_id, parent_span_id=ctx.parent_span_id,
                    name=name, kind=kind, status=status, attributes=attrs
                )
            )

    def current(self) -> Optional[TraceContext]:
        return _trace_ctx.get()


# -------------------------
# Message schemas and logging
# -------------------------

def _msg_id() -> str:
    return secrets.token_hex(12)

@dataclass(frozen=True)
class MessageEnvelope:
    msg_id: str
    trace: TraceContext
    msg_type: str
    payload: Any

    @staticmethod
    def wrap(payload: Any, trace: TraceContext) -> "MessageEnvelope":
        return MessageEnvelope(msg_id=_msg_id(), trace=trace, msg_type=type(payload).__name__, payload=payload)

@dataclass
class MessageLogRecord:
    t: float
    direction: str
    src: str
    dst: str
    msg_id: str
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    msg_type: str
    fields: Dict[str, Any]

class MessageLogger:
    def __init__(self, path: str, extractor: Callable[[Any], Dict[str, Any]]):
        self.extractor = extractor
        self._w = AsyncJsonlWriter(path)

    def log(self, direction: str, src: str, dst: str, env: MessageEnvelope) -> None:
        r = MessageLogRecord(
            t=time.perf_counter(),
            direction=direction,
            src=src,
            dst=dst,
            msg_id=env.msg_id,
            trace_id=env.trace.trace_id,
            span_id=env.trace.span_id,
            parent_span_id=env.trace.parent_span_id,
            msg_type=env.msg_type,
            fields=self.extractor(env.payload),
        )
        try:
            self._w.write_line(json.dumps(asdict(r), separators=(",", ":"), ensure_ascii=False))
        except Exception:
            return


# -------------------------
# Shared state: leases + completion idempotency (Chapter 6/11)
# -------------------------

@dataclass(frozen=True)
class LeaseRecord:
    task_id: int
    holder_id: str
    token: str
    expires_at: float
    version: int

@dataclass(frozen=True)
class CompletionRecord:
    task_id: int
    completed_by: str
    completed_at: float
    version: int

class InMemoryState:
    """
    Narrow authoritative store for coordination-critical records.

    This in-process implementation enforces exclusivity and idempotent completion under a single lock.
    A production system backs the same interface with a durable, strongly consistent substrate.
    """
    def __init__(self):
        self._lock = asyncio.Lock()
        self._v = 0
        self._leases: Dict[int, LeaseRecord] = {}
        self._completions: Dict[int, CompletionRecord] = {}

    def _next_version_locked(self) -> int:
        self._v += 1
        return self._v

    async def get_lease(self, task_id: int) -> Optional[LeaseRecord]:
        async with self._lock:
            return self._leases.get(task_id)

    async def acquire_lease(self, task_id: int, holder_id: str, ttl_s: float) -> LeaseRecord:
        now = time.perf_counter()
        async with self._lock:
            cur = self._leases.get(task_id)
            if cur is not None and cur.expires_at > now:
                raise RuntimeError("lease_unavailable")
            ver = self._next_version_locked()
            token = secrets.token_hex(16)
            rec = LeaseRecord(task_id=task_id, holder_id=holder_id, token=token, expires_at=now + max(0.01, ttl_s), version=ver)
            self._leases[task_id] = rec
            return rec

    async def renew_lease(self, task_id: int, token: str, ttl_s: float) -> LeaseRecord:
        now = time.perf_counter()
        async with self._lock:
            cur = self._leases.get(task_id)
            if cur is None:
                raise RuntimeError("lease_missing")
            if cur.token != token:
                raise RuntimeError("lease_token_mismatch")
            if cur.expires_at <= now:
                raise RuntimeError("lease_expired")
            ver = self._next_version_locked()
            rec = LeaseRecord(task_id=task_id, holder_id=cur.holder_id, token=cur.token, expires_at=now + max(0.01, ttl_s), version=ver)
            self._leases[task_id] = rec
            return rec

    async def validate_lease(self, task_id: int, token: str) -> bool:
        now = time.perf_counter()
        async with self._lock:
            cur = self._leases.get(task_id)
            return cur is not None and cur.token == token and cur.expires_at > now

    async def mark_complete(self, task_id: int, completed_by: str) -> CompletionRecord:
        now = time.perf_counter()
        async with self._lock:
            if task_id in self._completions:
                return self._completions[task_id]
            ver = self._next_version_locked()
            rec = CompletionRecord(task_id=task_id, completed_by=completed_by, completed_at=now, version=ver)
            self._completions[task_id] = rec
            return rec

    async def is_complete(self, task_id: int) -> bool:
        async with self._lock:
            return task_id in self._completions


# -------------------------
# Transport with trace propagation (Chapter 10)
# -------------------------

class Transport:
    def __init__(self, tracer: Tracer, msglog: MessageLogger):
        self.tracer = tracer
        self.msglog = msglog
        self.inboxes: Dict[str, asyncio.Queue[Tuple[str, MessageEnvelope]]] = {}

    def register(self, endpoint: str) -> asyncio.Queue[Tuple[str, MessageEnvelope]]:
        q: asyncio.Queue[Tuple[str, MessageEnvelope]] = asyncio.Queue()
        self.inboxes[endpoint] = q
        return q

    async def send(self, src: str, dst: str, payload: Any) -> None:
        if dst not in self.inboxes:
            return
        ctx = self.tracer.current() or TraceContext.new_root()
        env = MessageEnvelope.wrap(payload, trace=ctx)
        self.msglog.log("send", src, dst, env)
        await self.inboxes[dst].put((src, env))


# -------------------------
# Messages (control-plane semantics)
# -------------------------

@dataclass(frozen=True)
class Task:
    task_id: int
    size: float

@dataclass(frozen=True)
class SubmitTask:
    task: Task

@dataclass(frozen=True)
class AssignTask:
    task: Task
    lease_token: str

@dataclass(frozen=True)
class TaskDone:
    task_id: int
    lease_token: str

@dataclass(frozen=True)
class ClientDone:
    task_id: int


# -------------------------
# Agents
# -------------------------

class Agent:
    def __init__(self, agent_id: str, transport: Transport, tracer: Tracer):
        self.agent_id = agent_id
        self.transport = transport
        self.tracer = tracer
        self.inbox = transport.register(agent_id)
        self.alive = True

    async def run(self) -> None:
        while self.alive:
            src, env = await self.inbox.get()
            with self.tracer.use(env.trace):
                with self.tracer.span(
                    f"handle:{env.msg_type}",
                    kind="internal",
                    attributes={"agent": self.agent_id, "src": src, "msg_id": env.msg_id},
                ):
                    await self.handle(src, env.payload)

    async def handle(self, src: str, msg: Any) -> None:
        raise NotImplementedError


class WorkerAgent(Agent):
    def __init__(self, agent_id: str, transport: Transport, tracer: Tracer, service_rate: float):
        super().__init__(agent_id, transport, tracer)
        self.service_rate = max(1e-9, service_rate)
        self.backlog_units: float = 0.0

    async def handle(self, src: str, msg: Any) -> None:
        if isinstance(msg, AssignTask):
            await self._execute_and_report(src, msg)

    async def _execute_and_report(self, coordinator_id: str, assign: AssignTask) -> None:
        with self.tracer.span("exec.task", kind="exec", attributes={"task_id": assign.task.task_id, "worker": self.agent_id}):
            self.backlog_units += assign.task.size
            await asyncio.sleep(assign.task.size / self.service_rate)
            self.backlog_units = max(0.0, self.backlog_units - assign.task.size)

        await self.transport.send(self.agent_id, coordinator_id, TaskDone(task_id=assign.task.task_id, lease_token=assign.lease_token))


class CoordinatorAgent(Agent):
    def __init__(
        self,
        agent_id: str,
        transport: Transport,
        tracer: Tracer,
        state: InMemoryState,
        worker_ids: List[str],
        lease_ttl_s: float = 0.5,
    ):
        super().__init__(agent_id, transport, tracer)
        self.state = state
        self.worker_ids = worker_ids
        self.lease_ttl_s = lease_ttl_s
        self._rr = 0
        self._client_for_task: Dict[int, str] = {}

    def _pick_worker(self) -> str:
        wid = self.worker_ids[self._rr % len(self.worker_ids)]
        self._rr += 1
        return wid

    async def handle(self, src: str, msg: Any) -> None:
        if isinstance(msg, SubmitTask):
            await self._on_submit(src, msg.task)
        elif isinstance(msg, TaskDone):
            await self._on_done(src, msg)

    async def _on_submit(self, client_id: str, task: Task) -> None:
        with self.tracer.span("coord.assign", kind="coord", attributes={"task_id": task.task_id}):
            if await self.state.is_complete(task.task_id):
                await self.transport.send(self.agent_id, client_id, ClientDone(task_id=task.task_id))
                return

            wid = self._pick_worker()
            lease = await self.state.acquire_lease(task.task_id, holder_id=wid, ttl_s=self.lease_ttl_s)
            self._client_for_task[task.task_id] = client_id
            await self.transport.send(self.agent_id, wid, AssignTask(task=task, lease_token=lease.token))

    async def _on_done(self, worker_id: str, done: TaskDone) -> None:
        with self.tracer.span("coord.complete", kind="coord", attributes={"task_id": done.task_id, "worker": worker_id}):
            if not await self.state.validate_lease(done.task_id, done.lease_token):
                return

            _ = await self.state.mark_complete(done.task_id, completed_by=worker_id)
            client_id = self._client_for_task.get(done.task_id)
            if client_id is not None:
                await self.transport.send(self.agent_id, client_id, ClientDone(task_id=done.task_id))
                self._client_for_task.pop(done.task_id, None)


class ClientAgent(Agent):
    def __init__(self, agent_id: str, transport: Transport, tracer: Tracer):
        super().__init__(agent_id, transport, tracer)
        self.done: Dict[int, float] = {}

    async def handle(self, src: str, msg: Any) -> None:
        if isinstance(msg, ClientDone):
            self.done[msg.task_id] = time.perf_counter()


# -------------------------
# End-to-end runner
# -------------------------

def default_extractor(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, SubmitTask):
        return {"task_id": payload.task.task_id}
    if isinstance(payload, AssignTask):
        return {"task_id": payload.task.task_id}
    if isinstance(payload, TaskDone):
        return {"task_id": payload.task_id}
    if isinstance(payload, ClientDone):
        return {"task_id": payload.task_id}
    return {}


async def run_demo(n_workers: int = 4, n_tasks: int = 20) -> Dict[str, Any]:
    span_exporter = JsonlSpanExporter("spans.jsonl")
    tracer = Tracer(span_exporter)
    msglog = MessageLogger("messages.jsonl", extractor=default_extractor)

    transport = Transport(tracer, msglog)
    state = InMemoryState()

    client = ClientAgent("client", transport, tracer)
    workers = [WorkerAgent(f"w{i}", transport, tracer, service_rate=5.0) for i in range(n_workers)]
    coord = CoordinatorAgent("coord", transport, tracer, state, worker_ids=[w.agent_id for w in workers], lease_ttl_s=10.0)

    tasks = [asyncio.create_task(a.run()) for a in ([client, coord] + workers)]

    t0 = time.perf_counter()
    for tid in range(n_tasks):
        with tracer.span("client.submit", kind="client", attributes={"task_id": tid}):
            await transport.send("client", "coord", SubmitTask(task=Task(task_id=tid, size=1.0 + (tid % 3))))
    deadline = time.perf_counter() + 10.0
    while len(client.done) < n_tasks and time.perf_counter() < deadline:
        await asyncio.sleep(0.01)
    t1 = time.perf_counter()

    client.alive = False
    coord.alive = False
    for w in workers:
        w.alive = False
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    completed = len(client.done)
    return {
        "completed": completed,
        "duration_s": t1 - t0,
        "throughput_tps": completed / max(1e-9, (t1 - t0)),
    }


if __name__ == "__main__":
    print(asyncio.run(run_demo()))
