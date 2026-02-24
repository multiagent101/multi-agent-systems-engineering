# Listing 10.1 — Minimal tracing substrate: explicit context propagation + JSONL exporter.

from __future__ import annotations

import atexit
import contextlib
import contextvars
import json
import os
import queue
import secrets
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterator, Optional


_trace_ctx: contextvars.ContextVar[Optional["TraceContext"]] = contextvars.ContextVar("trace_ctx", default=None)


def _hex(nbytes: int) -> str:
    return secrets.token_hex(nbytes)


def _sanitize_json(x: Any, *, max_str: int = 512, max_depth: int = 4) -> Any:
    if max_depth <= 0:
        return "<truncated>"
    if x is None or isinstance(x, (bool, int, float)):
        return x
    if isinstance(x, str):
        return x if len(x) <= max_str else (x[:max_str] + "…")
    if isinstance(x, dict):
        out: Dict[str, Any] = {}
        for k, v in x.items():
            ks = str(k)
            out[ks] = _sanitize_json(v, max_str=max_str, max_depth=max_depth - 1)
        return out
    if isinstance(x, (list, tuple)):
        return [_sanitize_json(v, max_str=max_str, max_depth=max_depth - 1) for v in x]
    return _sanitize_json(repr(x), max_str=max_str, max_depth=max_depth)


class AsyncJsonlWriter:
    """
    Bounded, best-effort JSONL writer. Records are buffered and written by a background thread.
    If the queue is full, records are dropped to avoid blocking coordination paths.
    """
    def __init__(self, path: str, *, max_queue: int = 10000, flush_every: int = 200, fsync: bool = False):
        self.path = path
        self.flush_every = max(1, flush_every)
        self.fsync = fsync
        self._q: "queue.Queue[str]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, name=f"jsonl-writer:{path}", daemon=True)
        self._thr.start()
        atexit.register(self.close)

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
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
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
                        if self.fsync:
                            try:
                                os.fsync(f.fileno())
                            except Exception:
                                pass
                f.flush()
        except Exception:
            return


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
    kind: str               # "client" | "coord" | "exec" | "internal"
    status: str             # "ok" | "error"
    attributes: Dict[str, Any]


class JsonlExporter:
    def __init__(self, path: str):
        self.path = path
        self._w = AsyncJsonlWriter(path)

    def emit(self, rec: SpanRecord) -> None:
        try:
            d = asdict(rec)
            d["attributes"] = _sanitize_json(d.get("attributes", {}))
            line = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
            self._w.write_line(line)
        except Exception:
            return


class Tracer:
    def __init__(self, exporter: JsonlExporter):
        self.exporter = exporter

    @contextlib.contextmanager
    def use(self, ctx: TraceContext) -> Iterator[None]:
        token = _trace_ctx.set(ctx)
        try:
            yield
        finally:
            _trace_ctx.reset(token)

    @contextlib.contextmanager
    def span(self, name: str, *, kind: str = "internal", attributes: Optional[Dict[str, Any]] = None) -> Iterator[TraceContext]:
        parent = _trace_ctx.get()
        ctx = parent.child() if parent is not None else TraceContext.new_root()
        token = _trace_ctx.set(ctx)

        t0 = time.perf_counter()
        status = "ok"
        attrs = attributes if attributes is not None else {}
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
                    t_start=t0,
                    t_end=t1,
                    trace_id=ctx.trace_id,
                    span_id=ctx.span_id,
                    parent_span_id=ctx.parent_span_id,
                    name=name,
                    kind=kind,
                    status=status,
                    attributes=attrs,
                )
            )

    def current(self) -> Optional[TraceContext]:
        return _trace_ctx.get()
