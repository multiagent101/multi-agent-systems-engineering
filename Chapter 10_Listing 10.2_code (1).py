# Listing 10.2 — Message envelope + structured message logger with semantic field extraction.

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Optional

# TraceContext and AsyncJsonlWriter from Listing 10.1 are reused here.
# from tracing import TraceContext, AsyncJsonlWriter, _sanitize_json


def _msg_id() -> str:
    return secrets.token_hex(12)


def _op_id() -> str:
    return secrets.token_hex(12)


@dataclass(frozen=True)
class MessageEnvelope:
    msg_id: str
    op_id: Optional[str]      # stable across retries when idempotency is required
    trace: TraceContext       # propagated trace context
    msg_type: str
    payload: Any

    @staticmethod
    def wrap(payload: Any, trace: TraceContext, op_id: Optional[str] = None) -> "MessageEnvelope":
        return MessageEnvelope(
            msg_id=_msg_id(),
            op_id=op_id,
            trace=trace,
            msg_type=type(payload).__name__,
            payload=payload,
        )


@dataclass
class MessageLogRecord:
    t: float
    direction: str            # "send" | "recv"
    src: str
    dst: str
    msg_id: str
    op_id: Optional[str]
    trace_id: str
    trace_span_id: str
    trace_parent_span_id: Optional[str]
    emit_span_id: Optional[str]
    emit_parent_span_id: Optional[str]
    msg_type: str
    fields: Dict[str, Any]


class MessageLogger:
    """
    Logs send/recv events as JSONL with stable schema.
    The extractor selects a bounded set of semantic fields from payloads.
    """

    def __init__(self, path: str, extractor: Optional[Callable[[Any], Dict[str, Any]]] = None):
        self.path = path
        self.extractor = extractor or (lambda _: {})
        self._w = AsyncJsonlWriter(path)

    def log(
        self,
        direction: str,
        src: str,
        dst: str,
        env: MessageEnvelope,
        *,
        emit_ctx: Optional[TraceContext] = None,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        t = time.perf_counter()
        fields = dict(self.extractor(env.payload) or {})
        if extra_fields:
            fields.update(extra_fields)
        rec = MessageLogRecord(
            t=t,
            direction=direction,
            src=src,
            dst=dst,
            msg_id=env.msg_id,
            op_id=env.op_id,
            trace_id=env.trace.trace_id,
            trace_span_id=env.trace.span_id,
            trace_parent_span_id=env.trace.parent_span_id,
            emit_span_id=(emit_ctx.span_id if emit_ctx else None),
            emit_parent_span_id=(emit_ctx.parent_span_id if emit_ctx else None),
            msg_type=env.msg_type,
            fields=fields,
        )
        try:
            d = asdict(rec)
            d["fields"] = _sanitize_json(d.get("fields", {}))
            line = json.dumps(d, separators=(",", ":"), ensure_ascii=False)
            self._w.write_line(line)
        except Exception:
            return
