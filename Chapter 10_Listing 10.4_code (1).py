# Listing 10.4 — Instrumented transport wrapper: trace propagation + send/recv logging + handler spans.

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

# TraceContext, Tracer, MessageEnvelope, and MessageLogger are expected to be provided by
# earlier listings/modules in the same project.
from tracing import Tracer, TraceContext  # type: ignore
from msglog import MessageEnvelope, MessageLogger  # type: ignore

Handler = Callable[[str, MessageEnvelope], Awaitable[None]]  # handler(src, envelope)


class InstrumentedTransport:
    def __init__(self, tracer: Tracer, msglog: MessageLogger, *, inbox_max: int = 10000):
        self.tracer = tracer
        self.msglog = msglog
        self.inbox_max = max(1, inbox_max)
        self.inboxes: Dict[str, asyncio.Queue[Any]] = {}

    def register(self, endpoint: str) -> asyncio.Queue[Any]:
        q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self.inbox_max)
        self.inboxes[endpoint] = q
        return q

    async def send(self, src: str, dst: str, payload: Any, *, op_id: Optional[str] = None) -> None:
        ctx = self.tracer.current()

        if dst not in self.inboxes:
            # Best-effort: log and drop if destination is unavailable.
            if ctx is None:
                ctx = TraceContext.new_root()
            env = MessageEnvelope.wrap(payload, trace=ctx, op_id=op_id)
            self.msglog.log(
                "send",
                src,
                dst,
                env,
                emit_ctx=ctx,
                extra_fields={"dropped": True, "reason": "unknown_dst"},
            )
            return

        if ctx is None:
            # If send occurs outside an active span, anchor it with a local span.
            with self.tracer.span(
                f"send:{type(payload).__name__}",
                kind="internal",
                attributes={"src": src, "dst": dst},
            ):
                ctx = self.tracer.current() or TraceContext.new_root()
                env = MessageEnvelope.wrap(payload, trace=ctx, op_id=op_id)
                self.msglog.log("send", src, dst, env, emit_ctx=ctx)
                try:
                    self.inboxes[dst].put_nowait((src, env))
                except asyncio.QueueFull:
                    self.msglog.log(
                        "send",
                        src,
                        dst,
                        env,
                        emit_ctx=ctx,
                        extra_fields={"dropped": True, "reason": "inbox_full"},
                    )
            return

        env = MessageEnvelope.wrap(payload, trace=ctx, op_id=op_id)
        self.msglog.log("send", src, dst, env, emit_ctx=ctx)
        try:
            self.inboxes[dst].put_nowait((src, env))
        except asyncio.QueueFull:
            self.msglog.log(
                "send",
                src,
                dst,
                env,
                emit_ctx=ctx,
                extra_fields={"dropped": True, "reason": "inbox_full"},
            )

    async def recv_loop(self, endpoint: str, handler: Handler) -> None:
        inbox = self.inboxes[endpoint]
        while True:
            src, env = await inbox.get()
            # Install propagated context and create a handling span.
            with self.tracer.use(env.trace):
                with self.tracer.span(
                    f"handle:{env.msg_type}",
                    kind="internal",
                    attributes={"endpoint": endpoint, "src": src, "msg_id": env.msg_id, "op_id": env.op_id},
                ):
                    emit = self.tracer.current()
                    self.msglog.log("recv", src, endpoint, env, emit_ctx=emit)
                    await handler(src, env)
