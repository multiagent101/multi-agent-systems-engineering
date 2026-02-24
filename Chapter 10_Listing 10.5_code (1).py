# Listing 10.5 — Instrumented shared-state wrapper: spans + semantic fields for versioning outcomes.

from __future__ import annotations

from typing import Any, Optional, Callable

# SharedState interface from Chapter 6 is assumed.
# from shared_state import SharedState, ReadResult, Version, ValueVersion

# Tracer is expected to be provided by earlier listings/modules in the same project.
from tracing import Tracer  # type: ignore


class InstrumentedState:
    def __init__(self, inner: Any, tracer: Tracer):
        self.inner = inner
        self.tracer = tracer

    def read(self, key: str, *, at_seq: Optional[int] = None):
        attrs = {"key": key, "at_seq": at_seq}
        with self.tracer.span("state.read", kind="internal", attributes=attrs):
            rr = self.inner.read(key, at_seq=at_seq)
            siblings = getattr(rr, "siblings", ())
            attrs["siblings"] = len(siblings)
            return rr

    def put(self, key: str, value: Any, *, node_id: str, expected_seq: Optional[int] = None, base_vc=None):
        attrs = {
            "key": key,
            "node_id": node_id,
            "expected_seq": expected_seq,
            "has_base_vc": base_vc is not None,
        }
        with self.tracer.span("state.put", kind="internal", attributes=attrs):
            try:
                ver = self.inner.put(key, value, node_id=node_id, expected_seq=expected_seq, base_vc=base_vc)
                attrs["seq"] = getattr(ver, "seq", None)
                return ver
            except Exception as e:
                attrs["error"] = "put_failed"
                attrs["error.type"] = type(e).__name__
                raise

    def resolve(self, key: str, resolver: Callable, *, node_id: str):
        attrs = {"key": key, "node_id": node_id}
        with self.tracer.span("state.resolve", kind="internal", attributes=attrs):
            ver = self.inner.resolve(key, resolver, node_id=node_id)
            attrs["seq"] = getattr(ver, "seq", None)
            return ver
