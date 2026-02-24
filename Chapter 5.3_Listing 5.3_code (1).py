# Listing 5.3 — TransportProbe middleware: global + per-link message/byte accounting.

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# Message types referenced by schema-aware overhead estimation.
# These names match the benchmark messages defined in Section 5.2.
class Task: ...
class SubmitTask: ...
class AssignTask: ...
class Award: ...
class BidRequest: ...


@dataclass
class LinkCounters:
    messages: int = 0
    bytes_total: int = 0
    bytes_overhead: int = 0


@dataclass
class TransportSnapshot:
    captured_at: float
    total_messages: int
    total_bytes: int
    total_overhead_bytes: int
    per_link: Dict[Tuple[str, str], LinkCounters] = field(default_factory=dict)


class TransportProbe:
    """
    Transport-level accounting for message counts and byte volume.

    bytes_total estimates size using pickle as a consistent proxy within a fixed runtime.
    bytes_overhead is a payload-excluded estimate under the benchmark's payload model,
    where Task.payload_bytes is a workload parameter and not an embedded byte array.
    """

    def __init__(self) -> None:
        self.total_messages = 0
        self.total_bytes = 0
        self.total_overhead_bytes = 0
        self.per_link: Dict[Tuple[str, str], LinkCounters] = {}

    def _estimate_bytes(self, msg: Any) -> int:
        try:
            return len(pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return len(repr(msg).encode("utf-8"))

    def _payload_bytes(self, msg: Any) -> int:
        """
        Schema-aware payload extraction. If the workload models payload size
        as Task.payload_bytes, subtract it from overhead accounting for
        messages that embed Task objects, isolating coordination overhead.
        This keeps overhead comparable when payload sizes change.
        """
        # Defensive access; message schemas are dataclasses in the benchmark.
        task = getattr(msg, "task", None)
        if task is not None:
            return int(getattr(task, "payload_bytes", 0) or 0)
        return 0

    def on_send(self, src: str, dst: str, msg: Any) -> None:
        sz = self._estimate_bytes(msg)
        payload = 0
        overhead = sz

        self.total_messages += 1
        self.total_bytes += sz
        self.total_overhead_bytes += overhead

        key = (src, dst)
        lc = self.per_link.get(key)
        if lc is None:
            lc = LinkCounters()
            self.per_link[key] = lc
        lc.messages += 1
        lc.bytes_total += sz
        lc.bytes_overhead += overhead

    def snapshot(self) -> TransportSnapshot:
        # Copy per-link counters to freeze a point-in-time view.
        frozen = {
            k: LinkCounters(v.messages, v.bytes_total, v.bytes_overhead)
            for k, v in self.per_link.items()
        }
        return TransportSnapshot(
            captured_at=time.perf_counter(),
            total_messages=self.total_messages,
            total_bytes=self.total_bytes,
            total_overhead_bytes=self.total_overhead_bytes,
            per_link=frozen,
        )
