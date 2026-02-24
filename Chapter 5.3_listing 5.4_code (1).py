# Listing 5.4 — Periodic bandwidth logger producing per-link rates.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, TextIO

# Reuse TransportProbe and TransportSnapshot from Listing 5.3.
# from probe import TransportProbe, TransportSnapshot, LinkCounters


@dataclass
class LinkRate:
    msgs_per_s: float
    bytes_per_s: float
    overhead_bytes_per_s: float


def _delta_rate(prev: int, curr: int, dt: float) -> float:
    if dt <= 0:
        return 0.0
    return (curr - prev) / dt


class BandwidthLogger:
    def __init__(self, probe: "TransportProbe", interval_s: float, out: TextIO):
        self.probe = probe
        self.interval_s = max(0.01, interval_s)
        self.out = out
        self._running = False
        self._last: Optional["TransportSnapshot"] = None

    def _rates(
        self, last: "TransportSnapshot", now: "TransportSnapshot"
    ) -> Dict[Tuple[str, str], LinkRate]:
        dt = max(1e-9, now.captured_at - last.captured_at)
        rates: Dict[Tuple[str, str], LinkRate] = {}

        keys = set(last.per_link.keys()) | set(now.per_link.keys())
        for k in keys:
            a = last.per_link.get(k)
            b = now.per_link.get(k)
            am, ab, ao = (a.messages, a.bytes_total, a.bytes_overhead) if a else (0, 0, 0)
            bm, bb, bo = (b.messages, b.bytes_total, b.bytes_overhead) if b else (0, 0, 0)
            rates[k] = LinkRate(
                msgs_per_s=_delta_rate(am, bm, dt),
                bytes_per_s=_delta_rate(ab, bb, dt),
                overhead_bytes_per_s=_delta_rate(ao, bo, dt),
            )
        return rates

    async def run(self) -> None:
        self._running = True
        self._last = self.probe.snapshot()
        self.out.write("t_s,src,dst,msgs_per_s,bytes_per_s,overhead_bytes_per_s\n")
        self.out.flush()

        t0 = self._last.captured_at
        while self._running:
            await asyncio.sleep(self.interval_s)
            snap = self.probe.snapshot()
            last = self._last
            self._last = snap

            rates = self._rates(last, snap)
            for (src, dst), r in rates.items():
                self.out.write(
                    f"{snap.captured_at - t0:.6f},{src},{dst},"
                    f"{r.msgs_per_s:.6f},{r.bytes_per_s:.6f},{r.overhead_bytes_per_s:.6f}\n"
                )
            self.out.flush()

    def stop(self) -> None:
        self._running = False
