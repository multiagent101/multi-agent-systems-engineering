# Listing 5.7 — Delay injection: global and per-link delay profile updates.

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable


@dataclass(frozen=True)
class DelayEvent:
    at_s: float
    global_base_s: Optional[float] = None
    global_jitter_s: Optional[float] = None
    link: Optional[Tuple[str, str]] = None
    link_base_s: Optional[float] = None
    link_jitter_s: Optional[float] = None
    clear_link: bool = False


class InjectableNetwork:
    """
    Delay-injectable message transport with delivery hook and accounting callback.
    """
    def __init__(
        self,
        rng: random.Random,
        base_delay_s: float,
        jitter_s: float,
        on_send: Optional[Callable[[str, str, Any], None]] = None,
        on_deliver: Optional[Callable[[str, str, Any, float], None]] = None,
    ):
        self._rng = rng
        self._base = base_delay_s
        self._jit = jitter_s
        self._on_send = on_send
        self._on_deliver = on_deliver
        self._endpoints: Dict[str, asyncio.Queue[Any]] = {}
        self._overrides: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def register(self, endpoint_id: str, inbox: asyncio.Queue[Any]) -> None:
        self._endpoints[endpoint_id] = inbox

    def set_global_delay(self, base_s: float, jitter_s: float) -> None:
        self._base = max(0.0, base_s)
        self._jit = max(0.0, jitter_s)

    def set_link_delay(self, src: str, dst: str, base_s: float, jitter_s: float) -> None:
        self._overrides[(src, dst)] = (max(0.0, base_s), max(0.0, jitter_s))

    def clear_link_delay(self, src: str, dst: str) -> None:
        self._overrides.pop((src, dst), None)

    async def send(self, src: str, dst: str, msg: Any) -> None:
        if dst not in self._endpoints:
            return
        if self._on_send is not None:
            self._on_send(src, dst, msg)

        base, jit = self._overrides.get((src, dst), (self._base, self._jit))
        delay = max(0.0, base + self._rng.uniform(-jit, jit))
        if delay:
            await asyncio.sleep(delay)

        delivered_at = time.perf_counter()
        if self._on_deliver is not None:
            self._on_deliver(src, dst, msg, delivered_at)
        await self._endpoints[dst].put((src, msg))


class DelayInjector:
    def __init__(self, net: InjectableNetwork) -> None:
        self.net = net
        self.events: list[DelayEvent] = []
        self.log: list[tuple[float, str]] = []

    def add(self, ev: DelayEvent) -> None:
        self.events.append(ev)

    async def _execute(self, ev: DelayEvent, run_start: float) -> None:
        now = time.perf_counter()
        wait = (run_start + ev.at_s) - now
        if wait > 0:
            await asyncio.sleep(wait)

        if ev.global_base_s is not None or ev.global_jitter_s is not None:
            base = ev.global_base_s if ev.global_base_s is not None else self.net._base
            jit = ev.global_jitter_s if ev.global_jitter_s is not None else self.net._jit
            self.net.set_global_delay(base, jit)
            self.log.append((time.perf_counter(), f"global_delay base={base} jit={jit}"))

        if ev.link is not None:
            src, dst = ev.link
            if ev.clear_link:
                self.net.clear_link_delay(src, dst)
                self.log.append((time.perf_counter(), f"clear_link {src}->{dst}"))
            else:
                base = 0.0 if ev.link_base_s is None else ev.link_base_s
                jit = 0.0 if ev.link_jitter_s is None else ev.link_jitter_s
                self.net.set_link_delay(src, dst, base, jit)
                self.log.append((time.perf_counter(), f"link_delay {src}->{dst} base={base} jit={jit}"))

    async def run(self, run_start: float) -> None:
        tasks = []
        for ev in self.events:
            tasks.append(asyncio.create_task(self._execute(ev, run_start)))
        if tasks:
            await asyncio.gather(*tasks)
