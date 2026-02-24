# Listing 7.2 — Instrumentation pipeline: event bus + sampling loop + detector integration.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Dict, Callable

# Reuse detectors and event types from Listing 7.1.
# from stability_detectors import OscillationDetector, DeadlockDetector, DriftDetector, QueueSample, ResourceEvent, MetricSample


@dataclass(frozen=True)
class Alert:
    t: float
    kind: str
    message: str


class EventBus:
    def __init__(self, maxsize: int = 10000) -> None:
        self.queue_samples: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self.resource_events: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self.metric_samples: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self.alerts: asyncio.Queue[Alert] = asyncio.Queue(maxsize=maxsize)


class StabilityMonitor:
    """
    Consumes samples/events, runs detectors, and publishes alerts.
    """
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.osc = OscillationDetector()
        self.dead = DeadlockDetector()
        self.drift = DriftDetector()
        self._running = False

    async def run(self) -> None:
        self._running = True
        while self._running:
            # multiplex without leaking pending tasks
            t_q = asyncio.create_task(self.bus.queue_samples.get())
            t_r = asyncio.create_task(self.bus.resource_events.get())
            t_m = asyncio.create_task(self.bus.metric_samples.get())

            done, pending = await asyncio.wait(
                {t_q, t_r, t_m},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            for task in done:
                item = task.result()
                now = time.perf_counter()

                if isinstance(item, QueueSample):
                    msg = self.osc.add(item)
                    if msg:
                        await self.bus.alerts.put(Alert(now, "oscillation", msg))

                elif isinstance(item, ResourceEvent):
                    msg = self.dead.add(item)
                    if msg:
                        await self.bus.alerts.put(Alert(now, "deadlock", msg))

                elif isinstance(item, MetricSample):
                    msg = self.drift.add(item)
                    if msg:
                        await self.bus.alerts.put(Alert(now, "drift", msg))

    def stop(self) -> None:
        self._running = False


class Sampler:
    """
    Periodically polls registered signal functions and emits QueueSample/MetricSample.
    """
    def __init__(self, bus: EventBus, interval_s: float = 0.05):
        self.bus = bus
        self.interval_s = max(0.01, interval_s)
        self.queue_sources: Dict[str, Callable[[], int]] = {}
        self.metric_sources: Dict[str, Callable[[], float]] = {}
        self._running = False

    def register_queue(self, name: str, fn: Callable[[], int]) -> None:
        self.queue_sources[name] = fn

    def register_metric(self, name: str, fn: Callable[[], float]) -> None:
        self.metric_sources[name] = fn

    async def run(self) -> None:
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval_s)
            t = time.perf_counter()
            for name, fn in self.queue_sources.items():
                try:
                    self.bus.queue_samples.put_nowait(QueueSample(t=t, name=name, value=int(fn())))
                except asyncio.QueueFull:
                    pass
            for name, fn in self.metric_sources.items():
                try:
                    self.bus.metric_samples.put_nowait(MetricSample(t=t, name=name, value=float(fn())))
                except asyncio.QueueFull:
                    pass

    def stop(self) -> None:
        self._running = False
