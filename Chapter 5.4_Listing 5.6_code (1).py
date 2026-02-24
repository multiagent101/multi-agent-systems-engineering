# Listing 5.6 — Crash simulation: stop/restart endpoints under a scheduled fault plan.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

Factory = Callable[[], Awaitable[None]]  # coroutine factory for an endpoint run loop


@dataclass(frozen=True)
class CrashEvent:
    at_s: float                 # offset from run start
    endpoint_id: str
    down_for_s: float            # 0 means crash without restart


class ProcessRegistry:
    """
    Manages crashable endpoints by id.
    A registered endpoint is defined by a coroutine factory and an asyncio.Task.
    Restart re-invokes the factory; transport re-registration is the responsibility
    of the endpoint factory itself.
    """
    def __init__(self) -> None:
        self._factories: Dict[str, Factory] = {}
        self._tasks: Dict[str, asyncio.Task[Any]] = {}
        self._running: Dict[str, bool] = {}

    def register(self, endpoint_id: str, factory: Factory) -> None:
        self._factories[endpoint_id] = factory

    def is_running(self, endpoint_id: str) -> bool:
        return bool(self._running.get(endpoint_id, False))

    def start(self, endpoint_id: str) -> None:
        if endpoint_id not in self._factories:
            raise KeyError(f"unknown endpoint: {endpoint_id}")
        if self.is_running(endpoint_id):
            return
        task = asyncio.create_task(self._factories[endpoint_id]())
        self._tasks[endpoint_id] = task
        self._running[endpoint_id] = True

    async def stop(self, endpoint_id: str) -> None:
        task = self._tasks.get(endpoint_id)
        self._running[endpoint_id] = False
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            # crash-stop semantics: treat exceptions as terminal without propagation
            pass
        finally:
            self._tasks.pop(endpoint_id, None)

    async def restart(self, endpoint_id: str) -> None:
        await self.stop(endpoint_id)
        self.start(endpoint_id)


class CrashInjector:
    """
    Executes CrashEvent objects relative to a start timestamp.
    Each event is scheduled independently by absolute deadline.
    """
    def __init__(self, registry: ProcessRegistry) -> None:
        self.registry = registry
        self.events: list[CrashEvent] = []
        self.log: list[tuple[float, str, str]] = []  # (t, endpoint_id, action)

    def add(self, ev: CrashEvent) -> None:
        self.events.append(ev)

    async def _execute(self, ev: CrashEvent, run_start: float) -> None:
        now = time.perf_counter()
        wait = (run_start + ev.at_s) - now
        if wait > 0:
            await asyncio.sleep(wait)

        await self.registry.stop(ev.endpoint_id)
        self.log.append((time.perf_counter(), ev.endpoint_id, "crash"))

        if ev.down_for_s > 0:
            await asyncio.sleep(ev.down_for_s)
            await self.registry.restart(ev.endpoint_id)
            self.log.append((time.perf_counter(), ev.endpoint_id, "restart"))

    async def run(self, run_start: float) -> None:
        tasks = []
        for ev in self.events:
            tasks.append(asyncio.create_task(self._execute(ev, run_start)))
        if tasks:
            await asyncio.gather(*tasks)
