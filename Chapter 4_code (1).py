from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random


@dataclass(frozen=True)
class Task:
    task_id: int
    size: float            # abstract work units
    risk: float            # 0..1 multiplier on uncertainty
    deadline: int          # logical deadline in ticks


@dataclass
class Award:
    task: Task
    winner_id: int
    payment: float
    bids: Dict[int, float]


@dataclass
class Completion:
    task_id: int
    worker_id: int
    finished_at: int
    cost_incurred: float


class WorkerAgent:
    """
    Worker with private cost structure and capacity constraints.
    The bidding function exposes only a scalar bid per task.
    """
    def __init__(
        self,
        worker_id: int,
        capacity_per_tick: float,
        base_cost: float,
        risk_aversion: float,
        seed: int = 0,
    ):
        self.worker_id = worker_id
        self.capacity_per_tick = capacity_per_tick
        self.base_cost = base_cost
        self.risk_aversion = risk_aversion
        self.rng = random.Random(seed + worker_id)

        self.backlog: List[Task] = []
        self.credits: float = 0.0
        self.completed: List[Completion] = []

    def _current_load(self) -> float:
        return sum(t.size for t in self.backlog)

    def marginal_cost(self, task: Task, now: int) -> float:
        """
        Private cost model used internally and not revealed directly.
        Cost increases with load, task size, lateness proxy, and risk multiplier.
        """
        load = self._current_load()
        load_factor = 1.0 + 0.15 * math.log1p(load)
        urgency = max(0, now + math.ceil(task.size / max(1e-9, self.capacity_per_tick)) - task.deadline)
        urgency_factor = 1.0 + 0.25 * urgency
        risk_factor = 1.0 + self.risk_aversion * task.risk
        return self.base_cost * task.size * load_factor * urgency_factor * risk_factor

    def bid(self, task: Task, now: int) -> float:
        """
        Bid equals private marginal cost plus a small stochastic term to model
        estimation noise. In a strategic setting, this method encodes strategy.
        """
        c = self.marginal_cost(task, now)
        noise = self.rng.uniform(-0.01, 0.01) * c
        return max(0.0, c + noise)

    def can_accept(self, task: Task) -> bool:
        """
        Feasibility constraint approximating admission control via backlog bound.
        """
        return self._current_load() + task.size <= 4.0 * self.capacity_per_tick

    def accept_award(self, award: Award) -> None:
        self.backlog.append(award.task)
        self.credits += award.payment

    def tick(self, now: int) -> List[Completion]:
        """
        Simulate execution for one tick. Work is applied to the head-of-line task.
        Completion cost is computed at completion time using the current
        marginal-cost model; this is a modeling simplification and does not
        represent a stored acceptance-time cost.
        """
        if not self.backlog:
            return []

        cap = self.capacity_per_tick
        completions: List[Completion] = []

        task = self.backlog[0]
        if task.size <= cap:
            incurred = self.marginal_cost(task, now)
            self.backlog.pop(0)
            comp = Completion(task.task_id, self.worker_id, now, incurred)
            self.completed.append(comp)
            completions.append(comp)
        else:
            reduced = Task(task.task_id, task.size - cap, task.risk, task.deadline)
            self.backlog[0] = reduced

        return completions


class ReverseSecondPriceMarket:
    """
    For each task, selects the feasible lowest bidder and pays the second-lowest
    feasible bid. If fewer than two feasible bids exist, pays the winning bid.
    """
    def __init__(self, workers: List[WorkerAgent]):
        self.workers: Dict[int, WorkerAgent] = {w.worker_id: w for w in workers}
        self.awards: List[Award] = []
        self.time: int = 0

    def clear_task(self, task: Task) -> Optional[Award]:
        bids: Dict[int, float] = {}
        feasible: List[Tuple[float, int]] = []

        for wid, w in self.workers.items():
            if w.can_accept(task):
                b = w.bid(task, self.time)
                bids[wid] = b
                feasible.append((b, wid))

        if not feasible:
            return None

        feasible.sort()
        winner_bid, winner_id = feasible[0]

        if len(feasible) >= 2:
            payment = feasible[1][0]
        else:
            payment = winner_bid

        award = Award(task=task, winner_id=winner_id, payment=payment, bids=bids)
        self.awards.append(award)
        return award

    def allocate(self, tasks: List[Task]) -> List[Award]:
        out: List[Award] = []
        for task in tasks:
            award = self.clear_task(task)
            if award is not None:
                self.workers[award.winner_id].accept_award(award)
                out.append(award)
        return out

    def tick(self) -> List[Completion]:
        self.time += 1
        comps: List[Completion] = []
        for w in self.workers.values():
            comps.extend(w.tick(self.time))
        return comps


def demo_run() -> None:
    random.seed(1)

    workers = [
        WorkerAgent(worker_id=0, capacity_per_tick=3.0, base_cost=1.0, risk_aversion=0.6, seed=10),
        WorkerAgent(worker_id=1, capacity_per_tick=2.5, base_cost=0.9, risk_aversion=0.9, seed=10),
        WorkerAgent(worker_id=2, capacity_per_tick=2.0, base_cost=0.8, risk_aversion=0.4, seed=10),
        WorkerAgent(worker_id=3, capacity_per_tick=3.5, base_cost=1.2, risk_aversion=0.3, seed=10),
    ]
    market = ReverseSecondPriceMarket(workers)

    tasks: List[Task] = []
    for tid in range(18):
        size = random.choice([1.0, 2.0, 3.0, 4.0])
        risk = random.choice([0.0, 0.2, 0.5])
        deadline = random.randint(4, 10)
        tasks.append(Task(task_id=tid, size=size, risk=risk, deadline=deadline))

    wave1 = tasks[:8]
    wave2 = tasks[8:14]
    wave3 = tasks[14:]

    market.allocate(wave1)
    for _ in range(4):
        market.tick()

    market.allocate(wave2)
    for _ in range(4):
        market.tick()

    market.allocate(wave3)
    for _ in range(6):
        market.tick()

    for w in workers:
        print(
            f"worker={w.worker_id} credits={w.credits:.2f} "
            f"completed={len(w.completed)} backlog_load={sum(t.size for t in w.backlog):.1f}"
        )


if __name__ == "__main__":
    demo_run()
