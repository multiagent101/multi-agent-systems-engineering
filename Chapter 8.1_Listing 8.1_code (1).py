# Listing 8.1 — Incentive simulation: repeated procurement auctions with budgets and adaptive bidding.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import math
import random
import statistics


AuctionRule = Literal["first_price", "second_price"]
Strategy = Literal["truthful", "shaded", "withhold"]


@dataclass(frozen=True)
class Task:
    task_id: int
    size: float
    deadline: int


@dataclass
class Completion:
    task_id: int
    worker_id: int
    finished_at: int
    true_cost: float
    payment: float
    utility: float


@dataclass
class StepMetrics:
    tick: int
    cleared: int
    completed: int
    mean_price: float
    mean_true_cost: float
    price_volatility: float
    backlog_mean: float
    backlog_p95: float


class Agent:
    def __init__(
        self,
        agent_id: int,
        capacity_per_tick: float,
        base_cost: float,
        risk_aversion: float,
        strategy: Strategy,
        rng: random.Random,
        initial_credit: float = 0.0,
        credit_decay: float = 0.0,
    ):
        self.agent_id = agent_id
        self.capacity_per_tick = capacity_per_tick
        self.base_cost = base_cost
        self.risk_aversion = risk_aversion
        self.strategy = strategy
        self.rng = rng

        self.credit = initial_credit
        self.credit_decay = max(0.0, credit_decay)

        self.backlog: List[Task] = []
        self.completions: List[Completion] = []
        self._cost_accum: Dict[int, float] = {}

    def backlog_load(self) -> float:
        return sum(t.size for t in self.backlog)

    def marginal_cost(self, task: Task, now: int) -> float:
        load = self.backlog_load()
        load_factor = 1.0 + 0.15 * math.log1p(load)
        urgency = max(0, now + math.ceil(task.size / max(1e-9, self.capacity_per_tick)) - task.deadline)
        urgency_factor = 1.0 + 0.20 * urgency
        noise = self.rng.uniform(0.98, 1.02)
        return self.base_cost * task.size * load_factor * urgency_factor * (1.0 + self.risk_aversion * 0.2) * noise

    def bid(self, task: Task, now: int, expected_competition: float) -> float:
        c = self.marginal_cost(task, now)

        if self.strategy == "truthful":
            return max(0.0, c)

        if self.strategy == "shaded":
            # Simple shading heuristic:
            # when competition is high, shade downward to win; when low, shade upward to extract payment.
            # credit pressure increases willingness to win work.
            comp = max(0.1, min(2.0, expected_competition))
            credit_pressure = 1.0 / (1.0 + math.exp((self.credit - 5.0)))  # high when credit low
            win_bias = 0.92 - 0.08 * credit_pressure  # lower bid when credit low
            extract_bias = 1.05 + 0.05 / comp         # higher bid when competition low
            # Blend: prefer winning under high competition, extracting under low competition.
            alpha = min(1.0, comp / 1.5)
            b = (alpha * (c * win_bias)) + ((1.0 - alpha) * (c * extract_bias))
            return max(0.0, b)

        if self.strategy == "withhold":
            # Withhold capacity by pricing high when credit is high, effectively reducing participation.
            # Bid decreases when credit is low to regain eligibility under credit gating.
            avoid = 1.0 + 0.20 * math.tanh(self.credit / 10.0)
            need = 1.0 - 0.15 * math.tanh((5.0 - self.credit) / 5.0)
            b = c * max(0.7, avoid * need)
            return max(0.0, b)

        return max(0.0, c)

    def can_accept(self, task: Task) -> bool:
        # Local feasibility cap prevents unbounded backlog.
        return self.backlog_load() + task.size <= 4.0 * self.capacity_per_tick

    def accept(self, task: Task, payment: float) -> None:
        self.backlog.append(task)
        self.credit += payment

    def decay_credit(self) -> None:
        if self.credit_decay > 0.0:
            self.credit *= (1.0 - self.credit_decay)

    def tick(self, now: int) -> List[Completion]:
        self.decay_credit()
        if not self.backlog:
            return []
        cap = self.capacity_per_tick
        comps: List[Completion] = []
        t = self.backlog[0]
        work = min(cap, t.size)

        # Accumulate realized execution cost proportional to work done this tick.
        inc_cost = self.marginal_cost(Task(t.task_id, work, t.deadline), now)
        self._cost_accum[t.task_id] = self._cost_accum.get(t.task_id, 0.0) + inc_cost
        self.credit -= inc_cost

        if t.size <= cap:
            true_cost = self._cost_accum.pop(t.task_id, inc_cost)
            payment = 0.0  # filled by mechanism at award time; stored elsewhere
            self.backlog.pop(0)
            comps.append(Completion(t.task_id, self.agent_id, now, true_cost, payment, 0.0))
        else:
            self.backlog[0] = Task(t.task_id, t.size - cap, t.deadline)
        return comps


class ProcurementMarketSim:
    def __init__(
        self,
        agents: List[Agent],
        rule: AuctionRule,
        rng: random.Random,
        enforce_budget: bool = False,
        min_credit: float = -5.0,
        max_credit: float = 50.0,
    ):
        self.agents = agents
        self._agent_by_id: Dict[int, Agent] = {a.agent_id: a for a in agents}
        self.rule = rule
        self.rng = rng
        self.enforce_budget = enforce_budget
        self.min_credit = min_credit
        self.max_credit = max_credit

        self.now: int = 0
        self._award_payment: Dict[int, float] = {}  # task_id -> payment
        self._award_winner: Dict[int, int] = {}

        self.history_prices: List[float] = []
        self.history_true_costs: List[float] = []
        self.history_completed: List[Completion] = []

    def _expected_competition(self) -> float:
        # crude proxy: fraction of agents not overloaded
        feasible = sum(1 for a in self.agents if a.backlog_load() < 2.0 * a.capacity_per_tick)
        return max(0.1, feasible / max(1, len(self.agents)))

    def clear_task(self, task: Task) -> bool:
        comp = self._expected_competition()

        bids: List[Tuple[float, int]] = []
        true_costs: Dict[int, float] = {}
        for a in self.agents:
            if not a.can_accept(task):
                continue
            if self.enforce_budget and a.credit < self.min_credit:
                continue
            b = a.bid(task, self.now, expected_competition=comp)
            bids.append((b, a.agent_id))
            true_costs[a.agent_id] = a.marginal_cost(task, self.now)

        if not bids:
            return False

        bids.sort()
        winner_bid, winner_id = bids[0]
        payment = winner_bid
        if self.rule == "second_price":
            payment = bids[1][0] if len(bids) >= 2 else winner_bid

        winner = self._agent_by_id[winner_id]
        winner.accept(task, payment)
        if self.enforce_budget:
            winner.credit = min(winner.credit, self.max_credit)

        self._award_payment[task.task_id] = payment
        self._award_winner[task.task_id] = winner_id
        self.history_prices.append(payment)
        self.history_true_costs.append(true_costs[winner_id])
        return True

    def step(self, tasks: List[Task]) -> StepMetrics:
        cleared = 0
        for t in tasks:
            if self.clear_task(t):
                cleared += 1

        comps: List[Completion] = []
        self.now += 1
        for a in self.agents:
            out = a.tick(self.now)
            for c in out:
                # Attach award payment and realized utility based on stored award.
                pay = self._award_payment.get(c.task_id, 0.0)
                wid = self._award_winner.get(c.task_id, a.agent_id)
                if wid != a.agent_id:
                    # task migrated under failure would appear here; omitted in this reference sim
                    pay = 0.0
                util = pay - c.true_cost
                comps.append(Completion(c.task_id, a.agent_id, c.finished_at, c.true_cost, pay, util))
        completed = len(comps)
        self.history_completed.extend(comps)

        # Metrics
        mean_price = statistics.fmean(self.history_prices[-cleared:]) if cleared else 0.0
        mean_true = statistics.fmean(self.history_true_costs[-cleared:]) if cleared else 0.0
        # price volatility over recent window
        window = self.history_prices[-max(5, cleared):]
        vol = statistics.pstdev(window) if len(window) >= 2 else 0.0

        backlogs = [a.backlog_load() for a in self.agents]
        backlog_mean = statistics.fmean(backlogs) if backlogs else 0.0
        backlog_sorted = sorted(backlogs)
        backlog_p95 = backlog_sorted[int(0.95 * (len(backlog_sorted) - 1))] if backlogs else 0.0

        return StepMetrics(
            tick=self.now,
            cleared=cleared,
            completed=completed,
            mean_price=mean_price,
            mean_true_cost=mean_true,
            price_volatility=vol,
            backlog_mean=backlog_mean,
            backlog_p95=backlog_p95,
        )


def run_simulation(
    *,
    strategies: List[Strategy],
    seed: int = 3,
    rule: AuctionRule = "second_price",
    ticks: int = 200,
    arrival_rate: float = 3.0,
    enforce_budget: bool = True,
) -> Dict[str, float]:
    rng = random.Random(seed)

    def poisson_knuth(r: random.Random, lam: float) -> int:
        lam = max(0.0, lam)
        if lam == 0.0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= r.random()
        return k - 1

    agents: List[Agent] = []
    for i, strat in enumerate(strategies):
        agents.append(
            Agent(
                agent_id=i,
                capacity_per_tick=rng.choice([2.0, 2.5, 3.0]),
                base_cost=rng.choice([0.8, 0.9, 1.0, 1.1]),
                risk_aversion=rng.choice([0.3, 0.6, 0.9]),
                strategy=strat,
                rng=random.Random(seed + 100 + i),
                initial_credit=5.0,
                credit_decay=0.002,
            )
        )

    sim = ProcurementMarketSim(agents, rule=rule, rng=rng, enforce_budget=enforce_budget)

    task_id = 0
    metrics: List[StepMetrics] = []
    for _ in range(ticks):
        # Poisson arrivals per tick
        k = poisson_knuth(rng, arrival_rate)
        k = max(0, min(k, 10))
        tasks: List[Task] = []
        for _ in range(k):
            size = rng.choice([1.0, 2.0, 3.0, 4.0])
            deadline = sim.now + rng.randint(2, 8)
            tasks.append(Task(task_id=task_id, size=size, deadline=deadline))
            task_id += 1

        metrics.append(sim.step(tasks))

    # Aggregate behavioral metrics
    completions = sim.history_completed
    total_utility = sum(c.utility for c in completions)
    total_true_cost = sum(c.true_cost for c in completions)
    total_payment = sum(c.payment for c in completions)

    # Procurement spend proxy: realized work cost per payment unit (lower is better).
    true_cost_over_payment = (total_true_cost / max(1e-9, total_payment)) if total_payment > 0 else 0.0

    # Fairness proxy: completion distribution (including zero-completion agents).
    by_worker: Dict[int, int] = {a.agent_id: 0 for a in agents}
    for c in completions:
        by_worker[c.worker_id] = by_worker.get(c.worker_id, 0) + 1
    counts = list(by_worker.values()) or [0]
    fairness = (min(counts) / max(counts)) if max(counts) > 0 else 0.0

    price_vol = statistics.pstdev(sim.history_prices) if len(sim.history_prices) >= 2 else 0.0
    backlog_mean = statistics.fmean([m.backlog_mean for m in metrics]) if metrics else 0.0

    return {
        "completed": float(len(completions)),
        "total_true_cost": float(total_true_cost),
        "total_payment": float(total_payment),
        "total_utility": float(total_utility),
        "true_cost_over_payment": float(true_cost_over_payment),
        "fairness_min_over_max_completions": float(fairness),
        "price_volatility": float(price_vol),
        "mean_backlog": float(backlog_mean),
    }
