# Listing 15.1 — Risk-aware selection algorithm integrating cost and complexity.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Tuple


Arch = Literal["centralized", "peer", "market"]


@dataclass(frozen=True)
class Latency:
    p99_ms: float


@dataclass(frozen=True)
class Throughput:
    goodput_tps: float


@dataclass(frozen=True)
class Overhead:
    messages_per_task: float
    bytes_per_task: float


@dataclass(frozen=True)
class Evidence:
    coord: Latency
    e2e: Latency
    throughput: Throughput
    overhead: Overhead
    stale_completion_rate: Optional[float] = None  # rejected stale completions / total completions (None if unmeasured)


@dataclass(frozen=True)
class PerturbationEvidence:
    overhead: Overhead
    throughput: Throughput
    coord: Optional[Latency] = None
    e2e: Optional[Latency] = None


@dataclass(frozen=True)
class Constraints:
    requires_strong_ordering: bool = False
    must_survive_control_plane_crash: bool = False
    min_goodput_tps: float = 0.0
    max_coord_p99_ms: Optional[float] = None
    max_e2e_p99_ms: Optional[float] = None


@dataclass(frozen=True)
class RiskThresholds:
    max_overhead_amplification: float = 2.0
    min_latency_slack: float = 0.10
    max_stale_completion_rate: float = 0.01
    min_throughput_headroom: float = 0.0  # required relative headroom vs min_goodput_tps


@dataclass(frozen=True)
class CostParams:
    # Unit costs are optional; when absent, objective remains comparative.
    cost_per_msg: float = 0.0
    cost_per_byte: float = 0.0
    cost_per_tps_shortfall: float = 1.0


@dataclass(frozen=True)
class ComplexityIndex:
    # Architecture-level complexity coefficients for comparative penalty.
    centralized: float = 1.0
    peer: float = 1.6
    market: float = 1.5


@dataclass
class Decision:
    selected: Optional[Arch]
    ranked: List[Tuple[Arch, float]]
    reasons: Dict[Arch, List[str]] = field(default_factory=dict)


def _slack(bound_ms: Optional[float], p99_ms: float) -> Optional[float]:
    if bound_ms is None or bound_ms <= 0:
        return None
    return (bound_ms - p99_ms) / bound_ms


def _amp_ratio(base: float, pert: float) -> float:
    denom = max(1e-9, base)
    return pert / denom


def select_architecture(
    evidence: Dict[Arch, Evidence],
    constraints: Constraints,
    risk: RiskThresholds,
    cost: CostParams = CostParams(),
    complexity: ComplexityIndex = ComplexityIndex(),
    perturbations: Optional[Dict[Arch, PerturbationEvidence]] = None,
) -> Decision:
    reasons: Dict[Arch, List[str]] = {a: [] for a in evidence.keys()}
    feasible: Dict[Arch, bool] = {a: True for a in evidence.keys()}

    # Semantic gating
    for arch in list(evidence.keys()):
        if constraints.requires_strong_ordering and arch == "market":
            feasible[arch] = False
            reasons[arch].append("gated: strong ordering required; market requires an explicit ordering authority")
        if constraints.must_survive_control_plane_crash and arch in ("centralized", "market"):
            feasible[arch] = False
            reasons[arch].append("gated: control-plane crash survival required; requires an HA authority path")

    # Performance gating
    for arch, ev in evidence.items():
        if not feasible[arch]:
            continue
        if constraints.min_goodput_tps > 0 and ev.throughput.goodput_tps < constraints.min_goodput_tps:
            feasible[arch] = False
            reasons[arch].append("gated: goodput below minimum")
        if constraints.max_coord_p99_ms is not None and ev.coord.p99_ms > constraints.max_coord_p99_ms:
            feasible[arch] = False
            reasons[arch].append("gated: coordination p99 exceeds bound")
        if constraints.max_e2e_p99_ms is not None and ev.e2e.p99_ms > constraints.max_e2e_p99_ms:
            feasible[arch] = False
            reasons[arch].append("gated: end-to-end p99 exceeds bound")

    # Risk gating and scoring
    scores: Dict[Arch, float] = {}
    for arch, ev in evidence.items():
        if not feasible[arch]:
            continue

        s_coord = _slack(constraints.max_coord_p99_ms, ev.coord.p99_ms)
        s_e2e = _slack(constraints.max_e2e_p99_ms, ev.e2e.p99_ms)
        if s_coord is not None and s_coord < risk.min_latency_slack:
            feasible[arch] = False
            reasons[arch].append("gated: insufficient coordination latency slack")
            continue
        if s_e2e is not None and s_e2e < risk.min_latency_slack:
            feasible[arch] = False
            reasons[arch].append("gated: insufficient end-to-end latency slack")
            continue

        if constraints.min_goodput_tps > 0 and risk.min_throughput_headroom > 0.0:
            headroom = (ev.throughput.goodput_tps - constraints.min_goodput_tps) / max(1e-9, constraints.min_goodput_tps)
            if headroom < risk.min_throughput_headroom:
                feasible[arch] = False
                reasons[arch].append("gated: insufficient throughput headroom")
                continue

        if ev.stale_completion_rate is None:
            feasible[arch] = False
            reasons[arch].append("gated: stale completion rate unmeasured")
            continue
        if ev.stale_completion_rate > risk.max_stale_completion_rate:
            feasible[arch] = False
            reasons[arch].append("gated: stale completion rate exceeds threshold")
            continue

        pert = perturbations.get(arch) if perturbations else None
        if pert is not None:
            amp_m = _amp_ratio(ev.overhead.messages_per_task, pert.overhead.messages_per_task)
            amp_b = _amp_ratio(ev.overhead.bytes_per_task, pert.overhead.bytes_per_task)
            if amp_m > risk.max_overhead_amplification or amp_b > risk.max_overhead_amplification:
                feasible[arch] = False
                reasons[arch].append("gated: overhead amplification exceeds threshold")
                continue

            # If perturbed latencies are provided, apply slack gating there as well.
            if pert.coord is not None and constraints.max_coord_p99_ms is not None:
                s2 = _slack(constraints.max_coord_p99_ms, pert.coord.p99_ms)
                if s2 is not None and s2 < risk.min_latency_slack:
                    feasible[arch] = False
                    reasons[arch].append("gated: insufficient coordination slack under perturbation")
                    continue
            if pert.e2e is not None and constraints.max_e2e_p99_ms is not None:
                s2 = _slack(constraints.max_e2e_p99_ms, pert.e2e.p99_ms)
                if s2 is not None and s2 < risk.min_latency_slack:
                    feasible[arch] = False
                    reasons[arch].append("gated: insufficient end-to-end slack under perturbation")
                    continue

            reasons[arch].append(f"risk: overhead_amp_msgs={amp_m:.2f} overhead_amp_bytes={amp_b:.2f}")

        if s_coord is not None:
            reasons[arch].append(f"slack: coord={s_coord:.2f}")
        if s_e2e is not None:
            reasons[arch].append(f"slack: e2e={s_e2e:.2f}")

        # Cost objective: overhead + throughput shortfall + complexity penalty.
        overhead_cost = (cost.cost_per_msg * ev.overhead.messages_per_task) + (cost.cost_per_byte * ev.overhead.bytes_per_task)
        shortfall = max(0.0, constraints.min_goodput_tps - ev.throughput.goodput_tps)
        shortfall_cost = cost.cost_per_tps_shortfall * shortfall

        comp_pen = getattr(complexity, arch)

        total_cost = overhead_cost + shortfall_cost + comp_pen
        scores[arch] = -total_cost

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    selected = ranked[0][0] if ranked else None
    return Decision(selected=selected, ranked=ranked, reasons=reasons)
