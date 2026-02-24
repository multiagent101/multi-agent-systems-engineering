# Listing 5.10 — Architecture recommendation algorithm: gating + scoring + explanation.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal


Arch = Literal["centralized", "peer", "market"]


@dataclass(frozen=True)
class LatencyEvidence:
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(frozen=True)
class CommEvidence:
    messages_per_task: float
    overhead_bytes_per_task: float


@dataclass(frozen=True)
class RobustnessEvidence:
    worker_crash_recovery_s: Optional[float]          # None if not measured
    control_plane_crash_recovery_s: Optional[float]   # None if not supported/undefined
    delay_injection_recovery_s: Optional[float]       # None if not measured


@dataclass(frozen=True)
class ScalabilityEvidence:
    throughput_tps: float
    saturation_arrival_rate_tps: Optional[float]      # arrival rate at which saturation begins
    contention_signature: Optional[str] = None        # optional classifier label


@dataclass(frozen=True)
class BenchmarkEvidence:
    coordination: LatencyEvidence
    end_to_end: LatencyEvidence
    comm: CommEvidence
    robustness: RobustnessEvidence
    scalability: ScalabilityEvidence


@dataclass(frozen=True)
class Constraints:
    # Semantic constraints
    requires_strong_ordering: bool = False
    must_survive_control_plane_crash: bool = False

    # Performance constraints
    target_arrival_rate_tps: float = 0.0
    min_throughput_tps: float = 0.0

    max_coord_p99_ms: Optional[float] = None
    max_e2e_p99_ms: Optional[float] = None

    # Communication constraints (optional)
    max_messages_per_task: Optional[float] = None
    max_overhead_bytes_per_task: Optional[float] = None

    # Robustness constraints (optional)
    max_recovery_s: Optional[float] = None


@dataclass(frozen=True)
class Weights:
    coordination_latency: float = 0.25
    end_to_end_latency: float = 0.30
    throughput: float = 0.25
    communication: float = 0.10
    robustness: float = 0.10


@dataclass
class CandidateScore:
    arch: Arch
    feasible: bool
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    tight_metrics: List[str] = field(default_factory=list)


def _utility_lower_is_better(value: float, target: float, limit: float) -> float:
    """
    Returns 1.0 at or below target, 0.0 at or above limit, linear in between.
    """
    if limit <= target:
        return 0.0 if value > target else 1.0
    if value <= target:
        return 1.0
    if value >= limit:
        return 0.0
    return (limit - value) / (limit - target)


def _utility_higher_is_better(value: float, target: float, limit: float) -> float:
    """
    Returns 1.0 at or above target, 0.0 at or below limit, linear in between.
    """
    if target <= limit:
        return 0.0 if value < target else 1.0
    if value >= target:
        return 1.0
    if value <= limit:
        return 0.0
    return (value - limit) / (target - limit)


def _near_violation(value: float, bound: float, margin: float = 0.05) -> bool:
    """
    Flags fragility when value is within margin fraction of the bound.
    margin=0.05 means within 5% of the bound.
    """
    if bound <= 0:
        return False
    return value >= (1.0 - margin) * bound


def score_candidates(
    evidence_by_arch: Dict[Arch, BenchmarkEvidence],
    constraints: Constraints,
    weights: Weights = Weights(),
) -> List[CandidateScore]:
    out: List[CandidateScore] = []

    # Require explicit operating point for meaningful throughput scoring.
    tp_target = max(constraints.target_arrival_rate_tps, constraints.min_throughput_tps, 0.0)

    for arch, ev in evidence_by_arch.items():
        cs = CandidateScore(arch=arch, feasible=True)

        # --- Semantic gating ---
        if constraints.requires_strong_ordering:
            # Under this benchmark's completion semantics, require explicit evidence of control-plane crash recovery
            # to treat completion ordering/durability as supported at the system boundary.
            if ev.robustness.control_plane_crash_recovery_s is None:
                cs.feasible = False
                cs.reasons.append("gated: strong ordering required; durable ordered completion not supported by evidence")

        if constraints.must_survive_control_plane_crash:
            r = ev.robustness.control_plane_crash_recovery_s
            if r is None:
                cs.feasible = False
                cs.reasons.append("gated: must survive control-plane crash; recovery undefined in evidence")

        # --- Hard performance gating ---
        if constraints.min_throughput_tps > 0 and ev.scalability.throughput_tps < constraints.min_throughput_tps:
            cs.feasible = False
            cs.reasons.append("gated: throughput below minimum")

        if constraints.max_coord_p99_ms is not None and ev.coordination.p99_ms > constraints.max_coord_p99_ms:
            cs.feasible = False
            cs.reasons.append("gated: coordination p99 exceeds bound")

        if constraints.max_e2e_p99_ms is not None and ev.end_to_end.p99_ms > constraints.max_e2e_p99_ms:
            cs.feasible = False
            cs.reasons.append("gated: end-to-end p99 exceeds bound")

        if constraints.max_messages_per_task is not None and ev.comm.messages_per_task > constraints.max_messages_per_task:
            cs.feasible = False
            cs.reasons.append("gated: messages per task exceeds bound")

        if constraints.max_overhead_bytes_per_task is not None and ev.comm.overhead_bytes_per_task > constraints.max_overhead_bytes_per_task:
            cs.feasible = False
            cs.reasons.append("gated: overhead bytes per task exceeds bound")

        if constraints.max_recovery_s is not None:
            candidates = [
                x for x in [
                    ev.robustness.worker_crash_recovery_s,
                    ev.robustness.control_plane_crash_recovery_s,
                    ev.robustness.delay_injection_recovery_s,
                ] if x is not None
            ]
            if not candidates:
                cs.feasible = False
                cs.reasons.append("gated: recovery bound specified but recovery evidence is missing")
            elif max(candidates) > constraints.max_recovery_s:
                cs.feasible = False
                cs.reasons.append("gated: recovery time exceeds bound")

        if not cs.feasible:
            out.append(cs)
            continue

        # --- Utility scoring ---
        # Require explicit bounds/budgets to provide a common yardstick across candidates.
        u_coord = 0.5
        if constraints.max_coord_p99_ms is not None:
            coord_target = constraints.max_coord_p99_ms * 0.70
            coord_limit = constraints.max_coord_p99_ms
            u_coord = _utility_lower_is_better(ev.coordination.p99_ms, coord_target, coord_limit)

        u_e2e = 0.5
        if constraints.max_e2e_p99_ms is not None:
            e2e_target = constraints.max_e2e_p99_ms * 0.70
            e2e_limit = constraints.max_e2e_p99_ms
            u_e2e = _utility_lower_is_better(ev.end_to_end.p99_ms, e2e_target, e2e_limit)

        u_tp = 0.5
        if tp_target > 0:
            tp_limit = 0.5 * tp_target
            u_tp = _utility_higher_is_better(ev.scalability.throughput_tps, tp_target, tp_limit)

            # Prefer headroom versus saturation when (\lambda^*) evidence is available.
            lam_star = ev.scalability.saturation_arrival_rate_tps
            if lam_star is not None and lam_star > 0 and tp_target >= lam_star:
                cs.tight_metrics.append("saturation_margin")
                u_tp *= 0.5

        u_comm = 0.5
        if constraints.max_messages_per_task is not None or constraints.max_overhead_bytes_per_task is not None:
            u_parts: List[float] = []
            if constraints.max_messages_per_task is not None:
                msg_target = constraints.max_messages_per_task * 0.70
                msg_limit = constraints.max_messages_per_task
                u_parts.append(_utility_lower_is_better(ev.comm.messages_per_task, msg_target, msg_limit))
            if constraints.max_overhead_bytes_per_task is not None:
                b_target = constraints.max_overhead_bytes_per_task * 0.70
                b_limit = constraints.max_overhead_bytes_per_task
                u_parts.append(_utility_lower_is_better(ev.comm.overhead_bytes_per_task, b_target, b_limit))
            u_comm = sum(u_parts) / max(1, len(u_parts))

        u_rob = 0.5
        recs = [x for x in [
            ev.robustness.worker_crash_recovery_s,
            ev.robustness.control_plane_crash_recovery_s,
            ev.robustness.delay_injection_recovery_s,
        ] if x is not None]
        if constraints.max_recovery_s is not None and recs:
            worst_rec = max(recs)
            rec_limit = constraints.max_recovery_s
            rec_target = rec_limit * 0.60
            u_rob = _utility_lower_is_better(worst_rec, rec_target, rec_limit)
        elif recs:
            worst_rec = max(recs)
            rec_limit = worst_rec * 1.50
            rec_target = rec_limit * 0.60
            u_rob = _utility_lower_is_better(worst_rec, rec_target, rec_limit)
        else:
            cs.reasons.append("note: robustness evidence incomplete; robustness treated as neutral")

        cs.score = (
            weights.coordination_latency * u_coord +
            weights.end_to_end_latency * u_e2e +
            weights.throughput * u_tp +
            weights.communication * u_comm +
            weights.robustness * u_rob
        )

        # Fragility penalties near bounds (prefer slack).
        penalty = 0.0
        if constraints.max_coord_p99_ms is not None and _near_violation(ev.coordination.p99_ms, constraints.max_coord_p99_ms):
            penalty += 0.05
            cs.tight_metrics.append("coord_p99")
        if constraints.max_e2e_p99_ms is not None and _near_violation(ev.end_to_end.p99_ms, constraints.max_e2e_p99_ms):
            penalty += 0.05
            cs.tight_metrics.append("e2e_p99")
        if constraints.max_messages_per_task is not None and _near_violation(ev.comm.messages_per_task, constraints.max_messages_per_task):
            penalty += 0.03
            cs.tight_metrics.append("messages_per_task")
        if constraints.max_overhead_bytes_per_task is not None and _near_violation(ev.comm.overhead_bytes_per_task, constraints.max_overhead_bytes_per_task):
            penalty += 0.03
            cs.tight_metrics.append("overhead_bytes_per_task")
        if constraints.max_recovery_s is not None and recs and _near_violation(max(recs), constraints.max_recovery_s):
            penalty += 0.04
            cs.tight_metrics.append("recovery_s")

        cs.score = max(0.0, cs.score - penalty)

        out.append(cs)

    # Sort feasible candidates first, then by score descending.
    out.sort(key=lambda x: (not x.feasible, -x.score, x.arch))
    return out


@dataclass(frozen=True)
class Recommendation:
    selected: Optional[Arch]
    ranked: List[CandidateScore]


def recommend_architecture(
    evidence_by_arch: Dict[Arch, BenchmarkEvidence],
    constraints: Constraints,
    weights: Weights = Weights(),
) -> Recommendation:
    ranked = score_candidates(evidence_by_arch, constraints, weights)
    selected = ranked[0].arch if ranked and ranked[0].feasible else None
    return Recommendation(selected=selected, ranked=ranked)
