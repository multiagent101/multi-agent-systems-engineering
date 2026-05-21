# Multi-Agent Systems Engineering — Companion Code

Reference implementations for **Multi-Agent Systems Engineering** by Cannon T. Hale.
Book on Amazon: https://www.amazon.com/dp/B0GQ3XBCHK

Every listing in the book has a corresponding file in this repository. All code runs against Python's standard library — no external packages required.

---

## Requirements

- Python 3.10 or later
- No third-party packages — all implementations use asyncio, threading, dataclasses, random, and other standard library modules only

Verify your version:

    python --version

---

## Quick start

    git clone https://github.com/multiagent101/multi-agent-systems-engineering.git
    cd multi-agent-systems-engineering
    python "Chapter 1_code (1).py"

Each file is self-contained and can be run independently, except Chapter 11 Listing 11.2 which depends on Chapter 11 Listing 11.1 (see note below).

---

## File map

| File | Chapter | What it demonstrates |
|------|---------|----------------------|
| Chapter 1_code (1).py | 1 | Minimal synchronous multi-agent system: agent abstraction, environment, broadcast communication |
| Chapter 2_code (1).py | 2 | Async coordinator-worker system with shared task queue and result aggregation |
| Chapter 3_code (1).py | 3 | Raft-style replicated state machine: leader election, log replication, quorum commitment |
| Chapter 4_code (1).py | 4 | Reverse second-price procurement auction with capacity constraints and payment ledger |
| Chapter 5.2_Listing 5.2code (1).py | 5.2 | Coordination and end-to-end latency measurement with p50/p95/p99 summaries |
| Chapter 5.3_Listing 5.3_code (1).py | 5.3 | Message counting middleware and bandwidth usage logging |
| Chapter 5.3_listing 5.4_code (1).py | 5.3 | Communication overhead visualization |
| Chapter 5.3_listing 5.5_code (1).py | 5.3 | Extended transport instrumentation |
| Chapter 5.4_Listing 5.6_code (1).py | 5.4 | Node crash simulation and recovery time measurement |
| Chapter 5.4_Listing 5.7_code (1).py | 5.4 | Network delay injection |
| Chapter 5.4_Listing 5.8_code (1).py | 5.4 | Recovery meter: throughput time series after injected fault |
| Chapter 5.5_Listing 5.9_code (1).py | 5.5 | Saturation sweep, resource contention profiling, horizontal scaling experiments |
| Chapter 5.6_Listing 5.10_code (1).py | 5.6 | Automated recommendation algorithm: semantic gating, performance gating, scoring |
| Chapter 6_Listing 6.1_code (1).py | 6 | Shared-state layer with versioning, conflict resolution, and consistency enforcement |
| Chapter 6_Listing 6.2_code (1).py | 6 | Failure injection: inconsistency scenarios and performance implications |
| Chapter 7_Listing 7.1_code (1).py | 7 | Feedback loop and deadlock detection mechanisms |
| Chapter 7_Listing 7.2_code (1).py | 7 | Oscillation detection and non-stationarity instrumentation |
| Chapter 7_Listing 7.3_code (1).py | 7 | Mitigation strategies: backoff, rate limiting, admission control |
| Chapter 8.1_Listing 8.1_code (1).py | 8 | Incentive alignment simulation across centralized, peer, and market regimes |
| Chapter 9_Listisng 9.1_code (1).py | 9 | Minimal MARL environment (CTDE paradigm) |
| Chapter 9_Listing 9.2_code (1).py | 9 | Training script |
| Chapter 9_Listing 9.3_code (1).py | 9 | Evaluation benchmarks |
| Chapter 9_Listing 9.4_code (1).py | 9 | Extended MARL experiment |
| Chapter 10_Listing 10.1_code (1).py | 10 | Structured trace propagation and span exporter |
| Chapter 10_Listing 10.2_code (1).py | 10 | Message logging middleware |
| Chapter 10_Listing 10.3_code (1).py | 10 | Execution graph visualization |
| Chapter 10_Listing 10.4_code (1).py | 10 | Full instrumentation stack |
| Chapter 10_Listing 10.5_code (1).py | 10 | Debugging real failure scenarios |
| Chapter 11_Listing 11.1_code (1).py | 11 | Resilience primitives: RetryPolicy, CircuitBreaker, Bulkhead, IdempotencyStore, LeaseManager, Outbox |
| Chapter 11_Listing 11.2_code (1).py | 11 | Failure simulation: tool degradation with and without resilience patterns |
| Chapter 12_Listing 12.1_code (1).py | 12 | Production benchmark harness: throughput, latency profiling, communication overhead |
| Chapter 13_Listing 13.1_code (1).py | 13 | Complete end-to-end production-grade multi-agent system with leases, idempotency, tracing |
| Chapter 14_Listing 14.1_docker-compose.yml | 14 | Minimal container topology: coordinator and worker services |
| Chapter 15_Listing 15.1_code (1).py | 15 | Risk-aware architecture selection: semantic gate, risk gate, cost-complexity scoring |

---

## Important: Chapter 11 Listing 11.2 dependency

Chapter 11 Listing 11.2 imports from Chapter 11 Listing 11.1 using the alias fixed_code.
To run it, copy Listing 11.1 into the same directory as a file named fixed_code.py:

    cp "Chapter 11_Listing 11.1_code (1).py" fixed_code.py
    python "Chapter 11_Listing 11.2_code (1).py"

---

## Important: Chapter 5 benchmark dependency

The Chapter 5 listings (5.3 to 5.5) import from Chapter 5.2, which defines the shared transport and measurement classes.
Run all Chapter 5 files from the same directory. The import alias is latency_bench:

    cp "Chapter 5.2_Listing 5.2code (1).py" latency_bench.py
    python "Chapter 5.5_Listing 5.9_code (1).py"

---

## Notes on implementation scope

- Chapter 3 (Raft): crash faults are not simulated and persistence is not implemented. Safety arguments hold only under crash-free execution. This is stated explicitly in the book.
- Chapter 4 (Market): credits record payments as an accounting trace and are not conserved. Lease expiration and re-auctioning are not implemented. Both are stated in the book.
- Chapter 9 (MARL): the environment is a minimal in-process simulation demonstrating the CTDE paradigm.

These are intentional scope boundaries documented in the corresponding chapters, not bugs.

---

## Issues

If you find a discrepancy between the book text and the code, open a GitHub issue with the chapter number, listing number, and a description of what you expected versus what you observed.
