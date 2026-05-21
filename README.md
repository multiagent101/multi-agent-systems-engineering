Multi-Agent Systems Engineering — Companion Code
Reference implementations for Multi-Agent Systems Engineering by Cannon T. Hale.
Every listing in the book has a corresponding file in this repository. All code runs against Python's standard library — no external packages required.
---
Requirements
Python 3.10 or later
No third-party packages — all implementations use `asyncio`, `threading`, `dataclasses`, `random`, and other standard library modules only
Verify your version:
```bash
python --version
```
---
Quick start
```bash
git clone https://github.com/multiagent101/multi-agent-systems-engineering.git
cd multi-agent-systems-engineering
python "Chapter 1_code (1).py"
```
Each file is self-contained and can be run independently, except for Chapter 11 Listing 11.2 which depends on Chapter 11 Listing 11.1 (see note below).
---
File map — chapter by chapter
File	Chapter	What it demonstrates
`Chapter 1_code (1).py`	1 — Engineering Model	Minimal synchronous multi-agent system: agent abstraction, environment, broadcast communication
`Chapter 2_code (1).py`	2 — Centralized Orchestration	Async coordinator–worker system with shared task queue and result aggregation
`Chapter 3_code (1).py`	3 — Distributed & Peer	Raft-style replicated state machine: leader election, log replication, quorum commitment
`Chapter 4_code (1).py`	4 — Market & Incentive-Based	Reverse second-price procurement auction with capacity constraints and payment ledger
`Chapter 5.2_Listing 5.2code (1).py`	5.2 — Latency Benchmark	Coordination and end-to-end latency measurement with p50/p95/p99 summaries
`Chapter 5.3_Listing 5.3_code (1).py`	5.3 — Communication Cost	Message counting middleware and bandwidth usage logging
`Chapter 5.3_listing 5.4_code (1).py`	5.3 — Communication Cost	Communication overhead visualization
`Chapter 5.3_listing 5.5_code (1).py`	5.3 — Communication Cost	Extended transport instrumentation
`Chapter 5.4_Listing 5.6_code (1).py`	5.4 — Robustness Testing	Node crash simulation and recovery time measurement
`Chapter 5.4_Listing 5.7_code (1).py`	5.4 — Robustness Testing	Network delay injection
`Chapter 5.4_Listing 5.8_code (1).py`	5.4 — Robustness Testing	Recovery meter: throughput time series after injected fault
`Chapter 5.5_Listing 5.9_code (1).py`	5.5 — Scalability	Saturation sweep, resource contention profiling, horizontal scaling experiments
`Chapter 5.6_Listing 5.10_code (1).py`	5.6 — Architecture Selection	Automated recommendation algorithm: semantic gating → performance gating → scoring
`Chapter 6_Listing 6.1_code (1).py`	6 — State Management	Shared-state layer with versioning, conflict resolution, and consistency enforcement
`Chapter 6_Listing 6.2_code (1).py`	6 — State Management	Failure injection: inconsistency scenarios and performance implications
`Chapter 7_Listing 7.1_code (1).py`	7 — Stability & Emergent Failures	Feedback loop and deadlock detection mechanisms
`Chapter 7_Listing 7.2_code (1).py`	7 — Stability & Emergent Failures	Oscillation detection and non-stationarity instrumentation
`Chapter 7_Listing 7.3_code (1).py`	7 — Stability & Emergent Failures	Mitigation strategies: backoff, rate limiting, admission control
`Chapter 8.1_Listing 8.1_code (1).py`	8 — Incentives & Strategic Coordination	Incentive alignment simulation across centralized, peer, and market regimes
`Chapter 9_Listisng 9.1_code (1).py`	9 — MARL	Minimal MARL environment (CTDE paradigm)
`Chapter 9_Listing 9.2_code (1).py`	9 — MARL	Training script
`Chapter 9_Listing 9.3_code (1).py`	9 — MARL	Evaluation benchmarks
`Chapter 9_Listing 9.4_code (1).py`	9 — MARL	Extended MARL experiment
`Chapter 10_Listing 10.1_code (1).py`	10 — Observability	Structured trace propagation and span exporter
`Chapter 10_Listing 10.2_code (1).py`	10 — Observability	Message logging middleware
`Chapter 10_Listing 10.3_code (1).py`	10 — Observability	Execution graph visualization
`Chapter 10_Listing 10.4_code (1).py`	10 — Observability	Full instrumentation stack
`Chapter 10_Listing 10.5_code (1).py`	10 — Observability	Debugging real failure scenarios
`Chapter 11_Listing 11.1_code (1).py`	11 — Reliability	Resilience primitives: RetryPolicy, CircuitBreaker, Bulkhead, IdempotencyStore, LeaseManager, Outbox
`Chapter 11_Listing 11.2_code (1).py`	11 — Reliability	Failure simulation experiment: tool degradation with and without resilience patterns
`Chapter 12_Listing 12.1_code (1).py`	12 — Performance Engineering	Production benchmark harness: throughput, latency profiling, communication overhead
`Chapter 13_Listing 13.1_code (1).py`	13 — Reference System	Complete end-to-end production-grade multi-agent system with leases, idempotency, tracing
`Chapter 14_Listing 14.1_docker-compose.yml`	14 — Production Integration	Minimal container topology: coordinator + worker services
`Chapter 15_Listing 15.1_code (1).py`	15 — Risk Modeling	Risk-aware architecture selection algorithm: semantic gate → risk gate → cost–complexity scoring
---
Important: Chapter 11 Listing 11.2 dependency
`Chapter 11_Listing 11.2_code (1).py` imports from `Chapter 11_Listing 11.1_code (1).py` using the alias `fixed_code`. To run it, copy Listing 11.1 into the same directory as a file named `fixed_code.py`:
```bash
cp "Chapter 11_Listing 11.1_code (1).py" fixed_code.py
python "Chapter 11_Listing 11.2_code (1).py"
```
This mirrors the book's text, which states: "Reuse primitives from Listing 11.1." Listing 11.1 is the canonical source for `RetryPolicy`, `call_with_retry`, `CircuitBreaker`, `Bulkhead`, `IdempotencyStore`, and `LeaseManager`.
---
Chapter 5 benchmark dependency
The Chapter 5 benchmark listings (5.3–5.5) import from `Chapter 5.2_Listing 5.2code (1).py`, which defines the shared transport, worker, coordinator, and measurement classes. Run all Chapter 5 files from the same directory so the import resolves correctly. The import alias used is `latency_bench`:
```bash
cp "Chapter 5.2_Listing 5.2code (1).py" latency_bench.py
python "Chapter 5.5_Listing 5.9_code (1).py"
```
---
Running the Docker Compose topology (Chapter 14)
```bash
docker compose -f "Chapter 14_Listing 14.1_docker-compose.yml" up --scale worker=4
```
The compose file expects two environment variables:
`MAS_STATE_URL` — connection string for the shared state service (e.g., a Redis or PostgreSQL endpoint)
For local testing without a real state service, run the Chapter 13 reference system in-process instead:
```bash
python "Chapter 13_Listing 13.1_code (1).py"
```
---
What each part of the book covers
Part	Chapters	Focus
I	1	Engineering foundations: agent abstraction, state, policy, interface, environment, communication
II	2–5	Core architectures: centralized, peer/consensus, market — with benchmarks and selection algorithm
III	6–7	State management, consistency, deadlocks, feedback loops, emergent failures
IV	8–9	Incentive alignment, strategic coordination, multi-agent reinforcement learning
V	10–12	Production engineering: observability, fault tolerance, performance measurement
VI	13–15	End-to-end reference system, deployment, architecture risk modeling
---
Notes on implementation scope
Each listing is scoped as documented in the book:
Chapter 3 (Raft): crash faults are not simulated and persistence is not implemented. Safety arguments hold only under crash-free execution. This is stated explicitly in the book.
Chapter 4 (Market): credits record payments as an accounting trace and are not conserved across the system. Lease expiration and re-auctioning are not implemented. Both are stated explicitly in the book.
Chapter 9 (MARL): the environment is a minimal in-process simulation. It demonstrates the CTDE paradigm but is not a production RL training harness.
These are not bugs — they are intentional scope boundaries documented in the corresponding chapters.
---
Book
Multi-Agent Systems Engineering on Amazon
---
Issues
If you find a discrepancy between the book text and the code in this repository, open a GitHub issue with the chapter number, listing number, and a description of what you expected versus what you observed.
