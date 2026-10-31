# Track G: xv6 Kernel Optimization

## Objective
Measure whether a lottery scheduler gives proportional CPU shares and lower interactive latency than round robin without reducing throughput.

## Baseline
<!-- Describe the baseline method or initial unoptimized benchmark -->

## Formulation
- **Decision variables:** The ticket counts $t_i$ assigned to each process $i$ and the scheduling policy (round robin [0] or lottery [1]).
- **Objective values:**
  1. Proportional-share error:
     $$E = \frac{1}{2} \sum_i \left| s_i - \frac{t_i}{T} \right|$$
     where $s_i$ is the measured CPU share and $T = \sum_i t_i$ is the total number of tickets across all active processes.
  2. Average wake-up latency of an interactive process.
  3. Makespan of a fixed amount of CPU work.
- **Constraint:** The makespan must not grow by more than a few percent under lottery scheduling compared to round robin.

## Methods
- **Lottery Scheduling:** At each scheduling quantum, a random winning ticket is drawn uniformly using a 32-bit xorshift pseudo-random number generator; the process whose cumulative ticket interval contains the winning ticket is selected to run.
- **System Calls:**
  - `settickets(n)`: Sets the number of tickets allocated to the calling process.
  - `getpinfo(ps)`: Gathers process metrics including PID, in-use flag, ticket count, and scheduling ticks (`sched_count`).
  - `setsched(policy)`: Dynamically toggles between round robin (0) and lottery (1) scheduling policies.
- **Workloads:**
  - `W1 (Proportional Share)`: Three CPU-bound spinning children with 30, 20, and 10 tickets running for 300 ticks; metrics reported: `t30_ticks`, `t20_ticks`, `t10_ticks`, and `share_error_permille`.
  - `W2 (Equal Tickets, Fixed Work)`: Four children with 10 tickets each running fixed spin work (calibrated via `iters_per_tick * 100`); metrics reported: `makespan_ticks` and completion time `spread_ticks`.
  - `W3 (Interactive Latency next to Hogs)`: Three CPU-bound hogs (10 tickets each) competing with one interactive child (100 tickets) sleeping and spinning across 40 cycles; metrics reported: `lat_sum`, `lat_max`, `lat_count`, and `avg_latency_x100`.
- **Repetitions:** Each workload is executed for 3 repetitions (`REPS = 3`) under each policy from a clean state.
- **Correctness Checks:** Verified by `lotterytest` (syscall validation, ticket allocation, proportional quanta ratio) and standard xv6 `usertests` passing under both policies.

## How to run
See [scripts/README_RUN.md](scripts/README_RUN.md) for full instructions on building, running benchmarks, running tests in QEMU, and creating diff patches.

## Results
<!-- Experimental outcomes, comparison tables, and plots saved in results/ -->

## Correctness check
<!-- Verification steps, assertions, unit tests, and theoretical invariants checked -->

## Limitations
<!-- Computational bottlenecks, approximations, assumptions, and failure modes -->
