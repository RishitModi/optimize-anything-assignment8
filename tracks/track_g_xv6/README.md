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
Raw outputs are in results/track_g/ (benchrun_output.txt, lotterytest_output.txt, usertests_rr.txt, usertests_lottery.txt, diffstat.txt); the patch is patches/lottery_scheduler.diff. QEMU with CPUS=1, 3 repetitions per workload and policy, a tick is one timer quantum. W1 proportional share (tickets 30/20/10, 300 ticks): round robin gave 100/100/100 quanta (error 166 per mille); lottery gave on average 152.7/108.3/44.7 (shares 49.9, 35.4, 14.6 percent versus targets 50.0, 33.3, 16.7) with a mean error of 43 per mille (74 percent lower). W2 equal tickets, fixed work: makespan 13.3 ticks (round robin) versus 13.7 (lottery), within run-to-run variation; completion-time spread 2.0 versus 4.3 ticks. W3 interactive latency: average 1.00 tick (round robin) versus 0.29 (lottery), worst case 1 versus 2 ticks.

## Correctness check
- The modified kernel boots.
- `lotterytest` passes all four checks (argument validation, policy switching, `getpinfo`, 100-versus-1 ticket proportional share).
- `usertests` passes under round robin (policy 0) and under lottery (policy 1) (see the two usertests files).
- The policy 0 code path is the original scheduler with an added counter.

## Limitations
- One kernel feature only.
- Single CPU (CPUS=1), the multi-CPU case under `ptable.lock` was not tested.
- 3 repetitions and a coarse tick (about 10 ms nominal, not calibrated).
- The W2 workload was much shorter than designed (Failure Log case 24).
- Lottery scheduling gives proportional shares only on average and has a worse worst-case latency than round robin.
