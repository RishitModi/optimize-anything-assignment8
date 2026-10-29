"""Parallel computing tools, worker routines, and discrete-event scheduling simulators.

IMPORTANT: OpenMP and BLAS threading are constrained to 1 before importing NumPy
to prevent worker core oversubscription during multi-process benchmarks.
"""

from __future__ import annotations

import os

# Constrain linear algebra threads to 1 before importing numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import collections
import heapq
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np


# ---------------------------------------------------------------------------
# CPU Burn & Calibration
# ---------------------------------------------------------------------------

def burn(n_iter: int) -> int:
    """Pure-Python CPU-bound workload repeating a 32-bit LCG update n_iter times.

    Args:
        n_iter: Number of iterations.

    Returns:
        Final integer value.
    """
    val = 123456789
    for _ in range(n_iter):
        val = (val * 1103515245 + 12345) & 0x7FFFFFFF
    return val


def calibrate_iters_per_ms() -> int:
    """Calibrate iterations of burn per millisecond over approximately 0.3 seconds.

    Returns:
        Estimated number of LCG iterations that execute in 1 ms on this system.
    """
    n_probe = 500_000
    t0 = time.perf_counter()
    burn(n_probe)
    dt = time.perf_counter() - t0

    # Scale target to ~0.3 seconds for robust calibration
    target_dt = 0.3
    if dt < 0.05:
        multiplier = max(1.0, target_dt / max(dt, 1e-6))
        n_probe = int(n_probe * multiplier)
        t0 = time.perf_counter()
        burn(n_probe)
        dt = time.perf_counter() - t0

    iters_per_ms = max(100, int(n_probe / (dt * 1000.0)))
    return iters_per_ms


def run_task(n_iter: int) -> int:
    """Top-level task function for multiprocessing pool execution."""
    return burn(n_iter)


def run_chunk(list_of_n_iter: list[int]) -> list[int]:
    """Execute a chunk of tasks sequentially within a worker process."""
    return [burn(it) for it in list_of_n_iter]


# ---------------------------------------------------------------------------
# Scheduling Simulators
# ---------------------------------------------------------------------------

def lower_bound(durations: Sequence[float], p: int) -> float:
    """Compute the theoretical makespan lower bound: max(sum(durations) / p, max(durations))."""
    if not durations or p <= 0:
        return 0.0
    return float(max(sum(durations) / p, max(durations)))


def simulate_static_block(durations: Sequence[float], p: int) -> float:
    """Simulate static block partitioning of tasks across p workers.

    Tasks are partitioned into contiguous blocks of size ceil(n / p).
    Worker i executes block i.

    Args:
        durations: List of task durations.
        p: Number of parallel workers.

    Returns:
        Makespan (maximum completion time across workers).
    """
    n = len(durations)
    if n == 0 or p <= 0:
        return 0.0
    block_size = int(math.ceil(n / p))
    worker_loads = [0.0] * p
    for w in range(p):
        start = w * block_size
        end = min(n, (w + 1) * block_size)
        if start < n:
            worker_loads[w] = float(sum(durations[start:end]))
    return float(max(worker_loads))


def simulate_static_round_robin(durations: Sequence[float], p: int) -> float:
    """Simulate static round-robin assignment of tasks across p workers.

    Task j is assigned to worker (j mod p).

    Args:
        durations: List of task durations.
        p: Number of parallel workers.

    Returns:
        Makespan.
    """
    if not durations or p <= 0:
        return 0.0
    worker_loads = [0.0] * p
    for j, d in enumerate(durations):
        worker_loads[j % p] += float(d)
    return float(max(worker_loads))


def simulate_list_scheduling(durations: Sequence[float], p: int) -> float:
    """Simulate online greedy list scheduling (Graham's algorithm).

    Tasks are considered in their given order; each task is assigned to the
    worker that becomes free earliest.

    Args:
        durations: List of task durations.
        p: Number of parallel workers.

    Returns:
        Makespan.
    """
    if not durations or p <= 0:
        return 0.0
    # Min-heap of (finish_time, worker_id)
    heap: list[tuple[float, int]] = [(0.0, i) for i in range(p)]
    heapq.heapify(heap)

    for d in durations:
        finish_time, worker_id = heapq.heappop(heap)
        heapq.heappush(heap, (finish_time + float(d), worker_id))

    return float(max(t for t, _ in heap))


def simulate_lpt(durations: Sequence[float], p: int) -> float:
    """Simulate Longest Processing Time (LPT) scheduling.

    Tasks are sorted in non-increasing order of duration, then scheduled via list scheduling.
    Guarantees makespan <= (4/3 - 1/(3p)) * OPT.

    Args:
        durations: List of task durations.
        p: Number of parallel workers.

    Returns:
        Makespan.
    """
    if not durations or p <= 0:
        return 0.0
    sorted_durations = sorted(durations, reverse=True)
    return simulate_list_scheduling(sorted_durations, p)


def simulate_work_stealing(
    durations: Sequence[float],
    p: int,
    steal_cost: float = 0.5,
    seed: int = 42,
) -> float:
    """Discrete-event simulation of dynamic work stealing.

    Tasks are initially dealt in contiguous blocks to p double-ended queues (deques).
    Each worker processes tasks from the front of its deque. When a worker's deque
    becomes empty, it becomes idle and steals half of the remaining tasks (at least one)
    from the back of the deque of the victim with the most remaining work, paying steal_cost.

    Args:
        durations: Task durations.
        p: Number of workers.
        steal_cost: Synchronization / transfer latency cost paid on each steal.
        seed: Random seed for tie-breaking.

    Returns:
        Makespan.
    """
    n = len(durations)
    if n == 0 or p <= 0:
        return 0.0

    block_size = int(math.ceil(n / p))
    deques: list[collections.deque[float]] = []
    for w in range(p):
        start = w * block_size
        end = min(n, (w + 1) * block_size)
        deques.append(collections.deque(float(durations[i]) for i in range(start, end)))

    clocks = [0.0] * p

    while True:
        # Check if all deques are empty
        if all(len(q) == 0 for q in deques):
            break

        # Select the worker with the earliest clock among those eligible
        # A worker is eligible if its queue is non-empty, OR if there is work to steal
        candidates = []
        for i in range(p):
            candidates.append((clocks[i], i))
        candidates.sort(key=lambda x: x[0])

        progressed = False
        for _, i in candidates:
            if len(deques[i]) > 0:
                task = deques[i].popleft()
                clocks[i] += task
                progressed = True
                break
            else:
                # Attempt to steal
                # Find victim with largest remaining work
                victim = -1
                max_rem = 0.0
                for v in range(p):
                    if v != i and len(deques[v]) > 0:
                        rem_work = sum(deques[v])
                        if rem_work > max_rem:
                            max_rem = rem_work
                            victim = v

                if victim != -1 and len(deques[victim]) > 0:
                    m = len(deques[victim])
                    num_to_steal = max(1, m // 2)
                    stolen = [deques[victim].pop() for _ in range(num_to_steal)]
                    deques[i].extend(reversed(stolen))
                    # Pay steal cost and synchronize with victim
                    clocks[i] = max(clocks[i], clocks[victim]) + float(steal_cost)
                    progressed = True
                    break

        if not progressed:
            break

    return float(max(clocks))


# ---------------------------------------------------------------------------
# Pipeline Model Parallelism
# ---------------------------------------------------------------------------

def pipeline_makespan(
    n_stages: int,
    n_micro: int,
    stage_times: Sequence[float],
) -> dict[str, float]:
    """Compute exact GPipe-style forward pipeline schedule makespan and bubble fraction.

    Recurrence:
        finish[m][s] = max(finish[m][s-1], finish[m-1][s]) + stage_times[s]
    where m is micro-batch index and s is stage index.

    Args:
        n_stages: Number of pipeline partition stages S.
        n_micro: Number of micro-batches M.
        stage_times: Sequence of execution times per stage (length S).

    Returns:
        Dictionary with 'makespan', 'bubble_fraction', and 'total_work'.
    """
    S = n_stages
    M = n_micro
    if len(stage_times) != S:
        raise ValueError(f"stage_times length {len(stage_times)} must match n_stages {S}")

    finish = np.zeros((M, S), dtype=float)

    for m in range(M):
        for s in range(S):
            left = finish[m, s - 1] if s > 0 else 0.0
            above = finish[m - 1, s] if m > 0 else 0.0
            finish[m, s] = max(left, above) + float(stage_times[s])

    makespan = float(finish[M - 1, S - 1])
    total_work = float(M * sum(stage_times))
    bubble_fraction = 1.0 - (total_work / (S * makespan))

    return {
        "makespan": makespan,
        "bubble_fraction": float(bubble_fraction),
        "total_work": total_work,
    }


# ---------------------------------------------------------------------------
# Data-Parallel Workers
# ---------------------------------------------------------------------------

def grad_shard_resident(args: tuple[int, int, int, int, np.ndarray, float]) -> np.ndarray:
    """Compute logistic loss gradient on a worker-resident data shard.

    The worker deterministically generates its shard in-place from (seed, shard_index)
    so no dataset tensors are shipped over IPC. Only model weights w are sent.

    Args:
        args: Tuple of (seed, shard_index, n_per_shard, d, w, l2_reg)

    Returns:
        Gradient vector of length d.
    """
    seed, shard_index, n_per_shard, d, w, l2_reg = args
    rng = np.random.default_rng(seed + shard_index * 1000)
    X = rng.normal(size=(n_per_shard, d))
    y = (rng.uniform(size=n_per_shard) > 0.5).astype(float)

    z = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    grad = (X.T @ (p - y)) / n_per_shard + l2_reg * w
    return grad


def grad_shard_shipped(args: tuple[np.ndarray, np.ndarray, np.ndarray, float]) -> np.ndarray:
    """Compute logistic loss gradient on a dataset shard shipped over IPC each invocation.

    Args:
        args: Tuple of (X, y, w, l2_reg)

    Returns:
        Gradient vector of length d.
    """
    X, y, w, l2_reg = args
    n_samples = len(y)

    z = X @ w
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))
    grad = (X.T @ (p - y)) / n_samples + l2_reg * w
    return grad

