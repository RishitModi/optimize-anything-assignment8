"""QUBO and Ising formulations, energy functions, conversions, and discrete solvers.

Conventions:
    QUBO energy E(x) = x^T Q x with Q upper-triangular (diagonal = linear terms),
    x in {0,1}^n, to be MINIMISED.

    Ising energy E(s) = sum_{i<j} J[i,j] s_i s_j + sum_i h[i] s_i + offset with
    s in {-1,+1}^n and the substitution x_i = (1 - s_i)/2 (so s_i = +1 corresponds
    to x_i = 0, matching the qubit convention where Pauli-Z has eigenvalue +1 on |0>).

Derivation of QUBO to Ising mapping:
    Substitute x_i = (1 - s_i)/2 into E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j:
    1. Linear terms:
       Q_ii * x_i = Q_ii * (1 - s_i) / 2 = Q_ii / 2 - (Q_ii / 2) * s_i.
    2. Quadratic terms (i < j):
       Q_ij * x_i * x_j = Q_ij * ((1 - s_i)/2) * ((1 - s_j)/2)
                        = (Q_ij / 4) * (1 - s_i - s_j + s_i * s_j)
                        = Q_ij / 4 - (Q_ij / 4) * s_i - (Q_ij / 4) * s_j + (Q_ij / 4) * s_i * s_j.
    Collecting coefficients:
       J[i, j] = Q_ij / 4 (for i < j)
       h[i] = -Q_ii / 2 - (1 / 4) * sum_{j != i} Q_{min(i,j), max(i,j)}
       offset = sum_i (Q_ii / 2) + sum_{i<j} (Q_ij / 4).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from common.timing import Timer


def qubo_energy(Q: np.ndarray, x: np.ndarray | Sequence[int]) -> float:
    """Compute QUBO energy E(x) = x^T Q x for upper-triangular Q and binary vector x.

    Args:
        Q: 2D array of shape (n, n), upper-triangular matrix of quadratic and linear terms.
        x: 1D array of shape (n,) with binary entries in {0, 1}.

    Returns:
        Scalar QUBO energy value.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    # Energy evaluated as standard bilinear form x^T Q x
    return float(x_arr @ Q @ x_arr)


def ising_energy(
    h: np.ndarray | Sequence[float],
    J: Mapping[tuple[int, int] | str, float] | np.ndarray,
    offset: float,
    s: np.ndarray | Sequence[int],
) -> float:
    """Compute Ising energy E(s) = sum_{i<j} J[i,j] s_i s_j + sum_i h[i] s_i + offset.

    Args:
        h: 1D array of linear magnetic field terms, shape (n,).
        J: Coupling terms as a mapping with keys (i, j) or "i,j" (where i < j),
           or 2D array of couplings.
        offset: Energy constant term independent of spin state s.
        s: 1D array of spin variables with entries in {-1, +1}, shape (n,).

    Returns:
        Scalar Ising energy value.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    h_arr = np.asarray(h, dtype=np.float64)
    n = len(s_arr)

    # Base energy from constant offset and linear fields
    energy = float(offset + np.sum(h_arr * s_arr))

    # Add pairwise spin-spin interaction energy
    if isinstance(J, Mapping):
        for key, val in J.items():
            if isinstance(key, str):
                parts = key.split(",")
                i, j = int(parts[0]), int(parts[1])
            else:
                i, j = key
            energy += float(val) * s_arr[i] * s_arr[j]
    else:
        # 2D matrix form
        for i in range(n):
            for j in range(i + 1, n):
                energy += float(J[i, j]) * s_arr[i] * s_arr[j]

    return float(energy)


def qubo_to_ising(
    Q: np.ndarray,
) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    """Convert an upper-triangular QUBO matrix Q to exact Ising parameters (h, J, offset).

    Derived rigorously using substitution x_i = (1 - s_i) / 2:
        s_i = +1 <=> x_i = 0
        s_i = -1 <=> x_i = 1

    Args:
        Q: 2D array of shape (n, n), upper-triangular QUBO matrix.

    Returns:
        h: 1D array of shape (n,) containing linear spin coefficients.
        J: Dictionary mapping (i, j) with i < j to coupling coefficient J_ij.
        offset: Scalar energy shift.
    """
    n = Q.shape[0]
    # Symmetrize off-diagonals into strictly upper-triangular format if needed
    Q_ut = np.triu(Q, 1) + np.tril(Q, -1).T
    np.fill_diagonal(Q_ut, np.diag(Q))

    # Constant offset: sum_i Q_ii/2 + sum_{i<j} Q_ij/4
    offset = float(np.sum(np.diag(Q_ut)) / 2.0)
    # Linear magnetic fields: h_i = -Q_ii/2 - 1/4 * sum_{j!=i} Q_ij
    h = -np.diag(Q_ut) / 2.0
    J: dict[tuple[int, int], float] = {}

    for i in range(n):
        for j in range(i + 1, n):
            q_ij = float(Q_ut[i, j])
            coupling = q_ij / 4.0
            J[(i, j)] = coupling
            offset += coupling
            h[i] -= coupling
            h[j] -= coupling

    return h.astype(np.float64), J, float(offset)


def maxcut_qubo(graph: nx.Graph) -> np.ndarray:
    """Formulate Max-Cut as an upper-triangular QUBO matrix to be MINIMISED.

    Cut size across a binary partition x in {0, 1}^n is:
        Cut(x) = sum_{(i,j) in E, i<j} (x_i + x_j - 2 x_i x_j)
    To formulate as a minimization problem, we set E(x) = -Cut(x):
        -Cut(x) = sum_{(i,j) in E, i<j} (-x_i - x_j + 2 x_i x_j)
    Summing -x_i over all incidents edges gives -degree(i) * x_i on the diagonal.
    The off-diagonal entries for edges (i, j) with i < j are +2.

    Args:
        graph: NetworkX undirected graph with nodes indexed 0, ..., n-1.

    Returns:
        Q: 2D array of shape (n, n), upper-triangular QUBO matrix.
    """
    n = graph.number_of_nodes()
    Q = np.zeros((n, n), dtype=np.float64)

    # Diagonal: -degree(i) for each node
    for i in range(n):
        Q[i, i] = -float(graph.degree[i])

    # Off-diagonal: +2 for each undirected edge (i, j) with i < j
    for u, v in graph.edges():
        i, j = min(int(u), int(v)), max(int(u), int(v))
        Q[i, j] += 2.0

    return Q


def cut_value(graph: nx.Graph, x: np.ndarray | Sequence[int]) -> int:
    """Direct count of cut edges in graph under binary partition x.

    An edge (u, v) is cut if x[u] != x[v].

    Args:
        graph: NetworkX undirected graph.
        x: Binary assignment vector of shape (n,).

    Returns:
        Integer count of edges crossing the cut.
    """
    x_arr = np.asarray(x, dtype=int)
    cut = 0
    for u, v in graph.edges():
        if x_arr[int(u)] != x_arr[int(v)]:
            cut += 1
    return cut


def feature_selection_qubo(
    relevance: np.ndarray | Sequence[float],
    redundancy: np.ndarray,
    k: int,
    alpha: float,
    lam: float,
) -> tuple[np.ndarray, float]:
    """Formulate feature selection with cardinality constraint penalty as a QUBO.

    Objective to minimise:
        E(x) = -sum_i r_i x_i + alpha * sum_{i<j} R_ij x_i x_j + lam * (sum_i x_i - k)^2
    Expanding the penalty term:
        lam * (sum_i x_i - k)^2 = lam * [ (sum_i x_i)^2 - 2 k sum_i x_i + k^2 ]
                                = lam * [ sum_i x_i + 2 sum_{i<j} x_i x_j - 2 k sum_i x_i + k^2 ]
                                = sum_i lam * (1 - 2 k) x_i + sum_{i<j} 2 lam x_i x_j + lam * k^2

    Combining linear and quadratic terms:
        Q_ii = -r_i + lam * (1 - 2 k)
        Q_ij = alpha * R_ij + 2 * lam (for i < j)
        offset = lam * k^2

    Args:
        relevance: 1D array of shape (n,) containing feature relevance scores r_i.
        redundancy: 2D array of shape (n, n) containing pairwise redundancy scores R_ij.
        k: Desired target cardinality (number of features to select).
        alpha: Weight balancing redundancy penalty vs relevance.
        lam: Multiplier for the quadratic cardinality penalty.

    Returns:
        Q: 2D array of shape (n, n), upper-triangular QUBO matrix.
        offset: Energy constant shift equal to lam * k^2.
    """
    rel = np.asarray(relevance, dtype=np.float64)
    red = np.asarray(redundancy, dtype=np.float64)
    n = len(rel)

    Q = np.zeros((n, n), dtype=np.float64)

    # Linear terms: diagonal entries
    for i in range(n):
        Q[i, i] = -rel[i] + float(lam) * (1.0 - 2.0 * float(k))

    # Quadratic interaction terms: upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = float(alpha) * red[i, j] + 2.0 * float(lam)

    offset = float(lam) * (float(k) ** 2)
    return Q, offset


def brute_force_qubo(
    Q: np.ndarray,
    offset: float = 0.0,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    """Exact QUBO solver via vectorized chunked exhaustive search over all 2^n assignments.

    Args:
        Q: Upper-triangular QUBO matrix of shape (n, n).
        offset: Scalar constant energy offset. Defaults to 0.0.
        chunk_size: Maximum number of candidate bitstrings evaluated concurrently in NumPy.

    Returns:
        dict containing:
            - 'best_energy': float minimum energy achieved
            - 'best_bitstrings': list of bitstrings achieving minimum energy with x_0 first
            - 'best_x': 1D array of shape (n,) with binary entries for the optimal state
            - 'runtime_s': elapsed execution time in seconds
    """
    n = Q.shape[0]
    if n > 22:
        raise ValueError(f"brute_force_qubo only supported for n <= 22, got n={n}")

    total_states = 1 << n
    min_energy = float("inf")
    best_bitstrings: list[str] = []
    best_x: np.ndarray | None = None

    with Timer() as timer:
        # Process in chunks to maintain low memory usage
        for start_idx in range(0, total_states, chunk_size):
            end_idx = min(start_idx + chunk_size, total_states)
            indices = np.arange(start_idx, end_idx, dtype=np.int64)

            # Generate binary bit assignments for chunk: bit j corresponds to x_j
            bits = ((indices[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)

            # Vectorized energy computation: diag(bits @ Q @ bits^T) + offset
            energies = np.sum((bits @ Q) * bits, axis=1) + float(offset)
            chunk_min = float(np.min(energies))

            tol = 1e-9
            if chunk_min < min_energy - tol:
                min_energy = chunk_min
                best_bitstrings.clear()
                best_x = None

            if abs(chunk_min - min_energy) <= tol or chunk_min <= min_energy:
                matching = np.where(np.abs(energies - min_energy) <= tol)[0]
                for idx_in_chunk in matching:
                    b_row = bits[idx_in_chunk].astype(int)
                    # String representation with x_0 first
                    bitstr = "".join(str(b) for b in b_row)
                    best_bitstrings.append(bitstr)
                    if best_x is None:
                        best_x = b_row

    return {
        "best_energy": min_energy,
        "best_bitstrings": best_bitstrings,
        "best_x": best_x if best_x is not None else np.zeros(n, dtype=int),
        "runtime_s": timer.elapsed_sec,
    }


def greedy_flip_local_search(
    Q: np.ndarray,
    seed: int,
    restarts: int = 20,
) -> dict[str, Any]:
    """Greedy single-bit-flip local search with random restarts to minimise QUBO energy.

    From a random initial bitstring, repeatedly flips the single bit that achieves the
    maximum energy reduction until reaching a 1-flip local minimum.

    Args:
        Q: Upper-triangular QUBO matrix of shape (n, n).
        seed: Random seed for initialization.
        restarts: Number of independent random restarts.

    Returns:
        dict containing 'best_energy', 'best_x', and 'runtime_s'.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    best_overall_energy = float("inf")
    best_overall_x = np.zeros(n, dtype=int)

    with Timer() as timer:
        for _ in range(restarts):
            # Random starting point
            x = rng.integers(0, 2, size=n, dtype=int)
            cur_energy = qubo_energy(Q, x)

            while True:
                best_delta = 0.0
                best_flip = -1

                # Evaluate energy delta for flipping each bit i
                for i in range(n):
                    delta_xi = 1 - 2 * x[i]
                    # dE = delta_xi * (Q_ii + sum_{j<i} Q_ji x_j + sum_{j>i} Q_ij x_j)
                    dE = delta_xi * (
                        Q[i, i]
                        + np.sum(Q[:i, i] * x[:i])
                        + np.sum(Q[i, i + 1:] * x[i + 1:])
                    )
                    if dE < best_delta - 1e-9:
                        best_delta = dE
                        best_flip = i

                if best_flip != -1:
                    x[best_flip] = 1 - x[best_flip]
                    cur_energy += best_delta
                else:
                    # No 1-bit flip improves energy; local optimum reached
                    break

            if cur_energy < best_overall_energy - 1e-9:
                best_overall_energy = cur_energy
                best_overall_x = x.copy()

    return {
        "best_energy": best_overall_energy,
        "best_x": best_overall_x,
        "runtime_s": timer.elapsed_sec,
    }


def simulated_annealing_qubo(
    Q: np.ndarray,
    seed: int,
    n_sweeps: int = 2000,
    t0: float = 2.0,
    t1: float = 0.01,
) -> dict[str, Any]:
    """Single-bit-flip Metropolis simulated annealing on QUBO with geometric cooling.

    Args:
        Q: Upper-triangular QUBO matrix of shape (n, n).
        seed: Random seed for initialization and Metropolis decisions.
        n_sweeps: Number of Monte Carlo sweeps (each sweep proposes n bit flips).
        t0: Initial temperature.
        t1: Final temperature.

    Returns:
        dict containing 'best_energy', 'best_x', and 'runtime_s'.
    """
    rng = np.random.default_rng(seed)
    n = Q.shape[0]

    with Timer() as timer:
        x = rng.integers(0, 2, size=n, dtype=int)
        cur_energy = qubo_energy(Q, x)
        best_energy = cur_energy
        best_x = x.copy()

        # Geometric cooling factor
        if n_sweeps <= 1:
            decay = 1.0
        else:
            decay = (t1 / t0) ** (1.0 / (n_sweeps - 1))

        temperature = float(t0)

        for _ in range(n_sweeps):
            # Propose flipping each bit in a random order
            order = rng.permutation(n)
            for i in order:
                delta_xi = 1 - 2 * x[i]
                dE = delta_xi * (
                    Q[i, i]
                    + np.sum(Q[:i, i] * x[:i])
                    + np.sum(Q[i, i + 1:] * x[i + 1:])
                )

                # Metropolis-Hastings acceptance condition
                if dE < 0.0 or rng.random() < np.exp(-dE / max(temperature, 1e-12)):
                    x[i] = 1 - x[i]
                    cur_energy += dE
                    if cur_energy < best_energy - 1e-9:
                        best_energy = cur_energy
                        best_x = x.copy()

            temperature *= decay

    return {
        "best_energy": best_energy,
        "best_x": best_x,
        "runtime_s": timer.elapsed_sec,
    }

