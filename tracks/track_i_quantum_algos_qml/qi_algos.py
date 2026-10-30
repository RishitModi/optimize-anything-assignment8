"""Core textbook quantum algorithms implemented gate-by-gate on top of qi_sim.

Algorithms included:
--------------------
1. Deutsch-Jozsa algorithm (constant vs balanced oracle discrimination)
2. Bernstein-Vazirani algorithm (linear boolean function reconstruction)
3. Quantum Fourier Transform (QFT) and inverse QFT
4. Quantum Phase Estimation (QPE)
5. Shor's order-finding and integer factorization
6. Grover's unstructured database search
"""

from __future__ import annotations

import math
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from qi_sim import (
    H,
    X,
    apply_1q,
    apply_cnot,
    apply_cphase,
    apply_diag,
    apply_h_all,
    apply_swap,
    depolarizing_layer,
    probabilities,
    zero_state,
)


# =============================================================================
# 1. Deutsch-Jozsa Algorithm
# =============================================================================

def constant0(n: int) -> np.ndarray:
    """Return truth table f(x) = 0 for all x in {0, 1}^n."""
    return np.zeros(1 << n, dtype=np.int32)


def constant1(n: int) -> np.ndarray:
    """Return truth table f(x) = 1 for all x in {0, 1}^n."""
    return np.ones(1 << n, dtype=np.int32)


def balanced_random(n: int, seed: int = 42) -> np.ndarray:
    """Return a randomly chosen balanced truth table with exactly 2^(n-1) zeros and ones."""
    rng = np.random.default_rng(seed)
    half = 1 << (n - 1)
    arr = np.array([0] * half + [1] * half, dtype=np.int32)
    rng.shuffle(arr)
    return arr


def balanced_parity(n: int) -> np.ndarray:
    """Return a balanced truth table based on the bit parity: f(x) = sum_i x_i mod 2."""
    dim = 1 << n
    basis = np.arange(dim)
    # Extract bits across all positions
    bits = (basis[:, None] >> np.arange(n)) & 1
    return (np.sum(bits, axis=1) % 2).astype(np.int32)


def dj_probability_all_zero(
    n: int,
    f_table: np.ndarray,
    noise_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Simulate Deutsch-Jozsa circuit and return probability of measuring outcome |0...0>.

    Circuit structure:
      |0>^n -> H^n -> Phase Oracle (-1)^f(x) -> H^n -> Measure

    For a constant oracle, P(|0...0>) = 1.0.
    For a balanced oracle, P(|0...0>) = 0.0.

    Args:
        n: Number of query qubits.
        f_table: Truth table of length 2^n with integer values in {0, 1}.
        noise_p: Depolarizing error rate applied per layer.
        rng: Optional random generator for noise trajectories.

    Returns:
        Probability of observing the all-zeros bitstring |0...0>.
    """
    state = zero_state(n)
    state = apply_h_all(state, n)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    # Phase oracle: |x> -> (-1)^f(x) |x>
    diag = np.where(f_table == 1, -1.0, 1.0).astype(np.complex128)
    state = apply_diag(state, diag)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    state = apply_h_all(state, n)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    probs = probabilities(state)
    return float(probs[0])


# =============================================================================
# 2. Bernstein-Vazirani Algorithm
# =============================================================================

def bv_distribution(
    secret_bits: str | list[int] | np.ndarray,
    noise_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate Bernstein-Vazirani circuit to learn hidden bitstring s in {0, 1}^n.

    The oracle evaluates f(x) = s . x = sum_i s_i x_i mod 2.
    In the absence of noise, P(s) = 1.0 with a single quantum query.

    Args:
        secret_bits: Secret bitstring s (length n, qubit 0 is MSB).
        noise_p: Depolarizing error probability per layer.
        rng: Optional random generator for noise.

    Returns:
        Probability vector over all 2^n computational basis outcomes.
    """
    if isinstance(secret_bits, str):
        s = np.array([int(b) for b in secret_bits], dtype=np.int64)
    else:
        s = np.asarray(secret_bits, dtype=np.int64)
    n = len(s)
    dim = 1 << n

    basis = np.arange(dim)
    # Extract bits corresponding to each qubit (qubit 0 is MSB)
    bits = (basis[:, None] >> np.arange(n - 1, -1, -1)) & 1
    # Dot product s . x mod 2
    inner_prod = np.sum(bits * s[None, :], axis=1) % 2
    diag = np.where(inner_prod == 1, -1.0, 1.0).astype(np.complex128)

    state = zero_state(n)
    state = apply_h_all(state, n)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    state = apply_diag(state, diag)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    state = apply_h_all(state, n)
    if noise_p > 0.0 and rng is not None:
        state = depolarizing_layer(state, noise_p, rng, n)

    return probabilities(state)


# =============================================================================
# 3. Quantum Fourier Transform (QFT)
# =============================================================================

def qft(
    state: np.ndarray,
    qubits: list[int],
    n: int,
    inverse: bool = False,
) -> np.ndarray:
    """Apply Quantum Fourier Transform or inverse QFT on a subset of qubits.

    Converts computational basis states |j> into Fourier basis states:
      |j> -> (1 / sqrt(N)) sum_{k=0}^{N-1} exp(2 pi i j k / N) |k>
    Matrix action: F[k, j] = exp(2 pi i j k / N) / sqrt(N).
    Acting on statevector x, it computes:
      state -> np.fft.ifft(state) * sqrt(N).

    Textbook circuit:
      - For each qubit i: H, followed by controlled-phase rotations R_k = exp(i pi / 2^(j - i))
        controlled by qubit j.
      - Followed by bit-reversal SWAP gates.
      - Inverse QFT reverses the circuit order and negates rotation angles.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        qubits: Ordered list of qubit indices on which QFT acts (qubits[0] is MSB).
        n: Total number of qubits in the register.
        inverse: If True, applies inverse QFT.

    Returns:
        Updated statevector of same shape.
    """
    m = len(qubits)
    out = state

    if not inverse:
        # Forward QFT
        for i in range(m):
            qi = qubits[i]
            out = apply_1q(out, H, qi, n)
            for j in range(i + 1, m):
                qj = qubits[j]
                k = j - i
                angle = np.pi / (1 << k)
                out = apply_cphase(out, qj, qi, angle, n)
        # Bit-reversal SWAPs
        for i in range(m // 2):
            out = apply_swap(out, qubits[i], qubits[m - 1 - i], n)
    else:
        # Inverse QFT (mirrored order, negated angles)
        for i in range(m // 2):
            out = apply_swap(out, qubits[i], qubits[m - 1 - i], n)
        for i in range(m - 1, -1, -1):
            qi = qubits[i]
            for j in range(m - 1, i, -1):
                qj = qubits[j]
                k = j - i
                angle = -np.pi / (1 << k)
                out = apply_cphase(out, qj, qi, angle, n)
            out = apply_1q(out, H, qi, n)

    return out


# =============================================================================
# 4. Quantum Phase Estimation (QPE)
# =============================================================================

def qpe_distribution(phi: float, t: int) -> np.ndarray:
    """Simulate Quantum Phase Estimation for unitary U with eigenvalue exp(2 pi i phi).

    Uses t counting qubits (indices 0..t-1) and 1 eigenqubit (index t) prepared in |1>.
    Applies controlled-U^(2^(t-1-j)) operations from counting qubit j, followed by
    inverse QFT on the counting register.

    Args:
        phi: True phase in [0, 1).
        t: Number of counting qubits (register size).

    Returns:
        Probability distribution vector of length 2^t over counting basis outcomes.
    """
    tot_qubits = t + 1
    eigen_q = t

    # Initialize |0...0> and flip eigenqubit to |1>
    state = zero_state(tot_qubits)
    state = apply_1q(state, X, eigen_q, tot_qubits)

    # Put counting register into uniform superposition H^t
    for q in range(t):
        state = apply_1q(state, H, q, tot_qubits)

    # Controlled phase gates: controlled-U^(2^(t-1-j)) on eigenqubit
    # For counting qubit j, the power is 2^(t - 1 - j)
    for j in range(t):
        power = 1 << (t - 1 - j)
        angle = 2.0 * np.pi * phi * power
        state = apply_cphase(state, j, eigen_q, angle, tot_qubits)

    # Inverse QFT on counting register 0..t-1
    counting_qubits = list(range(t))
    state = qft(state, counting_qubits, tot_qubits, inverse=True)

    # Marginalize over the eigenqubit
    full_probs = probabilities(state)
    marginal = np.sum(full_probs.reshape((1 << t, 2)), axis=1)
    return marginal


# =============================================================================
# 5. Shor's Order Finding and Factorization
# =============================================================================

def true_order(a: int, N: int) -> int | None:
    """Find the exact multiplicative order of a modulo N by brute force."""
    if math.gcd(a, N) != 1:
        return None
    val = 1
    for r in range(1, N + 1):
        val = (val * a) % N
        if val == 1:
            return r
    return None


def shor_order_finding(N: int, a: int, t: int, w: int) -> tuple[np.ndarray, dict[str, int]]:
    """Simulate Shor's quantum order finding subroutine for a mod N.

    Registers:
      - Counting register: qubits 0..t-1 (size 2^t)
      - Work register: qubits t..t+w-1 (size 2^w), initialized to integer 1 (|0...01>)

    Controlled modular multiplications by a^(2^(t-1-j)) mod N are applied to the work register
    when counting qubit j is 1. Modular multiplication is implemented as a basis permutation:
      y -> (y * m) mod N  for y < N
      y -> y              for y >= N

    Args:
        N: Composite modulus to factor.
        a: Base coprime to N.
        t: Number of counting qubits.
        w: Number of work register qubits (2^w >= N).

    Returns:
        tuple (marginal probability vector over 2^t counting states, gate counts dict).
    """
    tot_qubits = t + w
    dim_c = 1 << t
    dim_w = 1 << w

    state = zero_state(tot_qubits)
    # Initialize work register to |1>: flip LSB of work register (qubit t + w - 1)
    state = apply_1q(state, X, tot_qubits - 1, tot_qubits)

    # H on counting qubits
    for q in range(t):
        state = apply_1q(state, H, q, tot_qubits)

    # Controlled modular multiplications
    # Reshape to 2D: (2^t counting outcomes, 2^w work states)
    state_2d = state.reshape((dim_c, dim_w))

    for j in range(t):
        power = 1 << (t - 1 - j)
        m = pow(a, power, N)

        # Build work-register permutation: perm_w[y] = (y * m) % N for y < N
        perm_w = np.arange(dim_w, dtype=np.int64)
        for y in range(N):
            perm_w[y] = (y * m) % N

        # Invert permutation so state[c, :] can be updated by gathering
        inv_perm_w = np.empty_like(perm_w)
        inv_perm_w[perm_w] = np.arange(dim_w)

        # Find counting states where qubit j is 1
        mask_c = ((np.arange(dim_c) >> (t - 1 - j)) & 1) == 1
        state_2d[mask_c] = state_2d[mask_c][:, inv_perm_w]

    state = state_2d.reshape(1 << tot_qubits)

    # Inverse QFT on counting register
    counting_qubits = list(range(t))
    state = qft(state, counting_qubits, tot_qubits, inverse=True)

    # Compute marginal probabilities of counting register
    full_probs = probabilities(state)
    marginal = np.sum(full_probs.reshape((dim_c, dim_w)), axis=1)

    gate_counts = {
        "controlled_mod_mul": t,
        "qft_hadamards": t,
        "qft_cphase": t * (t - 1) // 2,
        "qft_swaps": t // 2,
    }
    return marginal, gate_counts


def shor_postprocess(m: int, t: int, N: int, a: int) -> int | None:
    """Post-process a measured counting outcome m into a candidate order r using continued fractions.

    Args:
        m: Measured integer in {0, ..., 2^t - 1}.
        t: Number of counting qubits.
        N: Target modulus.
        a: Base coprime to N.

    Returns:
        Order r if verified (pow(a, r, N) == 1), else None.
    """
    if m == 0:
        return None
    frac = Fraction(m, 1 << t).limit_denominator(N)
    r0 = frac.denominator
    for mult in [1, 2, 3, 4]:
        cand_r = r0 * mult
        if 0 < cand_r < N and pow(a, cand_r, N) == 1:
            return cand_r
    return None


def shor_factors(a: int, r: int, N: int) -> set[int] | None:
    """Extract non-trivial factors of N given an even order r.

    Args:
        a: Base.
        r: Even order of a modulo N.
        N: Composite modulus.

    Returns:
        Set of non-trivial factors {f1, f2} if successful, else None.
    """
    if r % 2 != 0 or r <= 0:
        return None
    half_exp = pow(a, r // 2, N)
    if half_exp == N - 1 or half_exp == 1:
        return None

    f1 = math.gcd(half_exp - 1, N)
    f2 = math.gcd(half_exp + 1, N)
    factors = set()
    if 1 < f1 < N:
        factors.add(f1)
    if 1 < f2 < N:
        factors.add(f2)
    return factors if factors else None


# =============================================================================
# 6. Grover's Unstructured Search
# =============================================================================

def grover_optimal_iterations(N: int, M: int = 1) -> int:
    """Theoretical optimal number of Grover iterations floor(pi/4 * sqrt(N / M))."""
    return int(math.floor(np.pi / 4.0 * np.sqrt(float(N) / float(M))))


def grover_theory(N: int, M: int, k: int) -> float:
    """Exact theoretical Grover success probability after k iterations: sin^2((2k+1) theta)."""
    theta = np.arcsin(np.sqrt(float(M) / float(N)))
    return float(np.sin((2 * k + 1) * theta) ** 2)


def grover_gate_counts(n: int, k: int) -> dict[str, int]:
    """Compute physical gate counts for n-qubit Grover search after k iterations.

    Note: Counts each n-qubit multi-controlled-Z as ONE gate.

    Args:
        n: Number of qubits.
        k: Number of Grover iterations.

    Returns:
        Dictionary of gate counts.
    """
    return {
        "oracle_calls": k,
        "hadamards": n + 2 * n * k,
        "x_gates": 2 * n * k,
        "multi_controlled_z": 2 * k,
    }


def grover_run(
    n: int,
    marked: list[int] | set[int],
    iterations: int,
    noise_p: float = 0.0,
    rng: np.random.Generator | None = None,
    record_curve: bool = False,
) -> tuple[float, int, list[float]]:
    """Simulate Grover unstructured search algorithm on n qubits.

    Uniform superposition is prepared with H^n.
    Each iteration applies:
      1. Oracle: Phase inversion diag (-1 on marked states, +1 elsewhere).
      2. Diffusion operator: Inversion about the mean (2 * mean - state).
      3. Optional depolarizing layer.

    Args:
        n: Number of qubits.
        marked: Collection of target state indices in {0, ..., 2^n - 1}.
        iterations: Number of Grover iterations to execute.
        noise_p: Depolarizing error rate per iteration.
        rng: Seeded random generator for noise trajectories.
        record_curve: If True, tracks success probability at each iteration.

    Returns:
        tuple (final success probability, total oracle queries, curve of probabilities).
    """
    dim = 1 << n
    marked_list = list(marked)

    # Oracle diagonal: -1 on marked, +1 elsewhere
    oracle_diag = np.ones(dim, dtype=np.complex128)
    oracle_diag[marked_list] = -1.0

    state = zero_state(n)
    state = apply_h_all(state, n)

    curve: list[float] = []
    if record_curve:
        p_init = float(np.sum(probabilities(state)[marked_list]))
        curve.append(p_init)

    for it in range(iterations):
        # 1. Oracle
        state = state * oracle_diag

        # 2. Diffusion: 2 * mean - state
        mean_val = np.mean(state)
        state = 2.0 * mean_val - state

        # 3. Noise channel
        if noise_p > 0.0 and rng is not None:
            state = depolarizing_layer(state, noise_p, rng, n)

        if record_curve:
            p_cur = float(np.sum(probabilities(state)[marked_list]))
            curve.append(p_cur)

    success_prob = float(np.sum(probabilities(state)[marked_list]))
    return success_prob, iterations, curve


def linear_search_queries(N: int, target: int, order_seed: int) -> int:
    """Number of queries required by classical random linear scan to find target.

    Args:
        N: Search space size.
        target: Target item in {0, ..., N-1}.
        order_seed: Seed defining permutation order.

    Returns:
        1-based count of queries until target is inspected.
    """
    rng = np.random.default_rng(order_seed)
    perm = rng.permutation(N)
    loc = np.where(perm == target)[0][0]
    return int(loc + 1)

