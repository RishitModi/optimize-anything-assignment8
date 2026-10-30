"""Quantum annealing simulation and QUBO Chimera graph embedding analysis for Track I.

Features:
---------
1. Transverse-field quantum annealing simulation via second-order symmetric Trotter (Strang) splitting.
2. QUBO energy matrix and diagonal spectrum calculation.
3. QUBO benchmark instance loading from Track B.
4. Analytic Chimera clique embedding calculation (TRIAD literature model).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from qi_sim import apply_1q, rx


def load_instances(path: str | Path = "results/track_b/qubo_instances.json") -> list[dict[str, Any]]:
    """Load Max-Cut QUBO benchmark instances from Track B JSON artifact.

    Args:
        path: Path to qubo_instances.json.

    Returns:
        List of dictionaries with keys: n, Q (np.ndarray), opt_energy, opt_bitstrings.

    Raises:
        FileNotFoundError: If the benchmark instance JSON file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        # Also check relative to repo root if path was relative
        p_alt = _REPO_ROOT / path
        if p_alt.is_file():
            p = p_alt
        else:
            raise FileNotFoundError(
                f"Track B QUBO instances file not found at {path} or {p_alt}. "
                "Ensure Track B benchmarks were executed."
            )

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    instances: list[dict[str, Any]] = []
    for inst in data.get("instances", []):
        opt_e = inst.get("exact_optimal_energy")
        if opt_e is None:
            opt_e = inst.get("opt_energy", 0.0)
        instances.append({
            "n": int(inst["n"]),
            "Q": np.array(inst["Q"], dtype=np.float64),
            "opt_energy": float(opt_e),
            "opt_bitstrings": list(inst.get("optimal_bitstrings", [])),
        })

    return instances


def qubo_diag(Q: np.ndarray, n: int) -> np.ndarray:
    """Compute diagonal energies E(x) = x^T Q x for all 2^n computational basis states.

    Uses qubit-0-most-significant convention:
      Basis state index k in {0, ..., 2^n - 1} maps to bitstring x_0 x_1 ... x_{n-1}
      where x_i = (k >> (n - 1 - i)) & 1.

    Args:
        Q: n x n upper-triangular or symmetric QUBO matrix.
        n: Number of variables / qubits.

    Returns:
        1D float64 array of length 2^n containing energy of each basis state.
    """
    dim = 1 << n
    basis = np.arange(dim)
    # Extract bits: shape (2^n, n), column 0 is qubit 0 (MSB)
    bits = ((basis[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.float64)
    # x^T Q x for each row: sum_j (sum_k x_k Q_kj) x_j
    energies = np.sum((bits @ Q) * bits, axis=1)
    return energies


def optimal_indices(diag: np.ndarray) -> np.ndarray:
    """Return indices of all ground states within 1e-9 of the minimum energy."""
    min_e = np.min(diag)
    return np.where(np.abs(diag - min_e) <= 1e-9)[0]


def anneal_success(
    diag: np.ndarray,
    n: int,
    T: float,
    dt: float,
    opt_indices: list[int] | np.ndarray,
) -> tuple[float, float, float]:
    """Simulate quantum annealing using second-order symmetric Trotter (Strang) splitting.

    Time-dependent Hamiltonian:
      H(s) = -(1 - s) * sum_{i=0}^{n-1} X_i + s * H_P
    where:
      s = t / T in [0, 1] is the dimensionless annealing schedule,
      H_P is the problem diagonal scaled to n / (max(diag) - min(diag)) to standardize
          the energy scale across varying instance sizes (arbitrary simulation time units).

    Strang splitting per step dt at midpoint s = (t + dt/2)/T:
      1. Half-step problem Hamiltonian:
         state -> state * exp(-i (dt * s / 2) * H_P)
      2. Full-step driver Hamiltonian:
         For each qubit, apply RX(-2 * dt * (1 - s)), which equals exp(+i dt (1 - s) X).
      3. Half-step problem Hamiltonian:
         state -> state * exp(-i (dt * s / 2) * H_P)

    Initial state: uniform transverse-field ground state |+>^n.

    Args:
        diag: Unscaled problem diagonal energy spectrum (length 2^n).
        n: Number of qubits.
        T: Total annealing duration.
        dt: Time discretization step.
        opt_indices: Indices of ground states in the computational basis.

    Returns:
        tuple (success_prob, expected_energy, norm_error) where:
          - success_prob: Total probability of measuring any optimal ground state.
          - expected_energy: Expectation value of the unscaled problem Hamiltonian.
          - norm_error: Absolute deviation of total final state norm from 1.0.
    """
    dim = 1 << n
    span = float(np.max(diag) - np.min(diag))
    if span > 1e-12:
        H_P = diag * (float(n) / span)
    else:
        H_P = diag.copy()

    # Initial state: |+>^n = (1 / sqrt(2^n)) sum |x>
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(float(dim))

    n_steps = max(1, int(round(T / dt)))
    actual_dt = T / float(n_steps)

    for step in range(n_steps):
        s_mid = (step + 0.5) * actual_dt / T

        # Step 1: Half-step H_P
        phase_half = np.exp(-1.0j * (0.5 * actual_dt * s_mid) * H_P)
        state = state * phase_half

        # Step 2: Full-step Driver (Transverse Field)
        # exp(-i dt H_driver) = exp(+i dt (1 - s) sum_i X_i) = prod_i RX(-2 dt (1 - s))
        theta_rx = -2.0 * actual_dt * (1.0 - s_mid)
        u_rx = rx(theta_rx)
        for q in range(n):
            state = apply_1q(state, u_rx, q, n)

        # Step 3: Half-step H_P
        state = state * phase_half

    # Calculate final physical observables
    probs = np.real(state * np.conj(state))
    norm_val = float(np.sum(probs))
    norm_error = abs(norm_val - 1.0)

    # Normalize before computing probabilities to avoid numerical drift
    if norm_val > 0.0:
        probs = probs / norm_val

    success_prob = float(np.sum(probs[opt_indices]))
    expected_energy = float(np.sum(probs * diag))

    return success_prob, expected_energy, norm_error


def chimera_clique_embedding(n_logical: int) -> dict[str, Any]:
    """Compute standard TRIAD clique embedding resource requirements on a D-Wave Chimera graph.

    LITERATURE FORMULA:
      Standard TRIAD embedding of complete graph K_n into Chimera C_{L,L,4}:
      - M = ceil(n / 4)
      - Maximum chain length = M + 1
      - Physical qubits required = 4 * M * (M + 1)
      - A 16x16 Chimera graph (2048 physical qubits) can embed complete graphs up to n = 64 (M = 16).
      Note: This is an analytic literature formula and is not re-derived or executed on hardware.

    Args:
        n_logical: Number of fully-connected logical QUBO variables.

    Returns:
        Dictionary of physical embedding requirements and hardware feasibility flags.
    """
    M = int(math.ceil(n_logical / 4.0))
    chain_length = M + 1
    physical_qubits = 4 * M * (M + 1)
    fits_in_2048 = (M <= 16)
    chip_fraction = physical_qubits / 2048.0

    return {
        "n_logical": n_logical,
        "M": M,
        "chain_length": chain_length,
        "physical_qubits": physical_qubits,
        "fits_in_2048": fits_in_2048,
        "chip_fraction": chip_fraction,
        "note": "analytic (literature formula), not measured",
    }

