"""Quantum Approximate Optimization Algorithm (QAOA) for combinatorial optimization."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ensure repository root is on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import scipy.optimize

from qsim import apply_1q, apply_depolarizing_layer, expectation_diag, plus_state, rx


def load_instances(path: str | Path = "results/track_b/qubo_instances.json") -> list[dict[str, Any]]:
    """Load precomputed QUBO/Ising benchmark instances from JSON.

    Returns a list of parsed dictionaries containing:
        - 'n': number of qubits/vertices
        - 'Q': upper-triangular QUBO matrix as np.ndarray
        - 'h': Ising linear terms as np.ndarray
        - 'J': Ising couplings as dict with (int, int) tuple keys
        - 'offset': scalar energy shift
        - 'opt_energy': exact minimum energy
        - 'opt_cut': exact maximum cut value
        - 'opt_bitstrings': list of all optimal bitstrings (x_0 first)
    """
    p = Path(path)
    if not p.is_absolute():
        p = _REPO_ROOT / p

    if not p.exists():
        raise FileNotFoundError(
            f"Required QUBO instances file not found at: {p}. "
            "Please ensure Track B's run_all.py has been executed to generate results/track_b/qubo_instances.json."
        )

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    raw_instances = data["instances"] if isinstance(data, dict) and "instances" in data else data
    instances: list[dict[str, Any]] = []

    for item in raw_instances:
        # Convert "i,j" string keys to (i, j) tuple keys
        j_dict: dict[tuple[int, int], float] = {}
        for k_str, val in item["J"].items():
            parts = k_str.split(",")
            j_dict[(int(parts[0]), int(parts[1]))] = float(val)

        instances.append({
            "n": int(item["n"]),
            "Q": np.array(item["Q"], dtype=np.float64),
            "h": np.array(item["h"], dtype=np.float64),
            "J": j_dict,
            "offset": float(item["offset"]),
            "opt_energy": float(item["exact_optimal_energy"]),
            "opt_cut": int(item["optimal_cut_value"]),
            "opt_bitstrings": list(item["optimal_bitstrings"]),
        })

    return instances


def qubo_diag(Q: np.ndarray) -> np.ndarray:
    """Compute diagonal energies x^T Q x for all 2^n basis states with MSB-first bit order."""
    n = Q.shape[0]
    total_states = 1 << n
    indices = np.arange(total_states, dtype=np.int64)

    # Qubit k is at bit position (n - 1 - k)
    bits = ((indices[:, None] >> np.arange(n - 1, -1, -1, dtype=np.int64)) & 1).astype(np.float64)
    # Energy x^T Q x for each row
    diag = np.sum((bits @ Q) * bits, axis=1)
    return diag


def ising_diag(
    h: np.ndarray,
    J: Mapping[tuple[int, int], float],
    offset: float,
    n: int,
) -> np.ndarray:
    """Compute diagonal Ising energies E(s) for all 2^n basis states with s = 1 - 2x."""
    total_states = 1 << n
    indices = np.arange(total_states, dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n - 1, -1, -1, dtype=np.int64)) & 1).astype(np.float64)
    spins = 1.0 - 2.0 * bits

    diag = np.full(total_states, float(offset), dtype=np.float64)
    diag += spins @ h
    for (i, j), coupling in J.items():
        diag += coupling * spins[:, i] * spins[:, j]

    return diag


def qaoa_state(
    params: Sequence[float],
    diag: np.ndarray,
    n: int,
    p: int,
    noise_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Construct QAOA statevector for p layers with optional depolarizing noise."""
    gamma = params[:p]
    beta = params[p:]

    state = plus_state(n)

    for layer in range(p):
        g = float(gamma[layer])
        b = float(beta[layer])

        # 1. Cost unitary U_C(gamma) = exp(-i * gamma * diag)
        state = state * np.exp(-1.0j * g * diag)

        if noise_p > 0.0 and rng is not None:
            state = apply_depolarizing_layer(state, noise_p, rng, n)

        # 2. Mixer unitary U_M(beta) = prod_q RX(2 * beta) = prod_q exp(-i * beta * X_q)
        rx_mat = rx(2.0 * b)
        for q in range(n):
            state = apply_1q(state, rx_mat, q, n)

        if noise_p > 0.0 and rng is not None:
            state = apply_depolarizing_layer(state, noise_p, rng, n)

    return state


def qaoa_expectation(
    params: Sequence[float],
    diag: np.ndarray,
    n: int,
    p: int,
) -> float:
    """Compute exact expectation value <psi|H_C|psi> for noise-free QAOA."""
    state = qaoa_state(params, diag, n, p)
    return expectation_diag(state, diag)


def qaoa_expectation_shots(
    params: Sequence[float],
    diag: np.ndarray,
    n: int,
    p: int,
    shots: int,
    rng: np.random.Generator,
) -> float:
    """Estimate QAOA energy expectation by sampling basis state measurement shots."""
    state = qaoa_state(params, diag, n, p)
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)
    samples = rng.choice(len(diag), size=shots, p=probs)
    return float(np.mean(diag[samples]))


def approximation_ratio(expected_energy: float, opt_energy: float) -> float:
    """Compute QAOA approximation ratio = expected_energy / opt_energy.

    Since both are formulated as negative cut values (to be minimized),
    the ratio of negative energies equals expected_cut / optimal_cut.
    """
    if abs(opt_energy) < 1e-12:
        return 1.0
    return float(expected_energy / opt_energy)


def prob_optimal(state: np.ndarray, opt_bitstrings: Sequence[str], n: int) -> float:
    """Compute the probability of sampling any of the optimal bitstrings."""
    probs = np.abs(state) ** 2
    total_prob = 0.0
    for b_str in opt_bitstrings:
        # Convert bitstring (x_0 first) to basis index
        idx = int(b_str, 2)
        total_prob += float(probs[idx])
    return total_prob


def run_qaoa(
    instance: dict[str, Any],
    p: int,
    n_inits: int,
    seed: int,
    maxiter: int,
) -> dict[str, Any]:
    """Execute multi-start QAOA optimization using COBYLA."""
    n = instance["n"]
    diag = qubo_diag(instance["Q"])
    opt_energy = instance["opt_energy"]
    opt_bitstrings = instance["opt_bitstrings"]

    rng = np.random.default_rng(seed)

    best_energy = float("inf")
    best_params: np.ndarray | None = None
    runs: list[dict[str, Any]] = []

    for _ in range(n_inits):
        # Sample initial parameters: gamma in [0, pi], beta in [0, pi/2]
        gamma_init = rng.uniform(0.0, np.pi, size=p)
        beta_init = rng.uniform(0.0, np.pi / 2.0, size=p)
        params0 = np.concatenate([gamma_init, beta_init])

        eval_count = 0

        def objective(th: np.ndarray) -> float:
            nonlocal eval_count
            eval_count += 1
            return qaoa_expectation(th, diag, n, p)

        res = scipy.optimize.minimize(
            objective,
            params0,
            method="COBYLA",
            options={"maxiter": maxiter},
        )

        final_energy = float(res.fun)
        opt_params = res.x
        final_state = qaoa_state(opt_params, diag, n, p)

        ratio = approximation_ratio(final_energy, opt_energy)
        p_opt = prob_optimal(final_state, opt_bitstrings, n)

        runs.append({
            "energy": final_energy,
            "ratio": ratio,
            "prob_opt": p_opt,
            "evals": eval_count,
            "params": opt_params,
        })

        if final_energy < best_energy:
            best_energy = final_energy
            best_params = opt_params

    return {
        "best_energy": best_energy,
        "best_params": best_params,
        "best_ratio": approximation_ratio(best_energy, opt_energy),
        "runs": runs,
    }

