"""Variational Quantum Eigensolver (VQE) for quantum many-body systems."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

# Ensure repository root is on sys.path
_TRACK_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TRACK_ROOT.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from qsim import (
    Y_GATE,
    apply_1q,
    apply_cnot,
    apply_depolarizing_layer,
    dense_matrix_from_terms,
    expectation_pauli_hamiltonian,
    ry,
    zero_state,
)


def tfim_terms(n: int, g: float) -> list[tuple[float, str]]:
    """Generate Transverse-Field Ising Model (TFIM) terms on an open 1D chain.

    H = - sum_{i=0}^{n-2} Z_i Z_{i+1} - g * sum_{i=0}^{n-1} X_i
    """
    terms: list[tuple[float, str]] = []

    # Nearest-neighbor Ising interaction: -1.0 * Z_i Z_{i+1}
    for i in range(n - 1):
        pauli_chars = ["I"] * n
        pauli_chars[i] = "Z"
        pauli_chars[i + 1] = "Z"
        terms.append((-1.0, "".join(pauli_chars)))

    # Transverse magnetic field: -g * X_i
    for i in range(n):
        pauli_chars = ["I"] * n
        pauli_chars[i] = "X"
        terms.append((-float(g), "".join(pauli_chars)))

    return terms


def exact_ground_energy(terms: Sequence[tuple[float, str]], n: int) -> float:
    """Compute the exact ground-state energy via exact diagonalization of the Hamiltonian."""
    h_mat = dense_matrix_from_terms(terms, n)
    eigenvals = np.linalg.eigvalsh(h_mat)
    return float(eigenvals[0])


def ansatz_state(
    params: Sequence[float],
    n: int,
    L: int,
    noise_p: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate the real hardware-efficient ansatz state.

    The circuit consists of L layers, where each layer applies RY(theta) on every qubit
    followed by a linear CNOT entangling chain (0->1, 1->2, ..., n-2->n-1).
    A final layer of RY(theta) is appended on every qubit.
    Total parameter count is P = n * (L + 1).
    """
    psi = zero_state(n)

    for layer in range(L):
        # RY rotation on every qubit
        for q in range(n):
            theta = float(params[layer * n + q])
            psi = apply_1q(psi, ry(theta), q, n)

        if noise_p > 0.0 and rng is not None:
            psi = apply_depolarizing_layer(psi, noise_p, rng, n)

        # CNOT chain
        for c in range(n - 1):
            psi = apply_cnot(psi, control=c, target=c + 1, n)

        if noise_p > 0.0 and rng is not None:
            psi = apply_depolarizing_layer(psi, noise_p, rng, n)

    # Final RY layer
    for q in range(n):
        theta = float(params[L * n + q])
        psi = apply_1q(psi, ry(theta), q, n)

    if noise_p > 0.0 and rng is not None:
        psi = apply_depolarizing_layer(psi, noise_p, rng, n)

    return psi


def vqe_energy(
    params: Sequence[float],
    terms: Sequence[tuple[float, str]],
    n: int,
    L: int,
) -> float:
    """Compute exact expectation <psi(params)|H|psi(params)>."""
    psi = ansatz_state(params, n, L)
    return expectation_pauli_hamiltonian(psi, terms)


def param_shift_grad(
    params: Sequence[float],
    terms: Sequence[tuple[float, str]],
    n: int,
    L: int,
) -> np.ndarray:
    """Compute exact analytic gradient via the parameter-shift rule.

    For generator G = Y / 2:
        dE / dtheta_i = 0.5 * (E(theta + pi/2 * e_i) - E(theta - pi/2 * e_i))
    Cost: 2 * P circuit evaluations.
    """
    p_arr = np.array(params, dtype=np.float64)
    P = len(p_arr)
    grad = np.zeros(P, dtype=np.float64)

    shift = math.pi / 2.0
    for i in range(P):
        p_plus = p_arr.copy()
        p_plus[i] += shift
        e_plus = vqe_energy(p_plus, terms, n, L)

        p_minus = p_arr.copy()
        p_minus[i] -= shift
        e_minus = vqe_energy(p_minus, terms, n, L)

        grad[i] = 0.5 * (e_plus - e_minus)

    return grad


def build_derivative_states(
    params: Sequence[float],
    n: int,
    L: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the ansatz state psi and all parameter derivative states phi_i in P + 1 circuit runs.

    For parameter i = (target_layer * n + target_qubit), phi_i is obtained by applying Pauli Y
    to target_qubit immediately after its RY gate, before continuing with the rest of the circuit.
    """
    p_arr = np.array(params, dtype=np.float64)
    P = len(p_arr)

    # 1. Base ansatz state
    psi = ansatz_state(p_arr, n, L)

    phi_list: list[np.ndarray] = []

    # 2. Generate each phi_i
    for i in range(P):
        target_layer = i // n
        target_qubit = i % n

        state = zero_state(n)

        # Run circuit up to target_layer
        for layer in range(L):
            for q in range(n):
                theta = float(p_arr[layer * n + q])
                state = apply_1q(state, ry(theta), q, n)
                if layer == target_layer and q == target_qubit:
                    # Insert Pauli Y derivative operator
                    state = apply_1q(state, Y_GATE, q, n)

            # CNOT chain
            for c in range(n - 1):
                state = apply_cnot(state, control=c, target=c + 1, n)

        # Final RY layer
        for q in range(n):
            theta = float(p_arr[L * n + q])
            state = apply_1q(state, ry(theta), q, n)
            if target_layer == L and q == target_qubit:
                state = apply_1q(state, Y_GATE, q, n)

        phi_list.append(state)

    return psi, phi_list


def metric_tensor(params: Sequence[float], n: int, L: int) -> np.ndarray:
    """Compute the exact Fubini-Study metric tensor g_ij for Quantum Natural Gradient.

    Formula:
        g_ij = 0.25 * [ Re<phi_i | phi_j> - Re( <phi_i | psi> * <psi | phi_j> ) ]
    where |phi_i> is the unnormalized derivative state obtained by inserting Pauli Y
    after gate i.

    Note:
        This implementation computes derivative states directly, which requires
        statevector manipulation not directly measurable as a simple single observable
        on physical quantum hardware.
    """
    p_arr = np.array(params, dtype=np.float64)
    P = len(p_arr)

    psi, phi_list = build_derivative_states(p_arr, n, L)

    # Precompute overlaps <psi | phi_j>
    psi_phi = np.array([np.vdot(psi, phi) for phi in phi_list], dtype=np.complex128)

    g = np.zeros((P, P), dtype=np.float64)
    for i in range(P):
        for j in range(i, P):
            phi_i_phi_j = np.vdot(phi_list[i], phi_list[j])
            overlap_prod = np.conj(psi_phi[i]) * psi_phi[j]
            val = 0.25 * float(np.real(phi_i_phi_j - overlap_prod))
            g[i, j] = val
            g[j, i] = val

    return g

