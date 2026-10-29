"""Statevector quantum circuit simulator using NumPy.

Convention:
    An n-qubit quantum state is represented as a complex128 vector of length 2^n.
    Basis index i has binary representation x_0 x_1 ... x_{n-1} where qubit 0 is the
    MOST significant bit (MSB), i.e., x_k = (i >> (n - 1 - k)) & 1.
    Bit 0 corresponds to state |0> and Pauli-Z eigenvalue +1, so the corresponding
    Ising spin variable is s_k = 1 - 2 * x_k.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Standard single-qubit Pauli and Clifford gates
I_GATE = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
X_GATE = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y_GATE = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z_GATE = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
H_GATE = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)

PAULI_MAP = {
    "I": I_GATE,
    "X": X_GATE,
    "Y": Y_GATE,
    "Z": Z_GATE,
}


def zero_state(n: int) -> np.ndarray:
    """Return the |0...0> computational basis ground state of n qubits."""
    state = np.zeros(1 << n, dtype=np.complex128)
    state[0] = 1.0 + 0.0j
    return state


def plus_state(n: int) -> np.ndarray:
    """Return the uniform superposition |+...+> = H^n |0...0> of n qubits."""
    dim = 1 << n
    norm = 1.0 / math.sqrt(dim)
    return np.full(dim, norm, dtype=np.complex128)


def rx(theta: float) -> np.ndarray:
    """Return single-qubit rotation matrix RX(theta) = exp(-i * theta * X / 2)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -1.0j * s], [-1.0j * s, c]], dtype=np.complex128)


def ry(theta: float) -> np.ndarray:
    """Return single-qubit rotation matrix RY(theta) = exp(-i * theta * Y / 2)."""
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta: float) -> np.ndarray:
    """Return single-qubit rotation matrix RZ(theta) = exp(-i * theta * Z / 2)."""
    phase_neg = np.exp(-0.5j * theta)
    phase_pos = np.exp(0.5j * theta)
    return np.array([[phase_neg, 0.0], [0.0, phase_pos]], dtype=np.complex128)


def apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply a 2x2 unitary matrix U to qubit q of an n-qubit statevector.

    Reshapes state to (2^q, 2, 2^(n-q-1)) matching MSB-first ordering and contracts
    the target dimension with U.
    """
    dim_high = 1 << q
    dim_low = 1 << (n - q - 1)
    reshaped = state.reshape((dim_high, 2, dim_low))
    contracted = np.einsum("ij, ajb -> aib", U, reshaped)
    return contracted.reshape(-1)


def apply_cnot(state: np.ndarray, control: int, target: int, n: int) -> np.ndarray:
    """Apply a CNOT gate with given control and target qubits.

    Reshapes the state to (2,)*n, selects the control=1 slice, and swaps target=0
    and target=1 along the target axis.
    """
    psi = state.reshape((2,) * n).copy()

    idx_c1_t0 = [slice(None)] * n
    idx_c1_t0[control] = 1
    idx_c1_t0[target] = 0

    idx_c1_t1 = [slice(None)] * n
    idx_c1_t1[control] = 1
    idx_c1_t1[target] = 1

    t0 = tuple(idx_c1_t0)
    t1 = tuple(idx_c1_t1)

    tmp = psi[t0].copy()
    psi[t0] = psi[t1]
    psi[t1] = tmp

    return psi.reshape(-1)


def apply_pauli_string(state: np.ndarray, pauli_string: str) -> np.ndarray:
    """Apply an n-qubit Pauli string operator (e.g. 'XZIY') to state."""
    n = len(pauli_string)
    psi = state.copy()
    for q, char in enumerate(pauli_string):
        if char != "I":
            psi = apply_1q(psi, PAULI_MAP[char], q, n)
    return psi


def expectation_diag(state: np.ndarray, diag: np.ndarray) -> float:
    """Compute the expectation value <psi|diag|psi> = sum_i |psi_i|^2 * diag_i."""
    probs = np.abs(state) ** 2
    return float(np.sum(probs * diag))


def expectation_pauli_hamiltonian(
    state: np.ndarray,
    terms: Sequence[tuple[float, str]],
) -> float:
    """Compute exact expectation value <psi|H|psi> for a Pauli sum Hamiltonian."""
    total = 0.0
    for coef, pauli_string in terms:
        op_psi = apply_pauli_string(state, pauli_string)
        # Inner product <psi | op_psi>
        exp_val = float(np.real(np.vdot(state, op_psi)))
        total += float(coef) * exp_val
    return float(total)


def sample_counts(state: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Sample measurement outcome frequencies in computational basis via multinomial distribution."""
    probs = np.abs(state) ** 2
    probs = probs / np.sum(probs)
    return rng.multinomial(shots, probs)


def estimate_energy_shots(
    state: np.ndarray,
    terms: Sequence[tuple[float, str]],
    shots: int,
    rng: np.random.Generator,
) -> float:
    """Estimate Hamiltonian expectation using finite measurement shots.

    Hamiltonian must partition strictly into Z/I strings and X/I strings.
    Z terms are measured in the computational basis, while X terms are measured
    after applying Hadamard gates to all qubits.
    """
    n = len(terms[0][1])
    z_terms: list[tuple[float, str]] = []
    x_terms: list[tuple[float, str]] = []

    for coef, p_str in terms:
        all_z = all(ch in ("Z", "I") for ch in p_str)
        all_x = all(ch in ("X", "I") for ch in p_str)
        if all_z:
            z_terms.append((coef, p_str))
        elif all_x:
            x_terms.append((coef, p_str))
        else:
            raise ValueError(f"Term '{p_str}' is neither purely Z/I nor purely X/I.")

    total_est = 0.0

    # 1. Measure Z group in computational basis
    if z_terms:
        probs_z = np.abs(state) ** 2
        probs_z = probs_z / np.sum(probs_z)
        sampled_indices = rng.choice(1 << n, size=shots, p=probs_z)

        for coef, p_str in z_terms:
            z_qubits = [q for q, ch in enumerate(p_str) if ch == "Z"]
            if not z_qubits:
                # Identity operator
                total_est += float(coef)
                continue
            # Compute parity: (-1)^(sum_{q in z_qubits} x_q)
            parity = np.ones(shots, dtype=np.float64)
            for q in z_qubits:
                bit_q = (sampled_indices >> (n - 1 - q)) & 1
                parity *= 1.0 - 2.0 * bit_q
            total_est += float(coef) * float(np.mean(parity))

    # 2. Measure X group in Hadamard-rotated basis
    if x_terms:
        state_x = state.copy()
        for q in range(n):
            state_x = apply_1q(state_x, H_GATE, q, n)

        probs_x = np.abs(state_x) ** 2
        probs_x = probs_x / np.sum(probs_x)
        sampled_indices_x = rng.choice(1 << n, size=shots, p=probs_x)

        for coef, p_str in x_terms:
            x_qubits = [q for q, ch in enumerate(p_str) if ch == "X"]
            if not x_qubits:
                total_est += float(coef)
                continue
            parity_x = np.ones(shots, dtype=np.float64)
            for q in x_qubits:
                bit_q = (sampled_indices_x >> (n - 1 - q)) & 1
                parity_x *= 1.0 - 2.0 * bit_q
            total_est += float(coef) * float(np.mean(parity_x))

    return float(total_est)


def apply_depolarizing_layer(
    state: np.ndarray,
    p: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Simulate one trajectory of single-qubit depolarizing noise across all qubits.

    For each qubit independently, with probability p, applies a random Pauli {X, Y, Z}.
    """
    if p <= 0.0:
        return state.copy()

    psi = state.copy()
    paulis = [X_GATE, Y_GATE, Z_GATE]
    for q in range(n):
        if rng.random() < p:
            gate_idx = int(rng.integers(0, 3))
            psi = apply_1q(psi, paulis[gate_idx], q, n)
    return psi


def dense_matrix_from_terms(terms: Sequence[tuple[float, str]], n: int) -> np.ndarray:
    """Construct full 2^n x 2^n matrix representation via Kronecker products (n <= 10)."""
    dim = 1 << n
    matrix = np.zeros((dim, dim), dtype=np.complex128)

    for coef, pauli_str in terms:
        term_mat = PAULI_MAP[pauli_str[0]]
        for char in pauli_str[1:]:
            term_mat = np.kron(term_mat, PAULI_MAP[char])
        matrix += float(coef) * term_mat

    return matrix

