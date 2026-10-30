"""NumPy statevector quantum circuit simulator for Track I.

CONVENTION:
-----------
- n qubits: the state is a complex128 array whose LAST axis has dimension 2^n.
- Basis index bitstring: x_0 x_1 ... x_{n-1} with qubit 0 being the MOST SIGNIFICANT BIT (MSB).
- Basis state |0> has bit value 0 with Pauli Z eigenvalue +1.
- Basis state |1> has bit value 1 with Pauli Z eigenvalue -1.
- All gate functions support single statevectors of shape (2^n,) as well as batched
  states of shape (B, 2^n).
- 1-qubit gates support batched matrices U of shape (B, 2, 2) for per-sample parameterized execution.
"""

from __future__ import annotations

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

# Standard Pauli and Clifford single-qubit matrices (complex128)
H: np.ndarray = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
X: np.ndarray = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
Y: np.ndarray = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
Z: np.ndarray = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
I2: np.ndarray = np.eye(2, dtype=np.complex128)


def zero_state(n: int, batch_size: int | None = None) -> np.ndarray:
    """Create a quantum zero state |0...0> for n qubits.

    Args:
        n: Number of qubits.
        batch_size: Optional batch size. If None, returns 1D array of shape (2^n,).
                    If an integer, returns 2D array of shape (batch_size, 2^n).

    Returns:
        Complex128 statevector or batch of statevectors.
    """
    dim = 1 << n
    if batch_size is None:
        state = np.zeros(dim, dtype=np.complex128)
        state[0] = 1.0 + 0.0j
        return state
    state = np.zeros((batch_size, dim), dtype=np.complex128)
    state[:, 0] = 1.0 + 0.0j
    return state


def rx(theta: float | np.ndarray) -> np.ndarray:
    """Generate rotation matrix R_X(theta) = exp(-i theta X / 2).

    Args:
        theta: Rotation angle in radians (float scalar or 1D array of shape (B,)).

    Returns:
        Matrix of shape (2, 2) or (B, 2, 2).
    """
    if np.ndim(theta) == 0:
        th = float(theta)
        c = np.cos(th / 2.0)
        s = -1.0j * np.sin(th / 2.0)
        return np.array([[c, s], [s, c]], dtype=np.complex128)
    th_arr = np.asarray(theta, dtype=np.float64)
    c = np.cos(th_arr / 2.0)
    s = -1.0j * np.sin(th_arr / 2.0)
    B = th_arr.shape[0]
    out = np.empty((B, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = c
    out[:, 0, 1] = s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def ry(theta: float | np.ndarray) -> np.ndarray:
    """Generate rotation matrix R_Y(theta) = exp(-i theta Y / 2).

    Args:
        theta: Rotation angle in radians (float scalar or 1D array of shape (B,)).

    Returns:
        Matrix of shape (2, 2) or (B, 2, 2).
    """
    if np.ndim(theta) == 0:
        th = float(theta)
        c = np.cos(th / 2.0)
        s = np.sin(th / 2.0)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    th_arr = np.asarray(theta, dtype=np.float64)
    c = np.cos(th_arr / 2.0)
    s = np.sin(th_arr / 2.0)
    B = th_arr.shape[0]
    out = np.empty((B, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = c
    out[:, 0, 1] = -s
    out[:, 1, 0] = s
    out[:, 1, 1] = c
    return out


def rz(theta: float | np.ndarray) -> np.ndarray:
    """Generate rotation matrix R_Z(theta) = exp(-i theta Z / 2).

    Args:
        theta: Rotation angle in radians (float scalar or 1D array of shape (B,)).

    Returns:
        Matrix of shape (2, 2) or (B, 2, 2).
    """
    if np.ndim(theta) == 0:
        th = float(theta)
        e_neg = np.exp(-0.5j * th)
        e_pos = np.exp(0.5j * th)
        return np.array([[e_neg, 0.0], [0.0, e_pos]], dtype=np.complex128)
    th_arr = np.asarray(theta, dtype=np.float64)
    e_neg = np.exp(-0.5j * th_arr)
    e_pos = np.exp(0.5j * th_arr)
    B = th_arr.shape[0]
    out = np.zeros((B, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = e_neg
    out[:, 1, 1] = e_pos
    return out


def apply_1q(state: np.ndarray, U: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply single-qubit unitary U to qubit q for statevector or batch of statevectors.

    Uses tensor reshaping to (B, 2^q, 2, 2^(n-q-1)) and contraction via np.einsum.
    Supports U of shape (2, 2) or (B, 2, 2).

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        U: 2x2 unitary matrix or batch of matrices (B, 2, 2).
        q: Target qubit index (0 is MSB, n-1 is LSB).
        n: Total number of qubits in the register.

    Returns:
        Updated statevector of same shape as input state.
    """
    is_1d = (state.ndim == 1)
    if is_1d:
        st_batched = state.reshape((1, -1))
    else:
        st_batched = state

    B = st_batched.shape[0]
    dim_prefix = 1 << q
    dim_suffix = 1 << (n - q - 1)

    reshaped = st_batched.reshape((B, dim_prefix, 2, dim_suffix))

    if U.ndim == 2:
        contracted = np.einsum("ij, bajc -> baic", U, reshaped)
    elif U.ndim == 3:
        contracted = np.einsum("bij, bajc -> baic", U, reshaped)
    else:
        raise ValueError(f"U must have 2 or 3 dimensions, got {U.ndim}")

    out = contracted.reshape((B, 1 << n))
    return out[0] if is_1d else out


def apply_cnot(state: np.ndarray, c: int, t: int, n: int) -> np.ndarray:
    """Apply controlled-NOT gate with control c and target t.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        c: Control qubit index (0 is MSB).
        t: Target qubit index (0 is MSB).
        n: Total number of qubits.

    Returns:
        Updated statevector of same shape.
    """
    dim = 1 << n
    basis = np.arange(dim)
    ctrl_mask = (basis >> (n - 1 - c)) & 1
    targ_flip = 1 << (n - 1 - t)
    perm = np.where(ctrl_mask == 1, basis ^ targ_flip, basis)
    return state[..., perm]


def apply_cphase(
    state: np.ndarray,
    c: int,
    t: int,
    theta: float | np.ndarray,
    n: int,
) -> np.ndarray:
    """Apply controlled-phase rotation exp(i theta) when both control and target bits are 1.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        c: Control qubit index.
        t: Target qubit index.
        theta: Phase angle in radians (scalar or array of shape (B,)).
        n: Total number of qubits.

    Returns:
        Updated statevector of same shape.
    """
    dim = 1 << n
    basis = np.arange(dim)
    mask = (((basis >> (n - 1 - c)) & 1) == 1) & (((basis >> (n - 1 - t)) & 1) == 1)

    if np.ndim(theta) == 0:
        factor = np.exp(1.0j * float(theta))
        diag = np.ones(dim, dtype=np.complex128)
        diag[mask] = factor
        return state * diag

    th_arr = np.asarray(theta, dtype=np.float64)
    factors = np.exp(1.0j * th_arr)  # shape (B,)
    diag = np.ones((len(factors), dim), dtype=np.complex128)
    diag[:, mask] = factors[:, None]
    return state * diag


def apply_swap(state: np.ndarray, a: int, b: int, n: int) -> np.ndarray:
    """Apply SWAP gate between qubit a and qubit b.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        a: First qubit index.
        b: Second qubit index.
        n: Total number of qubits.

    Returns:
        Updated statevector of same shape.
    """
    if a == b:
        return state
    dim = 1 << n
    basis = np.arange(dim)
    bit_a = (basis >> (n - 1 - a)) & 1
    bit_b = (basis >> (n - 1 - b)) & 1
    diff = bit_a ^ bit_b
    perm = basis ^ ((diff << (n - 1 - a)) | (diff << (n - 1 - b)))
    return state[..., perm]


def apply_diag(state: np.ndarray, diag: np.ndarray) -> np.ndarray:
    """Apply diagonal unitary operator defined by diagonal vector diag.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        diag: Diagonal vector of shape (2^n,) or (B, 2^n).

    Returns:
        Elementwise product state * diag.
    """
    return state * diag


def apply_h_all(state: np.ndarray, n: int) -> np.ndarray:
    """Apply Hadamard gate to all n qubits in sequence.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        n: Number of qubits.

    Returns:
        Updated statevector after H^n.
    """
    out = state
    for q in range(n):
        out = apply_1q(out, H, q, n)
    return out


def probabilities(state: np.ndarray) -> np.ndarray:
    """Compute measurement outcome probabilities |amplitude|^2 for computational basis.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).

    Returns:
        Real float array of shape (2^n,) or (B, 2^n).
    """
    return np.real(state * np.conj(state))


def sample(state: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Sample measurement bitstrings using numpy multinomial distribution.

    Args:
        state: Single statevector of shape (2^n,) or batched (B, 2^n).
        shots: Number of measurement shots.
        rng: Seeded numpy random generator.

    Returns:
        Array of outcome counts of shape (2^n,) or (B, 2^n).
    """
    probs = probabilities(state)
    if probs.ndim == 1:
        # Normalize slightly in case of floating point precision drift
        p_norm = probs / np.sum(probs)
        return rng.multinomial(shots, p_norm)

    B, dim = probs.shape
    counts = np.empty((B, dim), dtype=np.int64)
    for b in range(B):
        row = probs[b]
        p_norm = row / np.sum(row)
        counts[b] = rng.multinomial(shots, p_norm)
    return counts


def expectation_z(state: np.ndarray, q: int, n: int) -> np.ndarray | float:
    """Compute expectation value <Z_q> of Pauli Z on qubit q.

    Basis states with bit q = 0 have eigenvalue +1; bit q = 1 have eigenvalue -1.

    Args:
        state: Statevector of shape (2^n,) or (B, 2^n).
        q: Qubit index (0 is MSB).
        n: Total number of qubits.

    Returns:
        Float expectation value in [-1, 1], or array of shape (B,).
    """
    dim = 1 << n
    basis = np.arange(dim)
    bit_q = (basis >> (n - 1 - q)) & 1
    z_eigenvals = 1.0 - 2.0 * bit_q.astype(np.float64)

    probs = probabilities(state)
    exp_val = np.sum(probs * z_eigenvals, axis=-1)
    if np.ndim(exp_val) == 0:
        return float(exp_val)
    return exp_val


def depolarizing_layer(
    state: np.ndarray,
    p: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    """Apply single-qubit depolarizing error channel trajectory to a single statevector.

    For each qubit q in 0..n-1, with probability p, applies a Pauli uniformly chosen from {X, Y, Z}.

    Args:
        state: Single statevector of shape (2^n,).
        p: Error probability per qubit.
        rng: Seeded numpy random generator.
        n: Number of qubits.

    Returns:
        Statevector after stochastic error trajectory.
    """
    if p <= 0.0:
        return state
    paulis = [X, Y, Z]
    cur_state = state
    for q in range(n):
        if rng.random() < p:
            pauli_idx = rng.integers(0, 3)
            cur_state = apply_1q(cur_state, paulis[pauli_idx], q, n)
    return cur_state


def dense_unitary_1q(U: np.ndarray, q: int, n: int) -> np.ndarray:
    """Construct full 2^n x 2^n matrix representation of single-qubit gate using np.kron.

    Used for unit tests and direct algebraic verification.

    Args:
        U: 2x2 matrix.
        q: Qubit index (0 is MSB).
        n: Total qubits.

    Returns:
        Full matrix of shape (2^n, 2^n).
    """
    factors = [I2] * n
    factors[q] = U
    mat = factors[0]
    for k in range(1, n):
        mat = np.kron(mat, factors[k])
    return mat


def dense_cnot(c: int, t: int, n: int) -> np.ndarray:
    """Construct full 2^n x 2^n matrix representation of CNOT gate using np.kron.

    Decomposes into |0><0|_c (x) I + |1><1|_c (x) X_t.

    Args:
        c: Control qubit index.
        t: Target qubit index.
        n: Total qubits.

    Returns:
        Full matrix of shape (2^n, 2^n).
    """
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    P1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

    term0_factors = [I2] * n
    term0_factors[c] = P0
    m0 = term0_factors[0]
    for k in range(1, n):
        m0 = np.kron(m0, term0_factors[k])

    term1_factors = [I2] * n
    term1_factors[c] = P1
    term1_factors[t] = X
    m1 = term1_factors[0]
    for k in range(1, n):
        m1 = np.kron(m1, term1_factors[k])

    return m0 + m1

