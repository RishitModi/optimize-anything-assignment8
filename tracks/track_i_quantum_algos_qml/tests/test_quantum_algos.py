"""Comprehensive unit and integration test suite for Track I: Quantum Algorithms and QML."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

# Ensure repository root and track root are on sys.path
_TRACK_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in [str(_REPO_ROOT), str(_TRACK_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from qi_algos import (
    balanced_parity,
    bv_distribution,
    constant0,
    dj_probability_all_zero,
    grover_run,
    grover_theory,
    qft,
    qpe_distribution,
    shor_factors,
    shor_order_finding,
    true_order,
)
from qi_anneal import (
    anneal_success,
    chimera_clique_embedding,
    optimal_indices,
    qubo_diag,
)
from qi_qml import (
    compute_kernel_matrix,
    make_dataset,
    param_shift_grad,
    variational_forward,
)
from qi_sim import (
    H,
    X,
    apply_1q,
    apply_cnot,
    apply_cphase,
    dense_cnot,
    dense_unitary_1q,
    probabilities,
    ry,
    zero_state,
)


def test_statevector_simulator_algebraic_equivalence() -> None:
    """Verify simulator tensor contraction and gates against full dense matrices for n=3."""
    rng = np.random.default_rng(42)
    n = 3
    dim = 1 << n

    # 1. Single state test
    st = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
    st = st / np.linalg.norm(st)

    th0, th1, th2 = 0.4, -0.7, 1.2
    th_cp = 0.85

    # Simulator circuit
    out_sim = st.copy()
    out_sim = apply_1q(out_sim, ry(th0), 0, n)
    out_sim = apply_1q(out_sim, ry(th1), 1, n)
    out_sim = apply_1q(out_sim, ry(th2), 2, n)
    out_sim = apply_cnot(out_sim, 0, 2, n)
    out_sim = apply_cphase(out_sim, 1, 2, th_cp, n)

    # Dense matrix construction
    m_ry0 = dense_unitary_1q(ry(th0), 0, n)
    m_ry1 = dense_unitary_1q(ry(th1), 1, n)
    m_ry2 = dense_unitary_1q(ry(th2), 2, n)
    m_cnot = dense_cnot(0, 2, n)

    # Dense cphase matrix
    basis = np.arange(dim)
    mask = (((basis >> (n - 1 - 1)) & 1) == 1) & (((basis >> (n - 1 - 2)) & 1) == 1)
    cp_diag = np.ones(dim, dtype=np.complex128)
    cp_diag[mask] = np.exp(1.0j * th_cp)
    m_cp = np.diag(cp_diag)

    dense_circuit = m_cp @ m_cnot @ m_ry2 @ m_ry1 @ m_ry0
    out_dense = dense_circuit @ st

    assert np.allclose(out_sim, out_dense, atol=1e-12)

    # 2. Batched state test
    B = 5
    st_batch = rng.normal(size=(B, dim)) + 1.0j * rng.normal(size=(B, dim))
    st_batch = st_batch / np.linalg.norm(st_batch, axis=1, keepdims=True)

    out_batch_sim = st_batch.copy()
    out_batch_sim = apply_1q(out_batch_sim, ry(th0), 0, n)
    out_batch_sim = apply_1q(out_batch_sim, ry(th1), 1, n)
    out_batch_sim = apply_1q(out_batch_sim, ry(th2), 2, n)
    out_batch_sim = apply_cnot(out_batch_sim, 0, 2, n)
    out_batch_sim = apply_cphase(out_batch_sim, 1, 2, th_cp, n)

    out_batch_dense = (dense_circuit @ st_batch.T).T
    assert np.allclose(out_batch_sim, out_batch_dense, atol=1e-12)


def test_bell_state_generation() -> None:
    """Verify preparation of maximally entangled Bell state (|00> + |11>) / sqrt(2)."""
    n = 2
    state = zero_state(n)
    state = apply_1q(state, H, 0, n)
    state = apply_cnot(state, 0, 1, n)

    probs = probabilities(state)
    assert math.isclose(probs[0], 0.5, abs_tol=1e-12)  # |00>
    assert math.isclose(probs[3], 0.5, abs_tol=1e-12)  # |11>
    assert math.isclose(probs[1], 0.0, abs_tol=1e-12)  # |01>
    assert math.isclose(probs[2], 0.0, abs_tol=1e-12)  # |10>


def test_qft_and_inverse_identity() -> None:
    """Verify QFT against np.fft.ifft * sqrt(N) and inverse(forward(x)) == x for n=4."""
    rng = np.random.default_rng(123)
    n = 4
    N = 1 << n
    st = rng.normal(size=N) + 1.0j * rng.normal(size=N)
    st = st / np.linalg.norm(st)

    qubits = list(range(n))
    qft_st = qft(st, qubits, n, inverse=False)
    expected = np.fft.ifft(st) * np.sqrt(N)
    assert np.allclose(qft_st, expected, atol=1e-9)

    inv_st = qft(qft_st, qubits, n, inverse=True)
    assert np.allclose(inv_st, st, atol=1e-9)


def test_deutsch_jozsa() -> None:
    """Verify Deutsch-Jozsa returns P=1 for constant and P=0 for balanced oracles (n=4)."""
    n = 4
    p_const = dj_probability_all_zero(n, constant0(n))
    assert math.isclose(p_const, 1.0, abs_tol=1e-9)

    p_bal = dj_probability_all_zero(n, balanced_parity(n))
    assert math.isclose(p_bal, 0.0, abs_tol=1e-9)


def test_bernstein_vazirani() -> None:
    """Verify Bernstein-Vazirani learns secret bitstring in 1 query for n=5."""
    secret = np.array([1, 0, 1, 1, 0])
    sec_idx = int(np.sum(secret * (1 << np.arange(4, -1, -1))))
    dist = bv_distribution(secret)
    assert math.isclose(float(dist[sec_idx]), 1.0, abs_tol=1e-9)


def test_qpe_exact_phase() -> None:
    """Verify QPE distribution for phi = 0.25 (1/4) with t=3 has probability 1 at outcome 2."""
    # With t=3, 2/8 = 0.25, so outcome 2 corresponds exactly to phi
    dist = qpe_distribution(phi=0.25, t=3)
    assert math.isclose(float(dist[2]), 1.0, abs_tol=1e-9)


def test_grover_theory_exact_agreement() -> None:
    """Verify Grover simulation exactly equals analytical formula for n=5, M=1, k=0..5."""
    n = 5
    N = 1 << n
    for k in range(6):
        p_sim, _, _ = grover_run(n, marked=[0], iterations=k)
        p_theo = grover_theory(N, M=1, k=k)
        assert math.isclose(p_sim, p_theo, abs_tol=1e-9)


def test_shor_algorithm_order_and_factors() -> None:
    """Verify Shor's algorithm on N=15, a=7 with t=8 counting qubits."""
    N = 15
    a = 7
    t = 8
    w = 4

    # 1. Classical order check
    r = true_order(a, N)
    assert r == 4

    # 2. Factor extraction
    factors = shor_factors(a, r, N)
    assert factors == {3, 5}

    # 3. Quantum statevector simulation
    probs, _ = shor_order_finding(N, a, t, w)
    assert math.isclose(float(np.sum(probs)), 1.0, abs_tol=1e-9)

    # 4. Multiples of 2^t / r = 256 / 4 = 64 should contain 100% of probability mass
    multiples_64 = np.arange(0, 1 << t, 64)
    mass_on_multiples = float(np.sum(probs[multiples_64]))
    assert math.isclose(mass_on_multiples, 1.0, abs_tol=1e-9)


def test_anneal_success_four_cycle() -> None:
    """Verify quantum annealing on 4-cycle Max-Cut reaches > 0.9 success with norm preservation."""
    n = 4
    # Construct 4-cycle Max-Cut QUBO: edges (0,1), (1,2), (2,3), (3,0)
    # Objective per edge: -x_u - x_v + 2 x_u x_v
    Q = np.zeros((n, n), dtype=np.float64)
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    for u, v in edges:
        Q[u, u] -= 1.0
        Q[v, v] -= 1.0
        mn, mx = min(u, v), max(u, v)
        Q[mn, mx] += 2.0

    diag = qubo_diag(Q, n)
    opt_idx = optimal_indices(diag)

    p_succ, exp_e, norm_err = anneal_success(diag, n, T=50.0, dt=0.05, opt_indices=opt_idx)
    assert norm_err < 1e-9
    assert p_succ > 0.90


def test_chimera_clique_embedding() -> None:
    """Verify TRIAD Chimera embedding capacity limits (n=64 fits, n=65 does not)."""
    fit_64 = chimera_clique_embedding(64)
    assert fit_64["fits_in_2048"] is True
    assert fit_64["M"] == 16
    assert fit_64["physical_qubits"] <= 2048

    fit_65 = chimera_clique_embedding(65)
    assert fit_65["fits_in_2048"] is False
    assert fit_65["M"] == 17
    # n = 65 needs a 17 x 17 Chimera grid, which does not fit the 16 x 16 grid of a 2048-qubit chip, although 1224 < 2048; the capacity limit is the grid size, not the qubit count
    assert fit_65["physical_qubits"] == 4 * 17 * 18
    assert chimera_clique_embedding(100)["physical_qubits"] == 4 * 25 * 26
    assert chimera_clique_embedding(100)["fits_in_2048"] is False


def test_qml_parameter_shift_gradient() -> None:
    """Verify parameter-shift rule matches central finite differences to 1e-6."""
    X, _, y, _ = make_dataset("moons", n=30, seed=0)
    params = np.array([0.15, -0.25, 0.35, -0.45, 0.1, 0.2, -0.3, 0.4, -0.1, 0.2, -0.2, 0.3])

    grad_ps = param_shift_grad(params, X, y)

    eps = 1e-5
    grad_fd = np.empty_like(params)
    for j in range(len(params)):
        p_plus = params.copy()
        p_plus[j] += eps
        l_plus = float(np.mean((variational_forward(p_plus, X) - y) ** 2))

        p_minus = params.copy()
        p_minus[j] -= eps
        l_minus = float(np.mean((variational_forward(p_minus, X) - y) ** 2))

        grad_fd[j] = (l_plus - l_minus) / (2.0 * eps)

    assert np.allclose(grad_ps, grad_fd, atol=1e-6)


def test_qml_quantum_kernel_properties() -> None:
    """Verify quantum kernel Gram matrix is symmetric, has unit diagonal, and is positive semidefinite."""
    X, _, _, _ = make_dataset("circles", n=25, seed=0)
    K = compute_kernel_matrix(X, X)

    # Symmetry
    assert np.allclose(K, K.T, atol=1e-9)
    # Unit diagonal
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)
    # Positive semidefiniteness (eigenvalues >= -1e-9)
    eigs = np.linalg.eigvalsh(K)
    assert np.min(eigs) >= -1e-9


def test_smoke_run_all(tmp_path: Path) -> None:
    """Smoke test running quick mode via dynamic import under track_i_run_all."""
    run_all_path = _TRACK_ROOT / "run_all.py"
    assert run_all_path.is_file(), f"run_all.py missing at {run_all_path}"

    spec = importlib.util.spec_from_file_location("track_i_run_all", str(run_all_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    smoke_out = tmp_path / "smoke_results"

    # Set command line arguments
    sys_argv_orig = sys.argv
    try:
        sys.argv = [
            "run_all.py",
            "--quick",
            "--out", str(smoke_out),
            "--max-minutes", "2.0",
        ]
        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
    finally:
        sys.argv = sys_argv_orig

    assert (smoke_out / "SUMMARY.md").is_file(), "SUMMARY.md was not generated"
    # Check that at least one CSV and PNG exist
    csv_files = list(smoke_out.glob("*.csv"))
    png_files = list(smoke_out.glob("*.png"))
    assert len(csv_files) > 0, "No CSV files generated"
    assert len(png_files) > 0, "No PNG plots generated"

