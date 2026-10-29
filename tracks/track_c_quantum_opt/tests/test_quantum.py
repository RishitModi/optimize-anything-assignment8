"""Unit and regression tests for Track C quantum optimization modules."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from optimizers_q import cobyla, gd_param_shift, qng, spsa
from qaoa import ising_diag, qaoa_expectation, qubo_diag
from qsim import (
    H_GATE,
    I_GATE,
    X_GATE,
    apply_1q,
    apply_cnot,
    apply_depolarizing_layer,
    estimate_energy_shots,
    expectation_pauli_hamiltonian,
    ry,
    zero_state,
)
from qubo import qubo_to_ising
from run_all import run_all
from vqe import ansatz_state, exact_ground_energy, metric_tensor, param_shift_grad, tfim_terms, vqe_energy


def test_qsim_dense_matrix_contract_n3() -> None:
    """Verify single-qubit RY gates and CNOT(0, 2) against Kronecker product matrices for n=3."""
    n = 3
    rng = np.random.default_rng(101)
    # Random normalized complex state
    raw_state = rng.normal(size=1 << n) + 1.0j * rng.normal(size=1 << n)
    state = raw_state / np.linalg.norm(raw_state)

    th0, th1, th2 = 0.4, 0.9, -0.6
    ry0, ry1, ry2 = ry(th0), ry(th1), ry(th2)

    # 1. Simulator sequential single-qubit application
    sim_ry = apply_1q(state, ry0, 0, n)
    sim_ry = apply_1q(sim_ry, ry1, 1, n)
    sim_ry = apply_1q(sim_ry, ry2, 2, n)

    # Dense matrix Kronecker product
    dense_ry = np.kron(ry0, np.kron(ry1, ry2))
    expected_ry = dense_ry @ state
    assert np.allclose(sim_ry, expected_ry, atol=1e-12)

    # 2. CNOT with control=0, target=2
    sim_cnot = apply_cnot(state, 0, 2, n)

    p0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    p1 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    dense_cnot = np.kron(p0, np.kron(I_GATE, I_GATE)) + np.kron(p1, np.kron(I_GATE, X_GATE))
    expected_cnot = dense_cnot @ state
    assert np.allclose(sim_cnot, expected_cnot, atol=1e-12)


def test_bell_state_generation() -> None:
    """Verify that H + CNOT produces the Bell state |Phi+> with equal probabilities on |00> and |11>."""
    state = zero_state(2)
    state = apply_1q(state, H_GATE, 0, 2)
    state = apply_cnot(state, 0, 1, 2)

    probs = np.abs(state) ** 2
    assert math.isclose(probs[0], 0.5, abs_tol=1e-12)  # |00>
    assert math.isclose(probs[1], 0.0, abs_tol=1e-12)  # |01>
    assert math.isclose(probs[2], 0.0, abs_tol=1e-12)  # |10>
    assert math.isclose(probs[3], 0.5, abs_tol=1e-12)  # |11>


def test_qubo_ising_diag_equivalence_n5() -> None:
    """Verify qubo_diag matches ising_diag across all 32 basis states for random 5-variable Q."""
    n = 5
    rng = np.random.default_rng(202)
    Q = np.triu(rng.normal(size=(n, n)))

    h, J, offset = qubo_to_ising(Q)

    diag_q = qubo_diag(Q)
    diag_i = ising_diag(h, J, offset, n)

    assert np.allclose(diag_q, diag_i, atol=1e-12)


def test_qaoa_zero_parameters_expectation() -> None:
    """Verify that QAOA expectation with all parameters set to zero returns the mean diagonal energy."""
    n = 4
    p = 2
    diag = np.linspace(-5.0, 5.0, 1 << n)
    params = np.zeros(2 * p)

    exp_val = qaoa_expectation(params, diag, n, p)
    expected_mean = float(np.mean(diag))
    assert math.isclose(exp_val, expected_mean, abs_tol=1e-12)


def test_param_shift_vs_finite_difference() -> None:
    """Verify parameter-shift rule against central finite difference on random 3-qubit, L=2 ansatz."""
    n = 3
    L = 2
    P = n * (L + 1)
    rng = np.random.default_rng(303)
    params = rng.uniform(0.0, 2.0 * np.pi, size=P)
    terms = tfim_terms(n, g=0.8)

    grad_ps = param_shift_grad(params, terms, n, L)

    eps = 1e-5
    grad_fd = np.zeros(P)
    for i in range(P):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        grad_fd[i] = (vqe_energy(p_plus, terms, n, L) - vqe_energy(p_minus, terms, n, L)) / (2.0 * eps)

    assert np.allclose(grad_ps, grad_fd, atol=1e-6)


def test_metric_tensor_vs_fidelity_hessian() -> None:
    """Verify Fubini-Study metric equals -0.5 * Fidelity Hessian at random 2-qubit, L=1 point."""
    n = 2
    L = 1
    P = n * (L + 1)
    rng = np.random.default_rng(404)
    params = rng.uniform(0.0, 2.0 * np.pi, size=P)

    g_metric = metric_tensor(params, n, L)

    psi_base = ansatz_state(params, n, L)
    eps = 1e-3
    hess_fid = np.zeros((P, P))

    for i in range(P):
        for j in range(P):
            pp = params.copy()
            pp[i] += eps
            pp[j] += eps
            f_pp = abs(np.vdot(psi_base, ansatz_state(pp, n, L))) ** 2

            pm = params.copy()
            pm[i] += eps
            pm[j] -= eps
            f_pm = abs(np.vdot(psi_base, ansatz_state(pm, n, L))) ** 2

            mp = params.copy()
            mp[i] -= eps
            mp[j] += eps
            f_mp = abs(np.vdot(psi_base, ansatz_state(mp, n, L))) ** 2

            mm = params.copy()
            mm[i] -= eps
            mm[j] -= eps
            f_mm = abs(np.vdot(psi_base, ansatz_state(mm, n, L))) ** 2

            hess_fid[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)

    g_expected = -0.5 * hess_fid
    assert np.allclose(g_metric, g_expected, atol=1e-4)


def test_tfim_exact_ground_energy_limits() -> None:
    """Verify TFIM exact ground energy at g=0 is -(n-1) and for decoupled transverse field is -g*n."""
    n = 3

    # Limit 1: g = 0 -> open chain of (n-1) couplings with ground state energy -(n-1)
    terms_g0 = tfim_terms(n, g=0.0)
    e0_g0 = exact_ground_energy(terms_g0, n)
    assert math.isclose(e0_g0, -(n - 1), abs_tol=1e-10)

    # Limit 2: purely transverse field without ZZ coupling
    terms_no_zz = [(-1.5, "".join(["X" if q == i else "I" for q in range(n)])) for i in range(n)]
    e0_x = exact_ground_energy(terms_no_zz, n)
    assert math.isclose(e0_x, -1.5 * n, abs_tol=1e-10)


def test_estimate_energy_shots_consistency() -> None:
    """Verify estimate_energy_shots mean over 2000 repetitions is within 4 standard errors of exact."""
    n = 3
    rng = np.random.default_rng(505)
    raw = rng.normal(size=1 << n) + 1.0j * rng.normal(size=1 << n)
    state = raw / np.linalg.norm(raw)

    terms = tfim_terms(n, g=0.7)
    e_exact = expectation_pauli_hamiltonian(state, terms)

    n_reps = 2000
    estimates = [estimate_energy_shots(state, terms, 1000, rng) for _ in range(n_reps)]

    sample_mean = float(np.mean(estimates))
    std_err = float(np.std(estimates) / math.sqrt(n_reps))

    assert abs(sample_mean - e_exact) <= 4.0 * std_err


def test_depolarizing_noise_zero_probability() -> None:
    """Verify that depolarizing noise with p=0 leaves any quantum state unchanged."""
    n = 3
    rng = np.random.default_rng(606)
    raw = rng.normal(size=1 << n) + 1.0j * rng.normal(size=1 << n)
    state = raw / np.linalg.norm(raw)

    noisy_state = apply_depolarizing_layer(state, 0.0, rng, n)
    assert np.allclose(state, noisy_state, atol=1e-14)


def test_smoke_track_c_run_all(tmp_path: Path) -> None:
    """Execute Track C run_all in quick mode into tmp_path and verify output artifacts."""
    out_dir = tmp_path / "track_c_test_results"
    exit_code = run_all(quick=True, out=str(out_dir))
    assert exit_code == 0, "run_all returned non-zero exit code"

    assert (out_dir / "SUMMARY.md").exists()
    assert (out_dir / "SUMMARY.md").stat().st_size > 0

    # Verify at least one CSV and PNG from each experiment
    expected_files = [
        "c1_qaoa_results.csv",
        "c1_qaoa_ratio_vs_p.png",
        "c2_vqe_summary.csv",
        "c2_vqe_convergence.png",
        "c3_shot_noise_scaling.csv",
        "c3_shot_rmse_scaling.png",
        "c4_qaoa_noise.csv",
        "c4_noise_impact.png",
        "c5_barren_plateaus.csv",
        "c5_barren_plateaus.png",
    ]
    for fname in expected_files:
        fpath = out_dir / fname
        assert fpath.exists(), f"Missing artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Empty artifact: {fname}"

