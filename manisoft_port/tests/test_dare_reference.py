import numpy as np
import torch
from scipy.linalg import solve_discrete_are

from antmaze_ac.control.differentiable_dare import (
    detectability_diagnostic,
    solve_dare,
)


def system(dtype=torch.float64):
    A = torch.tensor([[1.0, 0.1], [0.0, 0.95]], dtype=dtype)
    B = torch.tensor([[0.0], [0.1]], dtype=dtype)
    Q = torch.diag(torch.tensor([2.0, 0.5], dtype=dtype))
    R = torch.tensor([[0.2]], dtype=dtype)
    return A, B, Q, R


def test_scalar_and_scipy_reference_and_residual():
    A, B, Q, R = system()
    result = solve_dare(A, B, Q, R, tolerance=1e-11, max_iterations=5000, jitter=0.0)
    scipy_p = solve_discrete_are(A.numpy(), B.numpy(), Q.numpy(), R.numpy())
    np.testing.assert_allclose(result.P.numpy(), scipy_p, rtol=1e-8, atol=1e-9)
    assert result.residual < 1e-8
    assert result.closed_loop_spectral_radius < 1


def test_batch_matches_individual():
    A, B, Q, R = system()
    q_batch = torch.stack((Q, 2 * Q))
    batched = solve_dare(A, B, q_batch, R, tolerance=1e-10, max_iterations=2000, jitter=0.0)
    for index in range(2):
        individual = solve_dare(A, B, q_batch[index], R, tolerance=1e-10, max_iterations=2000, jitter=0.0)
        torch.testing.assert_close(batched.P[index], individual.P)
        torch.testing.assert_close(batched.gain[index], individual.gain)


def test_spectral_radius_can_be_deferred_without_skipping_residual_checks():
    A, B, Q, R = system()
    result = solve_dare(
        A,
        B,
        Q,
        R,
        tolerance=1e-11,
        max_iterations=5000,
        jitter=0.0,
        compute_closed_loop_spectral_radius=False,
    )
    assert bool(result.converged)
    assert torch.isfinite(result.P).all()
    assert torch.isfinite(result.gain).all()
    assert torch.isfinite(result.relative_residual)
    assert result.relative_residual < 1e-8
    assert torch.isnan(result.closed_loop_spectral_radius)


def test_near_unit_float32_system_is_promoted_and_stabilized():
    torch.manual_seed(9)
    A = torch.diag(torch.tensor([1.003, 1.001, 0.999, 0.97], dtype=torch.float32))
    A[0, 1] = 0.01
    B = torch.tensor(
        [[0.2, 0.0], [0.0, 0.2], [0.1, -0.1], [0.05, 0.1]],
        dtype=torch.float32,
    )
    Q = torch.diag(torch.tensor([5.0, 2.0, 1.0, 0.5], dtype=torch.float32))
    R = torch.eye(2, dtype=torch.float32) * 0.05
    result = solve_dare(A, B, Q, R, tolerance=1e-9, max_iterations=100, jitter=0.0)
    scipy_p = solve_discrete_are(
        A.double().numpy(),
        B.double().numpy(),
        Q.double().numpy(),
        R.double().numpy(),
    )
    assert result.P.dtype == torch.float64
    np.testing.assert_allclose(result.P.detach().numpy(), scipy_p, rtol=1e-7, atol=1e-8)
    assert result.iterations < 30
    assert result.relative_residual < 1e-10
    assert result.closed_loop_spectral_radius < 1


def test_unstabilizable_diagnostic():
    A = torch.diag(torch.tensor([1.1, 0.9], dtype=torch.float64))
    B = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    try:
        solve_dare(A, B, torch.eye(2, dtype=A.dtype), torch.eye(1, dtype=A.dtype))
    except RuntimeError as error:
        assert "uncontrollable unstable" in str(error)
    else:
        raise AssertionError("unstabilizable system should fail")


def test_nonconvergence_has_clear_diagnostic():
    A, B, Q, R = system()
    try:
        solve_dare(
            A,
            B,
            Q,
            R,
            tolerance=1e-15,
            max_iterations=1,
            check_stabilizable=False,
        )
    except RuntimeError as error:
        message = str(error)
        assert "did not converge" in message
        assert "residuals=" in message
    else:
        raise AssertionError("one Riccati iteration should not converge")


def test_nonconvergence_can_return_diagnostics_without_raising():
    A, B, Q, R = system()
    result = solve_dare(
        A,
        B,
        Q,
        R,
        tolerance=1e-15,
        max_iterations=1,
        check_stabilizable=False,
        fail_on_nonconvergence=False,
    )
    assert not bool(result.converged)
    assert torch.isfinite(result.P).all()
    assert torch.isfinite(result.gain).all()


def test_unobservable_unstable_mode_diagnostic():
    A = torch.diag(torch.tensor([1.1, 0.9], dtype=torch.float64))
    output = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    failures = detectability_diagnostic(A, output)
    assert failures and "unobservable unstable" in failures[0]
    Q = output.mT @ output
    B = torch.eye(2, dtype=torch.float64)
    try:
        solve_dare(A, B, Q, torch.eye(2, dtype=torch.float64))
    except RuntimeError as error:
        assert "detectability check failed" in str(error)
    else:
        raise AssertionError("undetectable unstable system should fail")


def test_float32_near_unit_circle_uses_accurate_doubling_solution():
    A = torch.diag(torch.tensor([1.001, 0.999, 0.98, 0.9], dtype=torch.float32))
    B = torch.tensor(
        [[0.2, 0.0], [0.1, 0.1], [0.0, 0.2], [0.1, -0.1]],
        dtype=torch.float32,
    )
    Q = torch.eye(4, dtype=torch.float32)
    R = torch.eye(2, dtype=torch.float32) * 0.05
    result = solve_dare(A, B, Q, R, tolerance=1e-9, max_iterations=100, jitter=0.0)
    scipy_p = solve_discrete_are(
        A.numpy().astype(np.float64),
        B.numpy().astype(np.float64),
        Q.numpy().astype(np.float64),
        R.numpy().astype(np.float64),
    )
    assert result.P.dtype == torch.float64
    np.testing.assert_allclose(result.P.numpy(), scipy_p, rtol=1e-7, atol=1e-8)
    assert result.iterations < 30
    assert result.closed_loop_spectral_radius < 1
