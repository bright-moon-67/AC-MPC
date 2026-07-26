from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DAREResult:
    P: torch.Tensor
    gain: torch.Tensor
    converged: torch.Tensor
    iterations: int
    residual: torch.Tensor
    relative_residual: torch.Tensor
    condition_number: torch.Tensor
    closed_loop_spectral_radius: torch.Tensor


def _batched(*matrices: torch.Tensor) -> tuple[list[torch.Tensor], bool]:
    single = all(matrix.ndim == 2 for matrix in matrices)
    converted = [matrix.unsqueeze(0) if matrix.ndim == 2 else matrix for matrix in matrices]
    batch_sizes = [matrix.shape[0] for matrix in converted]
    batch = max(batch_sizes)
    output = []
    for matrix in converted:
        if matrix.shape[0] not in (1, batch):
            raise ValueError(f"Incompatible batch dimensions {batch_sizes}")
        output.append(matrix.expand(batch, -1, -1))
    return output, single


def stabilizability_diagnostic(A: torch.Tensor, B: torch.Tensor, tolerance: float = 1e-7) -> list[str]:
    """PBH diagnostics for discrete-time unstable modes.

    This is a diagnostic gate, not part of the differentiable computation.
    """

    (A_batch, B_batch), _ = _batched(A, B)
    messages = []
    with torch.no_grad():
        for batch_index, (a, b) in enumerate(zip(A_batch, B_batch)):
            eigenvalues = torch.linalg.eigvals(a)
            failures = []
            identity = torch.eye(a.shape[-1], dtype=eigenvalues.dtype, device=a.device)
            complex_b = b.to(eigenvalues.dtype)
            for eigenvalue in eigenvalues[torch.abs(eigenvalues) >= 1.0 - tolerance]:
                pbh = torch.cat((eigenvalue * identity - a.to(eigenvalues.dtype), complex_b), dim=-1)
                smallest = torch.linalg.svdvals(pbh)[-1]
                if smallest < tolerance:
                    failures.append(f"{eigenvalue.item()} (PBH sigma_min={smallest.item():.2e})")
            if failures:
                messages.append(f"batch {batch_index}: uncontrollable unstable mode(s): {', '.join(failures)}")
    return messages


def detectability_diagnostic(
    A: torch.Tensor,
    output_matrix: torch.Tensor,
    tolerance: float = 1e-7,
) -> list[str]:
    """PBH diagnostics for discrete-time unobservable unstable modes."""

    (A_batch, output_batch), _ = _batched(A, output_matrix)
    messages = []
    with torch.no_grad():
        for batch_index, (a, output) in enumerate(zip(A_batch, output_batch)):
            eigenvalues = torch.linalg.eigvals(a)
            failures = []
            identity = torch.eye(a.shape[-1], dtype=eigenvalues.dtype, device=a.device)
            complex_output = output.to(eigenvalues.dtype)
            for eigenvalue in eigenvalues[torch.abs(eigenvalues) >= 1.0 - tolerance]:
                pbh = torch.cat(
                    (eigenvalue * identity - a.to(eigenvalues.dtype), complex_output),
                    dim=-2,
                )
                smallest = torch.linalg.svdvals(pbh)[-1]
                if smallest < tolerance:
                    failures.append(
                        f"{eigenvalue.item()} (PBH sigma_min={smallest.item():.2e})"
                    )
            if failures:
                messages.append(
                    f"batch {batch_index}: unobservable unstable mode(s): "
                    + ", ".join(failures)
                )
    return messages


def _cost_detectability_diagnostic(
    A: torch.Tensor,
    Q: torch.Tensor,
    tolerance: float = 1e-7,
) -> list[str]:
    (A_batch, Q_batch), _ = _batched(A, Q)
    symmetric_q = 0.5 * (Q_batch + Q_batch.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric_q)
    if bool(torch.any(eigenvalues < -tolerance)):
        return ["stage state Hessian Q is not positive semidefinite"]
    # output' output = Q. Row scaling by sqrt(eigenvalues) preserves the PBH
    # nullspace relevant to detectability.
    output = torch.diag_embed(eigenvalues.clamp_min(0).sqrt()) @ eigenvectors.mT
    return detectability_diagnostic(A_batch, output, tolerance)


def dare_residual(A: torch.Tensor, B: torch.Tensor, Q: torch.Tensor, R: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    (A, B, Q, R, P), single = _batched(A, B, Q, R, P)
    bt_p = B.mT @ P
    gain = torch.linalg.solve(R + bt_p @ B, bt_p @ A)
    equation = A.mT @ P @ A - A.mT @ P @ B @ gain + Q - P
    residual = torch.linalg.matrix_norm(equation, ord="fro", dim=(-2, -1))
    return residual[0] if single else residual


def solve_dare(
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    *,
    tolerance: float = 1e-7,
    max_iterations: int = 500,
    jitter: float = 1e-9,
    check_stabilizable: bool = True,
    check_detectable: bool = True,
    fail_on_nonconvergence: bool = True,
) -> DAREResult:
    """Solve batched DARE with differentiable structured doubling.

    Full-A Koopman models can contain controllable modes extremely close to the
    unit circle. Plain Riccati value iteration then converges linearly and is
    numerically unusable in float32. Structured doubling converges
    quadratically; float16/float32 inputs are promoted to float64 internally so
    the stabilizing solution and affine term remain reliable.
    """

    # Check the system pair before broadcasting shared A/B over a state-cost
    # batch. Otherwise an identical, relatively expensive PBH check would run
    # once per PPO minibatch element.
    if check_stabilizable:
        failures = stabilizability_diagnostic(A, B)
        if failures:
            raise RuntimeError("DARE stabilizability check failed: " + "; ".join(failures))
    if check_detectable:
        failures = _cost_detectability_diagnostic(A, Q)
        if failures:
            raise RuntimeError("DARE detectability check failed: " + "; ".join(failures))
    (A, B, Q, R), single = _batched(A, B, Q, R)
    n, m = A.shape[-1], B.shape[-1]
    if A.shape[-2:] != (n, n) or B.shape[-2] != n or Q.shape[-2:] != (n, n) or R.shape[-2:] != (m, m):
        raise ValueError("Invalid A, B, Q, R shapes")
    if not all(torch.isfinite(matrix).all() for matrix in (A, B, Q, R)):
        raise FloatingPointError("DARE inputs contain NaN or Inf")
    input_dtype = A.dtype
    work_dtype = (
        torch.float64
        if input_dtype in (torch.float16, torch.bfloat16, torch.float32)
        else input_dtype
    )
    A = A.to(work_dtype)
    B = B.to(work_dtype)
    Q = Q.to(work_dtype)
    R = R.to(work_dtype)
    batch = A.shape[0]
    identity_n = torch.eye(n, dtype=work_dtype, device=A.device).expand(batch, n, n)
    identity_r = torch.eye(m, dtype=work_dtype, device=A.device).expand(batch, m, m)
    regularized_r = R + float(jitter) * identity_r

    # SDA initialization for
    # P = Q + A' P A - A' P B (R + B' P B)^-1 B' P A.
    doubling_a = A
    doubling_g = B @ torch.linalg.solve(regularized_r, B.mT)
    P = 0.5 * (Q + Q.mT)
    converged = torch.zeros(A.shape[0], dtype=torch.bool, device=A.device)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        h_g_system = identity_n + P @ doubling_g
        g_h_system = identity_n + doubling_g @ P
        next_a = doubling_a @ torch.linalg.solve(g_h_system, doubling_a)
        next_g = (
            doubling_g
            + doubling_a
            @ doubling_g
            @ torch.linalg.solve(h_g_system, doubling_a.mT)
        )
        next_p = (
            P
            + doubling_a.mT
            @ torch.linalg.solve(h_g_system, P @ doubling_a)
        )
        next_g = 0.5 * (next_g + next_g.mT)
        next_p = 0.5 * (next_p + next_p.mT)
        difference = torch.linalg.matrix_norm(next_p - P, ord="fro", dim=(-2, -1))
        scale = 1.0 + torch.linalg.matrix_norm(next_p, ord="fro", dim=(-2, -1))
        converged = difference <= tolerance * scale
        doubling_a = next_a
        doubling_g = next_g
        P = next_p
        if bool(torch.all(converged)):
            break

    bt_p = B.mT @ P
    system = regularized_r + bt_p @ B
    gain = torch.linalg.solve(system, bt_p @ A)
    closed_loop = A - B @ gain
    residual = dare_residual(A, B, Q, R, P)
    relative_residual = residual / (
        1.0 + torch.linalg.matrix_norm(P, ord="fro", dim=(-2, -1))
    )
    condition = torch.linalg.cond(system)
    spectral_radius = torch.max(torch.abs(torch.linalg.eigvals(closed_loop)), dim=-1).values.real
    finite = all(
        torch.isfinite(value).all()
        for value in (
            P,
            gain,
            residual,
            relative_residual,
            condition,
            spectral_radius,
        )
    )
    if not finite:
        raise FloatingPointError("DARE produced NaN or Inf")
    if fail_on_nonconvergence and not bool(torch.all(converged)):
        failed = torch.nonzero(~converged, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            f"DARE did not converge for batch indices {failed} after {max_iterations} iterations; "
            f"residuals={residual.detach().cpu().tolist()}"
        )
    validation_tolerance = max(100.0 * float(tolerance), 1e-8)
    valid_solution = (relative_residual <= validation_tolerance) & (
        spectral_radius < 1.0
    )
    if fail_on_nonconvergence and not bool(torch.all(valid_solution)):
        failed = torch.nonzero(~valid_solution, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "DARE returned a non-stabilizing or inaccurate solution for batch "
            f"indices {failed}; relative_residuals="
            f"{relative_residual.detach().cpu().tolist()}, "
            "closed_loop_spectral_radius="
            f"{spectral_radius.detach().cpu().tolist()}"
        )
    converged = converged & valid_solution

    def restore(value: torch.Tensor) -> torch.Tensor:
        return value[0] if single else value

    return DAREResult(
        P=restore(P),
        gain=restore(gain),
        converged=restore(converged),
        iterations=iterations,
        residual=restore(residual),
        relative_residual=restore(relative_residual),
        condition_number=restore(condition),
        closed_loop_spectral_radius=restore(spectral_radius),
    )
