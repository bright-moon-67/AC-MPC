from __future__ import annotations

from dataclasses import dataclass

import torch

from .differentiable_dare import DAREResult, solve_dare


@dataclass
class AffineLQRResult:
    gain: torch.Tensor
    feedforward: torch.Tensor
    value_hessian: torch.Tensor
    value_linear_half: torch.Tensor
    dare: DAREResult


def affine_lqr(
    A: torch.Tensor,
    B: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    q: torch.Tensor,
    r: torch.Tensor,
    **dare_kwargs,
) -> AffineLQRResult:
    """Solve infinite-horizon affine LQR.

    Stage cost is ``z'Qz + u'Ru + q'z + r'u`` and the resulting controller is
    ``u=-Kz-d``. The linear half-vector ``s`` satisfies
    ``V(z)=z'Pz+2s'z+c``.
    """

    dare = solve_dare(A, B, Q, R, **dare_kwargs)
    single = dare.P.ndim == 2
    P = dare.P.unsqueeze(0) if single else dare.P
    K = dare.gain.unsqueeze(0) if single else dare.gain
    batch = P.shape[0]

    def expand_matrix(value: torch.Tensor) -> torch.Tensor:
        value = value.unsqueeze(0) if value.ndim == 2 else value
        return value.to(P.dtype).expand(batch, -1, -1)

    A_b, B_b, Q_b, R_b = map(expand_matrix, (A, B, Q, R))
    q_b = (q.unsqueeze(0) if q.ndim == 1 else q).to(P.dtype)
    r_b = (r.unsqueeze(0) if r.ndim == 1 else r).to(P.dtype)
    q_b = q_b.expand(batch, -1)
    r_b = r_b.expand(batch, -1)
    n = A_b.shape[-1]
    if q_b.shape != (batch, n) or r_b.shape != (batch, B_b.shape[-1]):
        raise ValueError("q and r must have shapes [batch,n] and [batch,m]")
    closed_loop = A_b - B_b @ K
    identity = torch.eye(n, dtype=P.dtype, device=P.device).expand(batch, n, n)
    rhs = 0.5 * (q_b - (K.mT @ r_b.unsqueeze(-1)).squeeze(-1))
    s = torch.linalg.solve(identity - closed_loop.mT, rhs.unsqueeze(-1)).squeeze(-1)
    control_system = R_b + B_b.mT @ P @ B_b
    d = torch.linalg.solve(
        control_system,
        B_b.mT @ s.unsqueeze(-1) + 0.5 * r_b.unsqueeze(-1),
    ).squeeze(-1)
    if not (torch.isfinite(s).all() and torch.isfinite(d).all()):
        raise FloatingPointError("Affine LQR produced NaN or Inf")
    return AffineLQRResult(
        gain=K[0] if single else K,
        feedforward=d[0] if single else d,
        value_hessian=P[0] if single else P,
        value_linear_half=s[0] if single else s,
        dare=dare,
    )
