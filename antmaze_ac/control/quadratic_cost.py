from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LiftedQuadraticCost:
    state_hessian: torch.Tensor
    control_hessian: torch.Tensor
    state_linear: torch.Tensor
    control_linear: torch.Tensor


def physical_to_lifted_cost(
    C: torch.Tensor,
    stage_hessian_diag: torch.Tensor,
    stage_linear: torch.Tensor,
    state_dim: int,
) -> LiftedQuadraticCost:
    """Map diagonal cost on ``y=[x,v]`` into lifted state/control blocks."""

    q_x = stage_hessian_diag[..., :state_dim]
    q_u = stage_hessian_diag[..., state_dim:]
    p_x = stage_linear[..., :state_dim]
    p_u = stage_linear[..., state_dim:]
    state_hessian = C.mT @ torch.diag_embed(q_x) @ C
    control_hessian = torch.diag_embed(q_u)
    state_linear = p_x @ C
    return LiftedQuadraticCost(state_hessian, control_hessian, state_linear, p_u)
