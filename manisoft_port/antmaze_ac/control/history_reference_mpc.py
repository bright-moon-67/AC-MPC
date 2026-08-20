from __future__ import annotations

import time

import numpy as np
import osqp
import scipy.sparse as sparse
import torch


class FixedCostHistoryKoopmanMPC:
    """OSQP expert used to bootstrap the learned history BC-KMPC actor.

    This is the reusable form of the controller validated by
    ``validate_koopman_mpc_reference_history.py``.  It operates on the same
    absolute-action history model and adds the fixed reference tracking,
    control and absolute-action terms used by the ManiSoft adaptation.
    """

    def __init__(
        self,
        *,
        model,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int = 10,
        state_weight: float = 200.0,
        action_weight: float = 8000.0,
        control_weight: float = 1.0,
        track_tip_only: bool = False,
        tip_indices: tuple[int, int, int] = (30, 31, 32),
        tip_state_scale: float = 5.0,
        tip_axis_scales: tuple[float, float, float] | None = None,
        physical_state_scales: np.ndarray | None = None,
        qp_max_iterations: int = 4000,
        qp_absolute_tolerance: float = 1e-5,
        qp_relative_tolerance: float = 1e-5,
    ) -> None:
        setup_started = time.perf_counter()
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if min(
            state_weight,
            control_weight,
            tip_state_scale,
            qp_absolute_tolerance,
            qp_relative_tolerance,
        ) <= 0:
            raise ValueError("MPC weights, limits and tolerances must be positive")
        if action_weight < 0:
            raise ValueError("action_weight must be non-negative")

        self.model = model
        self.device = state_mean.device
        self.dtype = state_mean.dtype
        self.state_mean_t = state_mean
        self.state_std_t = state_std
        self.state_mean = state_mean.detach().cpu().double().numpy()
        self.state_std = state_std.detach().cpu().double().numpy()
        self.action_low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        self.horizon = int(horizon)
        self.action_dim = int(model.action_dim)
        self.physical_dim = int(model.state_dim)
        self.variable_dim = self.horizon * self.action_dim
        self.state_weight = float(state_weight)
        self.action_weight = float(action_weight)
        self.control_weight = float(control_weight)
        if self.action_low.shape != (self.action_dim,) or self.action_high.shape != (
            self.action_dim,
        ):
            raise ValueError("Action limits do not match the Koopman action dimension")

        tip_index_array = np.asarray(tip_indices, dtype=np.int64)
        if physical_state_scales is not None:
            state_scales = np.asarray(
                physical_state_scales, dtype=np.float64
            ).reshape(-1)
            if state_scales.shape != (self.physical_dim,):
                raise ValueError("physical_state_scales has the wrong shape")
            if np.any(state_scales < 0) or not np.isfinite(state_scales).all():
                raise ValueError("physical_state_scales must be finite and non-negative")
        else:
            state_scales = np.ones(self.physical_dim, dtype=np.float64)
            if track_tip_only:
                state_scales.fill(0.0)
                state_scales[tip_index_array] = 1.0
            else:
                state_scales[tip_index_array] = float(tip_state_scale)
            if tip_axis_scales is not None:
                axis_scales = np.asarray(tip_axis_scales, dtype=np.float64)
                if axis_scales.shape != (3,) or np.any(axis_scales <= 0):
                    raise ValueError(
                        "tip_axis_scales must contain three positive values"
                    )
                state_scales[tip_index_array] = axis_scales
        self.state_scales = state_scales

        A = model.A.detach().cpu().double().numpy()
        B = model.B.detach().cpu().double().numpy()
        C = model.C[: self.physical_dim].detach().cpu().double().numpy()
        powers = [np.eye(model.lifted_dim, dtype=np.float64)]
        for _ in range(self.horizon):
            powers.append(powers[-1] @ A)
        normalized_free_prediction = np.vstack(
            [C @ powers[step + 1] for step in range(self.horizon)]
        )
        normalized_action_prediction = np.zeros(
            (self.horizon * self.physical_dim, self.variable_dim),
            dtype=np.float64,
        )
        for step in range(self.horizon):
            rows = slice(
                step * self.physical_dim,
                (step + 1) * self.physical_dim,
            )
            for control_step in range(step + 1):
                columns = slice(
                    control_step * self.action_dim,
                    (control_step + 1) * self.action_dim,
                )
                normalized_action_prediction[rows, columns] = (
                    C @ powers[step - control_step] @ B
                )

        physical_std = np.tile(self.state_std, self.horizon)
        self.free_prediction = physical_std[:, None] * normalized_free_prediction
        self.action_prediction = (
            physical_std[:, None] * normalized_action_prediction
        )
        self.physical_mean = np.tile(self.state_mean, self.horizon)
        state_diagonal = np.tile(
            self.state_weight * np.square(self.state_scales),
            self.horizon,
        )
        hessian = self.action_prediction.T @ (
            state_diagonal[:, None] * self.action_prediction
        )
        hessian += (self.action_weight + self.control_weight) * np.eye(
            self.variable_dim
        )
        hessian = 2.0 * ((hessian + hessian.T) * 0.5)

        constraint_matrix = sparse.eye(self.variable_dim, format="csc")
        initial_lower = np.tile(self.action_low, self.horizon)
        initial_upper = np.tile(self.action_high, self.horizon)
        self.solver = osqp.OSQP()
        self.solver.setup(
            P=sparse.triu(sparse.csc_matrix(hessian), format="csc"),
            q=np.zeros(self.variable_dim, dtype=np.float64),
            A=constraint_matrix,
            l=initial_lower,
            u=initial_upper,
            verbose=False,
            polishing=True,
            warm_starting=True,
            max_iter=int(qp_max_iterations),
            eps_abs=float(qp_absolute_tolerance),
            eps_rel=float(qp_relative_tolerance),
        )
        self.setup_seconds = float(time.perf_counter() - setup_started)

    def lift(self, state: np.ndarray, context: np.ndarray) -> np.ndarray:
        state_tensor = torch.as_tensor(
            state,
            dtype=self.dtype,
            device=self.device,
        )
        context_tensor = torch.as_tensor(
            context,
            dtype=self.dtype,
            device=self.device,
        )
        normalized_state = (
            state_tensor - self.state_mean_t
        ) / self.state_std_t
        with torch.no_grad():
            lifted = self.model.lift(
                normalized_state.unsqueeze(0),
                context_tensor.unsqueeze(0),
            )[0]
        return lifted.detach().cpu().double().numpy()

    def solve(
        self,
        *,
        state: np.ndarray,
        context: np.ndarray,
        reference_state: np.ndarray,
        reference_action: np.ndarray,
        initial_actions: np.ndarray | None = None,
    ) -> dict[str, np.ndarray | float | int | str]:
        started = time.perf_counter()
        reference_state = np.asarray(reference_state, dtype=np.float64).reshape(-1)
        reference_action = np.asarray(reference_action, dtype=np.float64).reshape(-1)
        if reference_state.shape != (self.physical_dim,):
            raise ValueError("reference_state has the wrong shape")
        if reference_action.shape != (self.action_dim,):
            raise ValueError("reference_action has the wrong shape")

        free_physical = (
            self.free_prediction @ self.lift(state, context)
            + self.physical_mean
        )
        state_error = free_physical - np.tile(reference_state, self.horizon)
        state_diagonal = np.tile(
            self.state_weight * np.square(self.state_scales),
            self.horizon,
        )
        linear = 2.0 * self.action_prediction.T @ (
            state_diagonal * state_error
        )
        linear -= 2.0 * self.action_weight * np.tile(
            reference_action,
            self.horizon,
        )

        self.solver.update(q=linear)
        if initial_actions is not None:
            initial_actions = np.asarray(initial_actions, dtype=np.float64)
            if initial_actions.shape != (self.horizon, self.action_dim):
                raise ValueError("QP warm start has the wrong shape")
            self.solver.warm_start(x=initial_actions.reshape(-1))
        result = self.solver.solve(raise_error=False)
        if result.info.status_val not in {1, 2} or result.x is None:
            raise RuntimeError(
                "History Koopman MPC expert failed: "
                f"status={result.info.status!r}, iterations={result.info.iter}"
            )
        actions = np.asarray(result.x, dtype=np.float64).reshape(
            self.horizon,
            self.action_dim,
        )
        actions = np.clip(actions, self.action_low, self.action_high)
        predicted_physical = (
            free_physical + self.action_prediction @ actions.reshape(-1)
        ).reshape(self.horizon, self.physical_dim)
        weighted_error = (
            predicted_physical - reference_state
        ) * self.state_scales
        cost = self.state_weight * np.square(weighted_error).sum()
        cost += self.action_weight * np.square(
            actions - reference_action
        ).sum()
        cost += self.control_weight * np.square(actions).sum()
        return {
            "actions": actions.astype(np.float32),
            "action": actions[0].astype(np.float32),
            "cost": float(cost),
            "qp_status": str(result.info.status),
            "qp_iterations": int(result.info.iter),
            "solve_seconds": float(time.perf_counter() - started),
        }
