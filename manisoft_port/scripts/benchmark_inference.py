#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from antmaze_ac.control.quadratic_cost import physical_to_lifted_cost
from antmaze_ac.control.steady_state_lqr import affine_lqr
from antmaze_ac.envs.factory import make_antmaze_env
from antmaze_ac.rl.ac_koopman_policy import GainHoldController
from antmaze_ac.rl.serialization import load_actor_checkpoint, make_policy


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(callable_, iterations, device):
    values = []
    for _ in range(iterations):
        synchronize(device)
        started = time.perf_counter_ns()
        callable_()
        synchronize(device)
        values.append((time.perf_counter_ns() - started) / 1e6)
    values = np.asarray(values)
    return {
        "mean_ms": float(values.mean()),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--koopman-checkpoint", default=None)
    parser.add_argument("--actor-checkpoint", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--control-episodes", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not args.koopman_checkpoint and not args.actor_checkpoint:
        raise ValueError("Provide --koopman-checkpoint or --actor-checkpoint")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    if args.actor_checkpoint:
        policy, _, koopman_payload = load_actor_checkpoint(args.actor_checkpoint, device)
    else:
        policy, koopman_payload = make_policy(args.koopman_checkpoint, device)
    policy.eval()
    control_period_ms = float(
        koopman_payload["config"]["evaluation"]["control_period_ms"]
    )
    iterations = int(
        args.iterations
        or koopman_payload["config"]["evaluation"]["benchmark_iterations"]
    )
    warmup = int(
        args.warmup
        or koopman_payload["config"]["evaluation"]["benchmark_warmup"]
    )
    gain_update_intervals = [
        int(value)
        for value in koopman_payload["config"]["evaluation"]["gain_update_intervals"]
    ]
    observation = policy.state_mean.clone()
    normalized = policy.normalize(observation)

    def encoder():
        return policy.koopman.lift(normalized)

    def actor():
        return policy.actor(normalized)

    q_diag, p = actor()
    cost = physical_to_lifted_cost(
        policy.koopman.C, q_diag, p, policy.koopman.state_dim
    )

    def dare():
        return affine_lqr(
            policy.koopman.A,
            policy.koopman.B,
            cost.state_hessian,
            cost.control_hessian,
            cost.state_linear,
            cost.control_linear,
            **policy.dare_kwargs,
        )

    lqr = dare()
    lifted = encoder()

    def feedback():
        lifted_for_control = lifted.to(lqr.gain.dtype)
        return (
            -(lqr.gain @ lifted_for_control.unsqueeze(-1)).squeeze(-1)
            - lqr.feedforward
        )

    def total():
        return policy(observation)

    with torch.no_grad():
        for _ in range(warmup):
            total()
        components = {
            "encoder": measure(encoder, iterations, device),
            "actor": measure(actor, iterations, device),
            "DARE": measure(dare, iterations, device),
            "feedback": measure(feedback, iterations, device),
            "total_interval_1": measure(total, iterations, device),
        }
    intervals = {}
    for interval in gain_update_intervals:
        mean = (
            components["encoder"]["mean_ms"]
            + components["feedback"]["mean_ms"]
            + (components["actor"]["mean_ms"] + components["DARE"]["mean_ms"]) / interval
        )
        p95 = (
            components["encoder"]["p95_ms"]
            + components["feedback"]["p95_ms"]
            + (components["actor"]["p95_ms"] + components["DARE"]["p95_ms"]) / interval
        )
        maximum = (
            components["encoder"]["max_ms"]
            + components["feedback"]["max_ms"]
            + (components["actor"]["max_ms"] + components["DARE"]["max_ms"]) / interval
        )
        intervals[str(interval)] = {
            "amortized_mean_ms": mean,
            "amortized_p95_ms": p95,
            "amortized_max_ms": maximum,
            "p95_below_control_period": p95 < control_period_ms,
        }
    if args.control_episodes:
        env = make_antmaze_env(backend=args.backend)
        for interval in gain_update_intervals:
            controller = GainHoldController(policy, interval)
            returns, saturations = [], []
            for episode in range(args.control_episodes):
                observation, _ = env.reset(seed=episode)
                controller.reset()
                episode_return = 0.0
                episode_saturation = []
                while True:
                    action = controller.act(
                        torch.as_tensor(observation, dtype=torch.float32, device=device)
                    )
                    observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    episode_return += reward
                    episode_saturation.append(info["action_saturation_ratio"])
                    if terminated or truncated:
                        break
                returns.append(episode_return)
                saturations.append(np.mean(episode_saturation))
            intervals[str(interval)].update(
                {
                    "control_episodes": args.control_episodes,
                    "sparse_return_mean": float(np.mean(returns)),
                    "success_rate": float(np.mean(np.asarray(returns) > 0)),
                    "saturation_rate": float(np.mean(saturations)),
                }
            )
        env.close()
    report = {
        "device": str(device),
        "batch_size": 1,
        "warmup": warmup,
        "iterations": iterations,
        "control_period_ms": control_period_ms,
        "components": components,
        "gain_update_intervals": intervals,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
