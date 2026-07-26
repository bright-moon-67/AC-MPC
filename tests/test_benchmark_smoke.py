import json
import subprocess
import sys

import numpy as np
import torch

from antmaze_ac.koopman.checkpoint import save_checkpoint
from antmaze_ac.koopman.model import DeepKoopman


def test_benchmark_handles_float64_dare_feedback(tmp_path):
    state_dim, action_dim = 5, 2
    model = DeepKoopman(state_dim, action_dim, lift_dim=3, hidden_dims=(8,))
    with torch.no_grad():
        model.A.copy_(torch.eye(model.lifted_dim) * 0.7)
        model.B.fill_(0.1)
    config = {
        "actor": {
            "hidden_dims": [8],
            "activation": "gelu",
            "critic_hidden_dims": [8],
            "critic_activation": "gelu",
            "log_std_init": 0.0,
        },
        "control": {
            "stage_cost_epsilon": 1e-4,
            "q_max": 2.0,
            "p_max": 1.0,
            "previous_action_cost_scale": 0.001,
            "delta_action_cost_scale": 0.001,
            "dare_tolerance": 1e-8,
            "dare_max_iterations": 100,
            "dare_jitter": 1e-9,
            "dare_fail_on_nonconvergence": True,
        },
        "evaluation": {
            "control_period_ms": 50.0,
            "benchmark_iterations": 2,
            "benchmark_warmup": 1,
            "gain_update_intervals": [1, 2],
        },
    }
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        epoch=0,
        best_validation=1.0,
        config=config,
        normalizers={
            "state": {
                "mean": np.zeros(state_dim).tolist(),
                "std": np.ones(state_dim).tolist(),
            },
            "delta_action": "physical_units",
        },
        elapsed_seconds=0.1,
    )
    output = tmp_path / "benchmark.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_inference.py",
            "--koopman-checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
            "--iterations",
            "2",
            "--warmup",
            "1",
            "--control-episodes",
            "0",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["components"]["feedback"]["mean_ms"] >= 0
    assert set(report["gain_update_intervals"]) == {"1", "2"}
