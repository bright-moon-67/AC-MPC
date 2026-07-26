import json
import subprocess
import sys

import numpy as np
import torch

from antmaze_ac.data.build_sequences import AugmentedDataset
from antmaze_ac.koopman.checkpoint import save_checkpoint
from antmaze_ac.koopman.model import DeepKoopman


def test_evaluation_script_smoke(tmp_path):
    state_dim, action_dim, rows = 5, 2, 32
    rng = np.random.default_rng(3)
    x = rng.normal(size=(rows, state_dim)).astype(np.float32)
    x[0, -action_dim:] = 0.0
    delta = rng.normal(scale=0.1, size=(rows, action_dim)).astype(np.float32)
    action = x[:, -action_dim:] + delta
    next_x = np.roll(x, -1, axis=0)
    next_x[:, -action_dim:] = action
    dataset = AugmentedDataset(
        state=x,
        action=delta,
        next_state=next_x,
        reward=np.zeros(rows, dtype=np.float32),
        done=np.zeros(rows, dtype=bool),
        terminal=np.zeros(rows, dtype=bool),
        timeout=np.zeros(rows, dtype=bool),
        episode_id=np.zeros(rows, dtype=np.int64),
        step_index=np.arange(rows, dtype=np.int64),
        current_action=action,
    )
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savez_compressed(
        data_root / "test.npz",
        **dataset.as_dict(),
    )

    model = DeepKoopman(state_dim, action_dim, lift_dim=3, hidden_dims=(8,))
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(
        checkpoint,
        model,
        optimizer=optimizer,
        epoch=0,
        best_validation=1.0,
        config={"koopman": {"eval_batch_size": 4}},
        normalizers={
            "state": {"mean": np.zeros(state_dim).tolist(), "std": np.ones(state_dim).tolist()},
            "delta_action": "physical_units",
        },
        elapsed_seconds=0.1,
    )
    output = tmp_path / "evaluation.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_koopman.py",
            "--checkpoint",
            str(checkpoint),
            "--data",
            str(data_root),
            "--horizons",
            "1,5,10,20,25",
            "--device",
            "cpu",
            "--output",
            str(output),
            "--no-save-curves",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report["horizons"]) == {"1", "5", "10", "20", "25"}
    assert report["batch_size"] == 4
    assert report["horizons"]["25"]["samples"] == rows - 25 + 1
