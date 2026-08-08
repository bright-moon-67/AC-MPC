"""Create a v2 causal trajectory whose actions are the controls applied by ManiSkill.

ManiSkill clips normalized actions to ``[-1, 1]`` before controller scaling.
Early v1 visual collections saved the source PPO samples even though the
resulting observations were generated with clipped actions.  This lossless
upgrade copies every observation and state into a new file, preserves the raw
policy samples as ``raw_actions``, and writes clipped controls as ``actions``.
The source is never modified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from antmaze_ac.koopman.checkpoint import sha256
from experiments.maniskill_pick_visual.collect_visual_pickcube import FORMAT_NAME


def upgrade(source_path: str | Path, output_path: str | Path) -> dict[str, object]:
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_json = output_path.with_suffix(".json")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if source_path == output_path:
        raise ValueError("Output must differ from the source")
    if output_path.exists() or output_json.exists():
        raise FileExistsError("Refusing to overwrite an existing upgraded dataset")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clipped_scalars = 0
    total_scalars = 0
    episodes = 0
    action_dim: int | None = None
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "x") as output:
        for key, value in source.attrs.items():
            output.attrs[key] = value
        output.attrs["format"] = FORMAT_NAME
        output.attrs["actions_are_applied"] = True
        output.attrs["action_semantics"] = (
            "actions are clipped to normalized environment bounds; "
            "raw_actions preserve source PPO policy outputs"
        )
        output.attrs["upgraded_from"] = str(source_path)
        for name, source_group in source.items():
            if not isinstance(source_group, h5py.Group) or not name.startswith("traj_"):
                source.copy(name, output, name=name)
                continue
            if "actions" not in source_group or "raw_actions" in source_group:
                raise ValueError(f"{name} is not an upgradeable v1 trajectory")
            raw_actions = np.asarray(source_group["actions"], dtype=np.float32)
            if raw_actions.ndim != 2 or not np.isfinite(raw_actions).all():
                raise ValueError(f"Malformed actions in {name}")
            if action_dim is None:
                action_dim = int(raw_actions.shape[1])
                output.attrs["action_low"] = -np.ones(action_dim, dtype=np.float32)
                output.attrs["action_high"] = np.ones(action_dim, dtype=np.float32)
            elif raw_actions.shape[1] != action_dim:
                raise ValueError("Action dimensions differ between episodes")
            applied_actions = np.clip(raw_actions, -1.0, 1.0).astype(
                np.float32, copy=False
            )
            target_group = output.create_group(name, track_order=True)
            for key, value in source_group.attrs.items():
                target_group.attrs[key] = value
            for dataset_name in source_group:
                if dataset_name != "actions":
                    source_group.copy(
                        dataset_name, target_group, name=dataset_name
                    )
            target_group.create_dataset("actions", data=applied_actions)
            target_group.create_dataset("raw_actions", data=raw_actions)
            clipped_scalars += int(np.count_nonzero(applied_actions != raw_actions))
            total_scalars += int(raw_actions.size)
            episodes += 1
        if episodes == 0:
            raise ValueError("Source contains no traj_i episodes")
    summary: dict[str, object] = {
        "format": FORMAT_NAME,
        "source": str(source_path),
        "output": str(output_path),
        "source_sha256": sha256(source_path),
        "output_sha256": sha256(output_path),
        "episodes": episodes,
        "action_scalars": total_scalars,
        "clipped_action_scalars": clipped_scalars,
        "clipped_fraction": clipped_scalars / total_scalars,
    }
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
