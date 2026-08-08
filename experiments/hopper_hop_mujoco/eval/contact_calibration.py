"""MuJoCo compliant-contact calibration (drop test).

Replicates the 2026-08-08 PhysX audit protocol so the two simulators are
compared on identical terms: a foot-like capsule (r = 0.04 m, half-length
0.095 m, ~1.22 kg, density 1000 kg/m^3) is dropped from 0.5 m onto a plane.

PhysX-hard baseline (measured, audit 2026-08-08, 4 pos iter / 1 vel iter,
dt = 0.01 s): impact ~2.84 m/s, peak penetration ~0.0 m, peak force ~223 N,
single-step arrest (near-rigid).

For every contact config this records:
    impact_velocity, max_penetration, contact_duration_s, peak_normal_force,
    impulse_Ns, settling_penetration, rebound_velocity, n_steps_to_peak_force

Outputs (unless --no-save):
    runs/hopper_hop_mujoco/contact_calibration/contact_calibration.json
    runs/hopper_hop_mujoco/contact_calibration/contact_calibration.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import mujoco

from experiments.hopper_hop_mujoco.envs.contact_config import (
    PRESET_CONTACT_CONFIGS,
    ContactConfig,
    load_contact_config,
)

# --------------------------------------------------------------------------- #
# In-memory foot drop-test model (identical geometry/mass to the PhysX audit)
# --------------------------------------------------------------------------- #
DROP_XML = """
<mujoco model="foot_contact_calibration">
  <option timestep="0.005" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 1" pos="0 0 0"/>
    <body name="foot" pos="0 0 0.5">
      <freejoint/>
      <geom name="foot_capsule" type="capsule" fromto="-0.095 0 0 0.095 0 0"
            size="0.04" density="1000"/>
    </body>
  </worldbody>
</mujoco>
"""

FOOT_RADIUS = 0.04  # capsule radius (bottom of capsule = body z - radius)
DROP_HEIGHT = 0.5  # m
G = 9.81


@dataclass
class CalibrationResult:
    config: str
    physics_dt: float
    foot_mass: float
    impact_velocity: float
    max_penetration: float
    rest_gap: float
    rest_force: float
    peak_normal_force: float
    impulse_Ns: float
    impact_duration_s: float
    rebound_velocity: float
    n_steps_to_peak_force: int


def run_drop(
    contact: ContactConfig,
    physics_dt: float = 0.005,
    drop_height: float = DROP_HEIGHT,
    horizon_s: float = 2.0,
) -> CalibrationResult:
    """Drop the foot capsule with the given contact config; measure metrics."""
    model = mujoco.MjModel.from_xml_string(DROP_XML)
    model.opt.timestep = physics_dt
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    floor_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    foot_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "foot_capsule")
    for gid in (floor_gid, foot_gid):
        model.geom_solref[gid] = list(contact.solref)
        model.geom_solimp[gid] = list(contact.solimp)
        model.geom_margin[gid] = contact.margin
        model.geom_friction[gid] = (contact.sliding_friction, 0.005, 0.0001)

    data = mujoco.MjData(model)
    foot_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    data.qpos[2] = drop_height  # z of freejoint (x=0,y=0,z=0.5, quat 1)
    mujoco.mj_forward(model, data)

    def _normal_force() -> float:
        # elliptic cone: normal row is efc_force[efc_address]; cfrc_ext is not
        # populated without an acc-stage sensor in MuJoCo 3.x.
        return float(
            sum(
                data.efc_force[c.efc_address]
                for c in data.contact
                if c.efc_address >= 0
            )
        )

    n_steps = int(horizon_s / physics_dt)
    dt = physics_dt
    z_bottom = np.empty(n_steps)
    forces = np.empty(n_steps)
    vels = np.empty(n_steps)
    prev_z = None
    for i in range(n_steps):
        if prev_z is not None:
            vels[i - 1] = (data.xpos[foot_body, 2] - prev_z) / dt
        prev_z = data.xpos[foot_body, 2]
        mujoco.mj_step(model, data)
        z = data.xpos[foot_body, 2]
        z_bottom[i] = z - FOOT_RADIUS
        forces[i] = _normal_force()
    vels[n_steps - 1] = vels[n_steps - 2] if n_steps > 1 else 0.0

    pen = np.maximum(-z_bottom, 0.0)
    in_contact = forces > 1e-3
    idx = np.flatnonzero(in_contact)
    if len(idx) == 0:
        raise RuntimeError(f"no contact registered for config {contact.name}")

    i0 = int(idx[0])
    impact_v = float(abs(vels[max(i0 - 1, 0)]))

    # impact window: first contact until the force settles near the weight
    weight = float(np.sum(model.body_mass[1:]) * G)
    settled = np.flatnonzero(
        (forces > 0.9 * weight) & (forces < 1.1 * weight)
    )
    i_end = int(settled[0]) if len(settled) else int(idx[-1])
    i_end = max(i_end, i0 + 1)

    peak_force = float(forces[idx].max())
    # impact impulse: integrate only over the impact window [i0, i_end],
    # NOT over the whole in-contact span (which would include the settled
    # weight held for the rest of the horizon and inflate the impulse).
    impulse = float(forces[i0 : i_end + 1].sum() * dt)
    max_pen = float(pen[i0 : i_end + 1].max()) if i_end > i0 else float(pen.max())
    # settled tail = last 1.0 s of the run (impact+settling happen < 0.5 s)
    tail = slice(max(0, n_steps - int(1.0 / dt)), n_steps)
    rest_gap = float(np.mean(z_bottom[tail]))
    rest_force = float(np.mean(forces[tail]))
    rebound = float(max(0.0, vels[i0 : i0 + int(0.5 / dt)].max()))
    apex = i0 + int(np.argmax(forces[i0:]))

    return CalibrationResult(
        config=contact.name,
        physics_dt=physics_dt,
        foot_mass=float(np.sum(model.body_mass[1:])),
        impact_velocity=impact_v,
        max_penetration=max_pen,
        rest_gap=rest_gap,
        rest_force=rest_force,
        peak_normal_force=peak_force,
        impulse_Ns=impulse,
        impact_duration_s=(i_end - i0) * dt,
        rebound_velocity=rebound,
        n_steps_to_peak_force=int(apex - i0),
    )


def run_sweep(
    configs: Sequence[str],
    physics_dts: Sequence[float],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    rows: List[CalibrationResult] = []
    for name in configs:
        contact = load_contact_config(name)
        for dt in physics_dts:
            row = run_drop(contact, physics_dt=dt)
            rows.append(row)
            print(
                f"[{name:<18} dt={dt:.4f}] impact={row.impact_velocity:5.2f} m/s  "
                f"maxPen={row.max_penetration:8.5f} m  "
                f"restGap={row.rest_gap:8.5f} m  restF={row.rest_force:5.1f} N  "
                f"peakF={row.peak_normal_force:7.1f} N  "
                f"imp={row.impulse_Ns:6.1f} Ns  "
                f"impDur={row.impact_duration_s:6.3f} s  "
                f"rebound={row.rebound_velocity:5.2f} m/s",
                flush=True,
            )

    report: Dict[str, Any] = {
        "protocol": {
            "drop_height_m": DROP_HEIGHT,
            "foot_radius_m": FOOT_RADIUS,
            "foot_half_length_m": 0.095,
            "foot_density_kgm3": 1000.0,
            "g_m_s2": G,
            "physics": "mujoco " + mujoco.__version__,
            "reference_physx_hard": {
                "impact_velocity": 2.84,
                "max_penetration": 0.0,
                "settling_penetration": 0.0,
                "peak_force": 223.5,
                "note": "audit 2026-08-08, 4 pos iter / 1 vel iter, dt=0.01",
            },
        },
        "analytical": {
            name: {
                "k_normalized": load_contact_config(name).k_normalized,
                "rest_penetration": load_contact_config(name).rest_penetration(),
                **load_contact_config(name).to_dict(),
            }
            for name in configs
        },
        "results": [asdict(r) for r in rows],
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "contact_calibration.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
        with open(output_dir / "contact_calibration.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(asdict(r))
        print(f"\nwrote: {output_dir / 'contact_calibration.json'}")
        print(f"wrote: {output_dir / 'contact_calibration.csv'}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        default="mujoco_default,mujoco_compliant,mujoco_hard",
        help="comma-separated contact preset names",
    )
    parser.add_argument(
        "--physics-dts",
        default="0.005,0.0025",
        help="comma-separated MuJoCo physics timesteps (s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/hopper_hop_mujoco/contact_calibration"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sweep(
        [c.strip() for c in args.configs.split(",") if c.strip()],
        [float(x) for x in args.physics_dts.split(",") if x.strip()],
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
