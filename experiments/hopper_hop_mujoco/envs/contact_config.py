"""MuJoCo contact-compliance configuration for the Hopper foot-ground pair.

Semantics (verified against the official MuJoCo 3.x docs — Modeling chapter,
"Solver parameters"; do NOT rely on memory for these):

solref (positive format) = (timeconst, dampratio):
    b = 2 / (d_width * timeconst)
    k = d(r) / (d_width^2 * timeconst^2 * dampratio^2)
    resting penetration (deep contact, d -> d_width) ~ au * (1-d) * timeconst^2 * dampratio^2
    * timeconst  : larger => SOFTER (violation resolved over a longer time).
    * dampratio  : 1 = critically damped; <1 bouncy; >1 overdamped.
solref (negative/direct format) = (-stiffness, -damping):
    b = damping / d_width ; k = stiffness * d(r) / d_width^2

solimp = (d0, d_width, width, midpoint, power):
    d(r) interpolates impedance d in (0,1) from d(0)=d0 to d(width)=d_width via a
    sigmoid (midpoint in [0,1] in units of width, power >= 1).
    * d0 (solimp[0]) = impedance at contact onset; lower => SOFTER onset
      (d0 = 0 gives smooth/differentiable contact-force onset).
    * d_width (solimp[1]) = impedance at full penetration; lower => SOFTER.
    * width (solimp[2]) = penetration depth over which impedance ramps;
      larger => SOFTER contact layer.
margin: geometric inflation; contacts generate force when distance < margin.
friction[0] (sliding): tangential Coulomb coefficient.

Presets are applied at runtime to the foot_heel, foot_toe AND floor geoms so
that the contact-pair mixing (same priority -> weighted average) yields exactly
the configured values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# --------------------------------------------------------------------------- #
# MuJoCo default geom contact arrays (verified by loading the model)
# --------------------------------------------------------------------------- #
MUJOCO_DEFAULT_SOLREF = (0.02, 1.0)
MUJOCO_DEFAULT_SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
MUJOCO_DEFAULT_MARGIN = 0.0
MUJOCO_DEFAULT_FRICTION = (1.0, 0.005, 0.0001)  # (sliding, torsional, rolling)

# PhysX (SAPIEN) material used by MS-HopperHop-v1 (audit 2026-08-08):
#   PhysxMaterial(static=0.3, dynamic=0.3, restitution=0.0)
PHYSX_FRICTION = 0.3


@dataclass(frozen=True)
class ContactConfig:
    """Native-MuJoCo compliant-contact parameters (no custom contact model)."""

    name: str = "custom"
    solref: tuple = MUJOCO_DEFAULT_SOLREF
    solimp: tuple = MUJOCO_DEFAULT_SOLIMP
    margin: float = MUJOCO_DEFAULT_MARGIN
    sliding_friction: float = MUJOCO_DEFAULT_FRICTION[0]
    description: str = ""

    def validate(self) -> None:
        if not len(self.solref) == 2:
            raise ValueError("solref must be (timeconst, dampratio)")
        if not len(self.solimp) == 5:
            raise ValueError("solimp must be (d0, d_width, width, midpoint, power)")
        if self.solimp[0] <= 0 or self.solimp[0] >= 1:
            raise ValueError("solimp[0] (d0) must be in (0, 1)")
        if self.solimp[1] <= 0 or self.solimp[1] >= 1:
            raise ValueError("solimp[1] (d_width) must be in (0, 1)")
        if self.solimp[2] < 0:
            raise ValueError("solimp[2] (width) must be >= 0")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        if self.sliding_friction < 0:
            raise ValueError("sliding_friction must be >= 0")

    # ------------------------------------------------------------------ #
    # Analytical predictions (standard solref format; d -> d_width).
    # Mass-normalized stiffness k [1/s^2] and effective rest penetration r.
    # ------------------------------------------------------------------ #
    @property
    def k_normalized(self) -> float:
        """Stiffness per unit effective mass, k = d/(d_width^2 * timeconst^2 * dampratio^2)."""
        tc, dr = self.solref
        dw = self.solimp[1]
        d = min(self.solimp[1], 0.9999)
        return d / (dw * dw * tc * tc * dr * dr)

    def rest_penetration(self, g: float = 9.81, d: Optional[float] = None) -> float:
        """Analytical resting penetration [m] for a contact loaded by gravity."""
        tc, dr = self.solref
        if d is None:
            d = min(self.solimp[1], 0.9999)
        return g * (1.0 - d) * tc * tc * dr * dr

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["k_normalized"] = self.k_normalized
        d["rest_penetration"] = self.rest_penetration()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContactConfig":
        keep = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in keep})

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ContactConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg


# --------------------------------------------------------------------------- #
# Presets: default / compliant / hard.
#
# Effective-stiffness ladder (analytical, ~7 kg hopper, g = 9.81):
#   mujoco_compliant : k_eff ~ 1.2 kN/m, resting penetration ~ 6.3 mm  (SOFT)
#   mujoco_default   : k_eff ~18   kN/m, resting penetration ~ 0.2 mm  (dm_control)
#   mujoco_hard      : k_eff ~280  kN/m, resting penetration ~ 0.25 um (near-rigid)
# --------------------------------------------------------------------------- #
PRESET_CONTACT_CONFIGS: Dict[str, ContactConfig] = {
    "mujoco_default": ContactConfig(
        name="mujoco_default",
        solref=MUJOCO_DEFAULT_SOLREF,
        solimp=MUJOCO_DEFAULT_SOLIMP,
        margin=MUJOCO_DEFAULT_MARGIN,
        sliding_friction=MUJOCO_DEFAULT_FRICTION[0],
        description=(
            "Native MuJoCo / dm_control Hopper contact defaults "
            "(solref=[0.02,1], solimp=[0.9,0.95,0.001,0.5,2], friction=1.0). "
            "Use this as the 'simulator-native' reference; friction differs "
            "from PhysX (0.3), so it is NOT the primary hard-vs-compliant "
            "comparison pair."
        ),
    ),
    "mujoco_compliant": ContactConfig(
        name="mujoco_compliant",
        solref=(0.08, 1.0),
        solimp=(0.2, 0.9, 0.01, 0.5, 2.0),
        margin=0.01,
        sliding_friction=PHYSX_FRICTION,
        description=(
            "Medium compliant contact for the primary comparison with "
            "PhysX-hard: timeconst 0.08 s (soft), d0=0.2 (soft onset), "
            "width=10 mm (gradual impedance ramp), margin=10 mm (force from a "
            "distance), sliding friction 0.3 matched to the PhysX material. "
            "Soft enough to show a finite compression process, hard enough to "
            "keep the hopper hopping."
        ),
    ),
    "mujoco_hard": ContactConfig(
        name="mujoco_hard",
        solref=(0.005, 1.0),
        solimp=(0.999, 0.999, 0.0001, 0.5, 2.0),
        margin=0.0,
        sliding_friction=PHYSX_FRICTION,
        description=(
            "Much-harder-than-default MuJoCo contact used as the scientific "
            "control: friction 0.3 matched to PhysX, timeconst 5 ms, "
            "d0=d_width=0.999 (near-rigid onset), width 0.1 mm. This isolates "
            "'simulator difference' from 'compliance difference' when compared "
            "against mujoco_compliant / physx_hard."
        ),
    ),
}


def load_contact_config(
    spec: Union[str, ContactConfig, Path, Dict[str, Any]]
) -> ContactConfig:
    """Resolve a preset name, a dict, a yaml path, or an already-built config."""
    if isinstance(spec, ContactConfig):
        return spec
    if isinstance(spec, dict):
        cfg = ContactConfig.from_dict(spec)
        cfg.validate()
        return cfg
    if isinstance(spec, Path) or (isinstance(spec, str) and spec.endswith(".yaml")):
        return ContactConfig.from_yaml(spec)
    if isinstance(spec, str):
        if spec not in PRESET_CONTACT_CONFIGS:
            raise KeyError(
                f"Unknown contact preset {spec!r}; available: "
                f"{sorted(PRESET_CONTACT_CONFIGS)}"
            )
        return PRESET_CONTACT_CONFIGS[spec]
    raise TypeError(f"Cannot resolve contact config from {type(spec)}")
