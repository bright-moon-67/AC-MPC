"""Narrow compatibility helpers for the pinned Playground/Brax stack."""

from __future__ import annotations

from typing import Any


def install_single_device_brax_compatibility() -> bool:
    """Restore the removed replication helper used by Brax 0.14.2.

    JAX 0.11 removed ``device_put_replicated`` before Brax 0.14.2 stopped
    calling it.  Our runner intentionally uses exactly one A100, for which
    the old operation is an explicit leading pmap axis followed by device
    placement.  Returns whether the shim was installed.
    """

    import jax
    import jax.numpy as jp

    if hasattr(jax, "device_put_replicated"):
        return False

    def device_put_replicated(tree: Any, devices: list[Any]) -> Any:
        if len(devices) != 1:
            raise RuntimeError(
                "The local Brax/JAX compatibility path supports exactly one device"
            )
        return jax.tree.map(
            lambda leaf: jax.device_put(jp.expand_dims(leaf, axis=0), devices[0]),
            tree,
        )

    jax.device_put_replicated = device_put_replicated  # type: ignore[attr-defined]
    return True
