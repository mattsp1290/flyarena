"""Dedicated unit test for the `-pi` boundary that motivates keeping
`world.wrap_angle` and `sensors._wrap_angle` as two separate functions
(thermo-maintainability review, suggestion S1): `world.ts`'s `wrapAngle`
applies a `wrapped === -Math.PI ? Math.PI : wrapped` fixup that
`sensors.ts`'s own local `wrapAngle` does not, so an input that wraps to
exactly `-pi` diverges between the two. Without a test pinning this exact
input, a future "simplification" that merges the two functions would pass
every other existing test and only diverge from TypeScript at this one
boundary."""
from __future__ import annotations

import math

from flyarena_training.sensors import _wrap_angle, _wrap_angle_batched
from flyarena_training.world import wrap_angle, wrap_angle_batched


def test_wrap_angle_pi_boundary_diverges_from_sensors_wrap_angle():
    boundary = -math.pi
    assert wrap_angle(boundary) == math.pi
    assert _wrap_angle(boundary) == -math.pi


def test_wrap_angle_batched_pi_boundary_matches_scalar_variants():
    import torch

    boundary = torch.tensor([-math.pi], dtype=torch.float64)
    assert wrap_angle_batched(boundary).item() == math.pi
    assert _wrap_angle_batched(boundary).item() == -math.pi
