"""Batched port of `observeAgent` (`src/lib/arena/sensors.ts`).

`OBSERVATION_CHANNELS` order: foodBearing, foodDistance, hazardBearing,
hazardDistance, forwardClearance, leftClearance, rightClearance, speed.

`observe_batch` is the production entry point: dense `[B, 8]` tensor ops
over a whole `WorldBatch` at once (see `world.py`'s module doc on batching).
The per-item reference implementation this was ported from now lives at
`training/tests/reference_world.py`, used only as a test oracle.
"""
from __future__ import annotations

import math

import torch

from .config import ArenaConfig
from .world import AGENT_IDS, WorldBatch

OBSERVATION_CHANNELS: tuple[str, ...] = (
    "foodBearing",
    "foodDistance",
    "hazardBearing",
    "hazardDistance",
    "forwardClearance",
    "leftClearance",
    "rightClearance",
    "speed",
)


def _wrap_angle(angle: float) -> float:
    """Port of `sensors.ts`'s own local `wrapAngle` (lines 23-24) — a
    *different* function from `world.ts`'s `wrapAngle`
    (`world.py`'s `wrap_angle`): `sensors.ts` does not apply that function's
    `wrapped === -Math.PI ? Math.PI : wrapped` fixup. The two must stay
    separate here too, or a bearing that wraps to exactly `-pi` would return
    `+1.0` (this port, if reusing `world.py`'s version) instead of TS's
    `-1.0`. Kept as a scalar function (unlike the rest of this module) so
    `tests/test_wrap_angle.py` can pin the exact boundary input against
    `world.wrap_angle` without needing a tensor."""
    two_pi = math.pi * 2
    return math.fmod(math.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi


def _wrap_angle_batched(angle: torch.Tensor) -> torch.Tensor:
    """Batched port of `_wrap_angle` above (no `-pi` fixup)."""
    two_pi = math.pi * 2
    return torch.fmod(torch.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi


def _clamp01_batch(value: torch.Tensor) -> torch.Tensor:
    return value.clamp(0.0, 1.0)


def _nearest_egocentric_batch(
    agent_position: torch.Tensor, agent_heading: torch.Tensor, targets: torch.Tensor, sensor_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """`agent_position`: `[B, 2]`, `agent_heading`: `[B]`, `targets`: `[B, T, 2]`.
    Returns `(bearing, distance)`, each `[B]`. Ties (equal minimum distance)
    keep the lowest target index, matching the per-item loop's strict `<`
    (first-seen minimum wins) — `torch.min`'s documented first-occurrence
    tie-break matches that exactly."""
    batch_size, target_count, _ = targets.shape
    if target_count == 0:
        zeros = torch.zeros(batch_size, dtype=agent_position.dtype, device=agent_position.device)
        return zeros, torch.ones_like(zeros)

    diff = targets - agent_position.unsqueeze(1)  # [B, T, 2]
    dist = torch.linalg.norm(diff, dim=-1)  # [B, T]
    nearest_distance, nearest_index = dist.min(dim=1)  # [B]
    nearest = targets.gather(1, nearest_index.view(-1, 1, 1).expand(-1, 1, 2)).squeeze(1)  # [B, 2]

    absolute_bearing = torch.atan2(nearest[:, 0] - agent_position[:, 0], nearest[:, 1] - agent_position[:, 1])
    bearing = _wrap_angle_batched(absolute_bearing - agent_heading) / math.pi
    distance = _clamp01_batch(nearest_distance / sensor_range)

    zero_distance = nearest_distance == 0
    bearing = torch.where(zero_distance, torch.zeros_like(bearing), bearing)
    distance = torch.where(zero_distance, torch.zeros_like(distance), distance)
    return bearing, distance


def _wall_clearance_batch(
    agent_position: torch.Tensor, agent_heading: torch.Tensor, relative_angle: float, config: ArenaConfig
) -> torch.Tensor:
    angle = agent_heading + relative_angle
    dx = torch.sin(angle)
    dz = torch.cos(angle)
    max_x = config.half_width - config.agent_radius
    max_z = config.half_depth - config.agent_radius
    ax, az = agent_position[:, 0], agent_position[:, 1]

    small = 1e-12
    x_target = torch.where(dx > 0, torch.full_like(dx, max_x), torch.full_like(dx, -max_x))
    x_distance = torch.where(dx.abs() < small, torch.full_like(dx, math.inf), (x_target - ax) / dx)
    z_target = torch.where(dz > 0, torch.full_like(dz, max_z), torch.full_like(dz, -max_z))
    z_distance = torch.where(dz.abs() < small, torch.full_like(dz, math.inf), (z_target - az) / dz)

    return _clamp01_batch(torch.minimum(x_distance, z_distance).clamp(min=0.0) / config.sensor_range)


def observe_batch(state: WorldBatch, agent_id: str, config: ArenaConfig) -> torch.Tensor:
    """Port of `observeAgent`, batched. Returns `[B, 8]` (float64) in
    `OBSERVATION_CHANNELS` order for every item in `state`."""
    agent_index = AGENT_IDS.index(agent_id)
    agent_position = state.agent_position[:, agent_index, :]  # [B, 2]
    agent_heading = state.agent_heading[:, agent_index]  # [B]
    agent_velocity = state.agent_velocity[:, agent_index, :]  # [B, 2]

    food_bearing, food_distance = _nearest_egocentric_batch(
        agent_position, agent_heading, state.food_position, config.sensor_range
    )
    hazard_bearing, hazard_distance = _nearest_egocentric_batch(
        agent_position, agent_heading, state.hazard_position, config.sensor_range
    )

    forward_clearance = _wall_clearance_batch(agent_position, agent_heading, 0.0, config)
    left_clearance = _wall_clearance_batch(agent_position, agent_heading, -math.pi / 2, config)
    right_clearance = _wall_clearance_batch(agent_position, agent_heading, math.pi / 2, config)
    speed = _clamp01_batch(torch.hypot(agent_velocity[:, 0], agent_velocity[:, 1]) / config.max_speed)

    return torch.stack(
        [food_bearing, food_distance, hazard_bearing, hazard_distance, forward_clearance, left_clearance, right_clearance, speed],
        dim=1,
    )
