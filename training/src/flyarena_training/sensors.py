"""Port of `observeAgent` (`src/lib/arena/sensors.ts`).

`OBSERVATION_CHANNELS` order: foodBearing, foodDistance, hazardBearing,
hazardDistance, forwardClearance, leftClearance, rightClearance, speed.
"""
from __future__ import annotations

import math
from typing import Sequence

from .config import ArenaConfig
from .world import AGENT_IDS, AgentState, WorldItem

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


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _wrap_angle(angle: float) -> float:
    """Port of `sensors.ts`'s own local `wrapAngle` (lines 23-24) — a
    *different* function from `world.ts`'s `wrapAngle`
    (`world.py`'s `wrap_angle`): `sensors.ts` does not apply that function's
    `wrapped === -Math.PI ? Math.PI : wrapped` fixup. The two must stay
    separate here too, or a bearing that wraps to exactly `-pi` would return
    `+1.0` (this port, if reusing `world.py`'s version) instead of TS's
    `-1.0`."""
    two_pi = math.pi * 2
    return math.fmod(math.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi


def _nearest_egocentric(
    agent: AgentState, targets: Sequence, sensor_range: float
) -> tuple[float, float]:
    nearest: tuple[float, float] | None = None
    nearest_distance = math.inf
    ax, az = agent.position
    for target in targets:
        tx, tz = target.position
        distance = math.hypot(tx - ax, tz - az)
        if distance < nearest_distance:
            nearest = (tx, tz)
            nearest_distance = distance
    if nearest is None:
        return 0.0, 1.0
    if nearest_distance == 0:
        return 0.0, 0.0
    absolute_bearing = math.atan2(nearest[0] - ax, nearest[1] - az)
    return _wrap_angle(absolute_bearing - agent.heading) / math.pi, _clamp01(nearest_distance / sensor_range)


def _wall_clearance(agent: AgentState, relative_angle: float, config: ArenaConfig) -> float:
    angle = agent.heading + relative_angle
    dx = math.sin(angle)
    dz = math.cos(angle)
    max_x = config.half_width - agent.radius
    max_z = config.half_depth - agent.radius
    ax, az = agent.position
    x_distance = math.inf if abs(dx) < 1e-12 else ((max_x if dx > 0 else -max_x) - ax) / dx
    z_distance = math.inf if abs(dz) < 1e-12 else ((max_z if dz > 0 else -max_z) - az) / dz
    return _clamp01(max(0.0, min(x_distance, z_distance)) / config.sensor_range)


def observe_agent(item: WorldItem, agent_id: str, config: ArenaConfig) -> tuple[float, ...]:
    """Port of `observeAgent`. Returns the 8-channel observation in
    `OBSERVATION_CHANNELS` order."""
    agent = item.agents[AGENT_IDS.index(agent_id)]
    food_bearing, food_distance = _nearest_egocentric(agent, item.foods, config.sensor_range)
    hazard_bearing, hazard_distance = _nearest_egocentric(agent, item.hazards, config.sensor_range)

    return (
        food_bearing,
        food_distance,
        hazard_bearing,
        hazard_distance,
        _wall_clearance(agent, 0.0, config),
        _wall_clearance(agent, -math.pi / 2, config),
        _wall_clearance(agent, math.pi / 2, config),
        _clamp01(math.hypot(agent.velocity[0], agent.velocity[1]) / config.max_speed),
    )
