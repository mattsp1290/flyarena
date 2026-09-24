"""Per-item reference oracle for `flyarena_training.world.step_world_batched`
and `flyarena_training.sensors.observe_batch`.

This is a straight-line, unbatched port of `stepWorld`/`observeAgent`
(`src/lib/arena/world.ts`, `src/lib/arena/sensors.ts`): one Python `for` loop
over plain dataclasses, operation-for-operation identical to what
`flyarena_training.world`/`flyarena_training.sensors` looked like before the
CRITICAL batching fix (thermo-architecture review, finding #1). It is
**test-only** — not part of the `flyarena_training` package, not imported by
any production code path — kept solely so `test_batched_matches_reference.py`
can cross-check the dense `torch` implementation against an independent,
easy-to-eyeball-correct implementation across many random seeds/ticks/
actions, which is a stronger regression backstop than the golden-trace
fixtures alone (those cover a fixed, small set of seeds/ticks).

Not collected by pytest (module name doesn't match `test_*`/`*_test`).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

from flyarena_training.actions import decode_action
from flyarena_training.config import ARENA_CONFIG, ArenaConfig
from flyarena_training.world import (
    AGENT_IDS,
    AgentScore,
    AgentState,
    FoodState,
    HazardState,
    WorldItem,
    _overlaps,
    place_without_overlap,
    wrap_angle,
)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _clamp_agent_to_arena(agent: AgentState, config: ArenaConfig) -> None:
    max_x = config.half_width - agent.radius
    max_z = config.half_depth - agent.radius
    x, z = agent.position
    vx, vz = agent.velocity
    if x < -max_x or x > max_x:
        x = _clamp(x, -max_x, max_x)
        vx = 0.0
    if z < -max_z or z > max_z:
        z = _clamp(z, -max_z, max_z)
        vz = 0.0
    agent.position = (x, z)
    agent.velocity = (vx, vz)


def _move_hazards(hazards: list[HazardState], config: ArenaConfig) -> None:
    dt = config.fixed_delta_seconds
    for hazard in hazards:
        hazard.previous_position = hazard.position
        x = hazard.position[0] + hazard.velocity[0] * dt
        z = hazard.position[1] + hazard.velocity[1] * dt
        vx, vz = hazard.velocity
        max_x = config.half_width - hazard.radius
        max_z = config.half_depth - hazard.radius
        if x < -max_x or x > max_x:
            x = _clamp(x, -max_x, max_x)
            vx = -vx
        if z < -max_z or z > max_z:
            z = _clamp(z, -max_z, max_z)
            vz = -vz
        hazard.position = (x, z)
        hazard.velocity = (vx, vz)


def _process_contacts(item: WorldItem, config: ArenaConfig) -> None:
    for food in item.foods:
        collector: AgentState | None = None
        for agent in item.agents:
            if _overlaps(agent.position[0], agent.position[1], agent.radius, food.position[0], food.position[1], food.radius):
                collector = agent
                break
        if collector is not None:
            collector.score.food_pickups += 1
            collector.score.movement_score += config.food_score
            blockers: list[tuple[float, float, float]] = [
                (a.position[0], a.position[1], a.radius) for a in item.agents
            ]
            blockers += [(f.position[0], f.position[1], f.radius) for f in item.foods if f is not food]
            blockers += [(h.position[0], h.position[1], h.radius) for h in item.hazards]
            item.rng_state, position = place_without_overlap(item.rng_state, food.radius, config, blockers)
            food.position = position
            food.respawns += 1

    for agent in item.agents:
        active = [
            _overlaps(
                agent.position[0], agent.position[1], agent.radius, hazard.position[0], hazard.position[1], hazard.radius
            )
            for hazard in item.hazards
        ]
        for index, is_active in enumerate(active):
            if is_active and not agent.active_hazard[index]:
                agent.score.hazard_contacts += 1
                agent.score.movement_score -= config.hazard_penalty
        agent.active_hazard = active


def step_world_item_reference(
    item: WorldItem,
    actions: Mapping[str, Sequence[float] | None],
    config: ArenaConfig = ARENA_CONFIG,
) -> WorldItem:
    """Reference (unbatched) port of `stepWorld`, one fixed 30 Hz tick. The
    input is not mutated."""
    dt = config.fixed_delta_seconds
    new_agents: list[AgentState] = []
    for agent_id, agent in zip(AGENT_IDS, item.agents):
        thrust, yaw, brake = decode_action(actions.get(agent_id))
        new_agent = AgentState(
            position=agent.position,
            previous_position=agent.position,
            velocity=agent.velocity,
            heading=agent.heading,
            previous_heading=agent.heading,
            radius=agent.radius,
            active_hazard=list(agent.active_hazard),
            score=AgentScore(
                food_pickups=agent.score.food_pickups,
                hazard_contacts=agent.score.hazard_contacts,
                distance_travelled=agent.score.distance_travelled,
                movement_score=agent.score.movement_score,
            ),
        )
        new_agent.heading = wrap_angle(new_agent.heading + yaw * config.turn_rate * dt)
        vx, vz = new_agent.velocity
        vx += math.sin(new_agent.heading) * thrust * config.acceleration * dt
        vz += math.cos(new_agent.heading) * thrust * config.acceleration * dt
        drag = max(0.0, 1 - (config.rolling_drag + brake * config.brake_drag) * dt)
        vx *= drag
        vz *= drag
        speed = math.hypot(vx, vz)
        if speed > config.max_speed:
            scale = config.max_speed / speed
            vx *= scale
            vz *= scale
        new_agent.velocity = (vx, vz)
        px, pz = new_agent.position
        px += vx * dt
        pz += vz * dt
        new_agent.position = (px, pz)
        _clamp_agent_to_arena(new_agent, config)
        distance = math.hypot(
            new_agent.position[0] - new_agent.previous_position[0],
            new_agent.position[1] - new_agent.previous_position[1],
        )
        new_agent.score.distance_travelled += distance
        new_agent.score.movement_score += distance * config.movement_score_per_unit
        new_agents.append(new_agent)

    new_foods = [FoodState(position=f.position, radius=f.radius, respawns=f.respawns) for f in item.foods]
    new_hazards = [
        HazardState(position=h.position, previous_position=h.previous_position, velocity=h.velocity, radius=h.radius)
        for h in item.hazards
    ]
    new_item = WorldItem(
        seed=item.seed,
        rng_state=item.rng_state,
        tick=item.tick,
        time_seconds=item.time_seconds,
        agents=new_agents,
        foods=new_foods,
        hazards=new_hazards,
    )

    _move_hazards(new_item.hazards, config)
    _process_contacts(new_item, config)
    new_item.tick += 1
    new_item.time_seconds = new_item.tick * dt
    return new_item


def step_world_reference(
    items: Sequence[WorldItem],
    actions: Sequence[Mapping[str, Sequence[float] | None]],
    config: ArenaConfig = ARENA_CONFIG,
) -> list[WorldItem]:
    """Reference (unbatched) port of a batched `stepWorld` call: independently
    steps every item with its own action mapping."""
    return [step_world_item_reference(item, action, config) for item, action in zip(items, actions)]


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _wrap_angle_sensor(angle: float) -> float:
    two_pi = math.pi * 2
    return math.fmod(math.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi


def _nearest_egocentric(agent: AgentState, targets: Sequence, sensor_range: float) -> tuple[float, float]:
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
    return _wrap_angle_sensor(absolute_bearing - agent.heading) / math.pi, _clamp01(nearest_distance / sensor_range)


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


def observe_agent_reference(item: WorldItem, agent_id: str, config: ArenaConfig) -> tuple[float, ...]:
    """Reference (unbatched) port of `observeAgent`. Returns the 8-channel
    observation in `flyarena_training.sensors.OBSERVATION_CHANNELS` order."""
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
