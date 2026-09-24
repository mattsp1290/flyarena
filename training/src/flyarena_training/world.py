"""Port of `src/lib/arena/world.ts`: `createWorld`, `stepWorld`, and the
`nextRandomState`-driven rejection-sampling placement they share.

Design note on "batched": a `WorldItem` is one simulation instance; `create_world`/
`step_world` operate over a *list* of `B` independent items rather than a
dense `[B, ...]` tensor. This is deliberate, not a shortcut: `placeWithoutOverlap`
is sequential rejection sampling whose attempt count is data-dependent (it
depends on where blockers already are), and it is not just used at reset —
`processContacts` also calls it, per item, whenever that item's agent eats a
food on that tick. The plan
(`.agents/plans/trained-readout/02-gpu-port-and-parity.md`) explicitly
directs reset placement to run "on CPU per seed"; food-respawn placement
during `stepWorld` is the exact same function for the exact same reason, so
it gets the same treatment. Only the neural rate model (`model.py`) is dense
GPU-batched: that is where this port's throughput requirement
(`test_gpu.py`, steps/second at B = 4096) actually applies.

Every arithmetic op below is plain Python `float`/`int`, which is IEEE-754
binary64 exactly like a JS `number` — porting each TS line in the same
operation order reproduces TS's result to within a handful of ULPs (the
`%` operator is the one place this needs care: JS `%` matches C `fmod`
(sign of the dividend), not Python's `%` (sign of the divisor), so
`wrap_angle` below uses `math.fmod`, not `%`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .actions import decode_action
from .config import ARENA_CONFIG, ArenaConfig
from .rng import next_random_state, normalize_seed, random_unit

# Fixed agent order, matching `createWorld`'s `[createAgent('left', ...),
# createAgent('right', ...)]` construction order in world.ts.
AGENT_IDS: tuple[str, str] = ("left", "right")


def wrap_angle(angle: float) -> float:
    """Port of `wrapAngle` (`src/lib/arena/world.ts`). Note: `sensors.ts`
    defines its own *different* local `wrapAngle` with no `-Math.PI`
    fixup — `sensors.py`'s `_wrap_angle` ports that one separately; do not
    reuse this function there."""
    two_pi = math.pi * 2
    wrapped = math.fmod(math.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def _overlaps(ax: float, az: float, a_radius: float, bx: float, bz: float, b_radius: float) -> bool:
    return math.hypot(ax - bx, az - bz) <= a_radius + b_radius


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return minimum if value < minimum else maximum if value > maximum else value


def _random_position(rng_state: int, radius: float, config: ArenaConfig) -> tuple[int, tuple[float, float]]:
    rng_state, x_unit = random_unit(rng_state)
    rng_state, z_unit = random_unit(rng_state)
    inset = config.spawn_inset + radius
    x = -config.half_width + inset + x_unit * (2 * (config.half_width - inset))
    z = -config.half_depth + inset + z_unit * (2 * (config.half_depth - inset))
    return rng_state, (x, z)


def _position_is_clear(
    position: tuple[float, float], radius: float, blockers: Sequence[tuple[float, float, float]]
) -> bool:
    px, pz = position
    return all(not _overlaps(px, pz, radius, bx, bz, br) for (bx, bz, br) in blockers)


def place_without_overlap(
    rng_state: int, radius: float, config: ArenaConfig, blockers: Sequence[tuple[float, float, float]]
) -> tuple[int, tuple[float, float]]:
    """Port of `placeWithoutOverlap`. `blockers` is `(x, z, radius)` triples."""
    for _ in range(64):
        rng_state, candidate = _random_position(rng_state, radius, config)
        if _position_is_clear(candidate, radius, blockers):
            return rng_state, candidate

    # Bounded, runtime-independent lattice fallback (matches world.ts exactly:
    # divisions = 64, inclusive of both endpoints, z outer / x inner).
    inset = config.spawn_inset + radius
    min_x = -config.half_width + inset
    min_z = -config.half_depth + inset
    width = 2 * (config.half_width - inset)
    depth = 2 * (config.half_depth - inset)
    divisions = 64
    for z_index in range(divisions + 1):
        for x_index in range(divisions + 1):
            candidate = (min_x + (width * x_index) / divisions, min_z + (depth * z_index) / divisions)
            if _position_is_clear(candidate, radius, blockers):
                return rng_state, candidate
    raise ValueError("Invalid arena config: bounded placement could not find non-overlapping space")


@dataclass
class AgentScore:
    food_pickups: int = 0
    hazard_contacts: int = 0
    distance_travelled: float = 0.0
    movement_score: float = 0.0


@dataclass
class AgentState:
    position: tuple[float, float]
    previous_position: tuple[float, float]
    velocity: tuple[float, float]
    heading: float
    previous_heading: float
    radius: float
    # Boolean-by-hazard-index in place of TS's `activeHazardIds: string[]`:
    # hazard ids ("hazard-0", "hazard-1", ...) are a fixed 1:1 bijection with
    # index for the lifetime of a run (hazards never respawn or reorder,
    # only move — see `export-traces.ts`'s `SerializedWorld` doc comment),
    # so tracking "is hazard i currently active" by index is exactly
    # equivalent to TS's id-keyed `Set`, without needing string ids at all.
    active_hazard: list[bool]
    score: AgentScore


@dataclass
class FoodState:
    position: tuple[float, float]
    radius: float
    respawns: int = 0


@dataclass
class HazardState:
    position: tuple[float, float]
    previous_position: tuple[float, float]
    velocity: tuple[float, float]
    radius: float


@dataclass
class WorldItem:
    """One simulation instance. Mirrors `WorldState` (`src/lib/arena/types.ts`)."""

    seed: int
    rng_state: int
    tick: int
    time_seconds: float
    agents: list[AgentState] = field(default_factory=list)  # index 0 = left, 1 = right
    foods: list[FoodState] = field(default_factory=list)
    hazards: list[HazardState] = field(default_factory=list)


def create_world_item(seed: int, config: ArenaConfig = ARENA_CONFIG) -> WorldItem:
    """Port of `createWorld`."""
    normalized_seed = normalize_seed(seed)
    rng_state = normalized_seed

    agents = [
        AgentState(
            position=(-config.half_width / 4, 0.0),
            previous_position=(-config.half_width / 4, 0.0),
            velocity=(0.0, 0.0),
            heading=math.pi / 2,
            previous_heading=math.pi / 2,
            radius=config.agent_radius,
            active_hazard=[False] * config.hazard_count,
            score=AgentScore(),
        ),
        AgentState(
            position=(config.half_width / 4, 0.0),
            previous_position=(config.half_width / 4, 0.0),
            velocity=(0.0, 0.0),
            heading=-math.pi / 2,
            previous_heading=-math.pi / 2,
            radius=config.agent_radius,
            active_hazard=[False] * config.hazard_count,
            score=AgentScore(),
        ),
    ]
    blockers: list[tuple[float, float, float]] = [(a.position[0], a.position[1], a.radius) for a in agents]

    foods: list[FoodState] = []
    for _ in range(config.food_count):
        rng_state, position = place_without_overlap(rng_state, config.food_radius, config, blockers)
        foods.append(FoodState(position=position, radius=config.food_radius))
        blockers.append((position[0], position[1], config.food_radius))

    hazards: list[HazardState] = []
    for index in range(config.hazard_count):
        rng_state, position = place_without_overlap(rng_state, config.hazard_radius, config, blockers)
        rng_state, direction_unit = random_unit(rng_state)
        angle = direction_unit * math.pi * 2
        speed = 1.25 + index * 0.25
        velocity = (math.sin(angle) * speed, math.cos(angle) * speed)
        hazards.append(
            HazardState(position=position, previous_position=position, velocity=velocity, radius=config.hazard_radius)
        )
        blockers.append((position[0], position[1], config.hazard_radius))

    return WorldItem(
        seed=normalized_seed,
        rng_state=rng_state,
        tick=0,
        time_seconds=0.0,
        agents=agents,
        foods=foods,
        hazards=hazards,
    )


def create_world(seeds: Sequence[int], config: ArenaConfig = ARENA_CONFIG) -> list[WorldItem]:
    """Batch of `len(seeds)` independent world instances. See this module's
    doc comment for why this is a list rather than a dense tensor."""
    return [create_world_item(seed, config) for seed in seeds]


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


def step_world_item(
    item: WorldItem,
    actions: Mapping[str, Sequence[float] | None],
    config: ArenaConfig = ARENA_CONFIG,
) -> WorldItem:
    """Port of `stepWorld`, one fixed 30 Hz tick. `actions` maps agent id ->
    a 3-element `[thrust, yaw, brake]` array (or `None`/missing for the zero
    action). The input is not mutated."""
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


def step_world(
    items: Sequence[WorldItem],
    actions: Sequence[Mapping[str, Sequence[float] | None]],
    config: ArenaConfig = ARENA_CONFIG,
) -> list[WorldItem]:
    """Batched `stepWorld`: independently steps every item. See this
    module's doc comment for why this loops per item on CPU."""
    return [step_world_item(item, action, config) for item, action in zip(items, actions)]


def world_item_from_dict(data: Mapping, config: ArenaConfig = ARENA_CONFIG) -> WorldItem:
    """Reconstruct a `WorldItem` from a `SerializedWorld`/`SerializedWorldTick`
    dict (`scripts/training/export-traces.ts`'s `serializeWorld`/
    `serializeWorldTick`), e.g. a golden trace's `initialWorld` or a
    `--include-world` trace's `worldAfter[i]`. Used for teacher-forcing:
    resets this port's state to the recorded ground truth before each tick.

    `previousPosition`/`previousHeading` are not present in the serialized
    form (and are not needed): `step_world_item` overwrites both
    unconditionally at the start of every tick before reading them, exactly
    like `stepWorld` does (see `SerializedWorld`'s doc comment in
    export-traces.ts), so seeding them to the current value is correct, not
    a placeholder.

    Agents are matched to `AGENT_IDS` by `data["agentIds"]`, not by array
    position: today the two always coincide (`createWorld`'s fixed
    construction order), but reading the id explicitly means this function
    stays correct even if that ever changes independently of this module.
    """
    serialized_agent_ids = data["agentIds"]
    agent_positions = data["agentPositions"]
    agent_velocities = data["agentVelocities"]
    agent_headings = data["agentHeadings"]
    agent_scores = data["agentScores"]
    active_ids_by_agent = data.get("agentActiveHazardIds")

    agents: list[AgentState] = []
    for agent_id in AGENT_IDS:
        i = serialized_agent_ids.index(agent_id)
        active_hazard = [False] * config.hazard_count
        if active_ids_by_agent is not None:
            for hazard_id in active_ids_by_agent[i]:
                index = int(hazard_id.rsplit("-", 1)[1])
                active_hazard[index] = True
        position = tuple(agent_positions[i])
        agents.append(
            AgentState(
                position=position,
                previous_position=position,
                velocity=tuple(agent_velocities[i]),
                heading=agent_headings[i],
                previous_heading=agent_headings[i],
                radius=config.agent_radius,
                active_hazard=active_hazard,
                score=AgentScore(
                    food_pickups=agent_scores[i][0],
                    hazard_contacts=agent_scores[i][1],
                    distance_travelled=agent_scores[i][2],
                    movement_score=agent_scores[i][3],
                ),
            )
        )

    foods = [
        FoodState(position=tuple(position), radius=config.food_radius, respawns=respawns)
        for position, respawns in zip(data["foodPositions"], data["foodRespawns"])
    ]
    hazards = [
        HazardState(
            position=tuple(position),
            previous_position=tuple(position),
            velocity=tuple(velocity),
            radius=config.hazard_radius,
        )
        for position, velocity in zip(data["hazardPositions"], data["hazardVelocities"])
    ]

    return WorldItem(
        seed=0,
        rng_state=data["rngState"],
        tick=data["tick"],
        time_seconds=data["timeSeconds"],
        agents=agents,
        foods=foods,
        hazards=hazards,
    )
