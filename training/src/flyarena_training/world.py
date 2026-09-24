"""Port of `src/lib/arena/world.ts`: `createWorld`, `stepWorld`, and the
`nextRandomState`-driven rejection-sampling placement they share.

Design note on "batched": the deterministic per-tick physics (movement
integration, wall clamping, hazard motion/bounce, contact detection, and
score bookkeeping) is dense `[B, ...]` `torch` tensor ops in
`step_world_batched` below, operating on `WorldBatch` (float64, on `device`).
Only `placeWithoutOverlap` — sequential rejection sampling whose attempt
count is data-dependent (it depends on where blockers already are) — stays
per-item, on CPU: it is used at reset (`create_world_item`, invoked once per
seed to build the tensors `world_batch_from_items` stacks) and, inside
`step_world_batched`, as a **masked** fallback that runs `place_without_overlap`
only for the batch items whose agent ate a food that tick (typically a small
fraction of `B`), then scatters the results back into the batched food-position
tensor. This is the fix the thermo-architecture review's finding #1 asked
for: everything that doesn't need data-dependent rejection sampling is now a
batched tensor op; only the part that genuinely can't be is a per-item loop,
and it runs over a small subset of the batch rather than every item every
tick.

Every arithmetic op below (batched or per-item) is IEEE-754 binary64,
matching JS `number` exactly — porting each TS line in the same operation
order reproduces TS's result to within a handful of ULPs (the `%` operator
is the one place this needs care: JS `%` matches C `fmod` (sign of the
dividend), not Python's `%` (sign of the divisor), so `wrap_angle`/
`wrap_angle_batched` below use `math.fmod`/`torch.fmod`, not `%`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch

from .actions import decode_action_batch
from .config import ARENA_CONFIG, ArenaConfig
from .rng import normalize_seed, random_unit

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


def wrap_angle_batched(angle: torch.Tensor) -> torch.Tensor:
    """Batched port of `wrapAngle`, elementwise. See `wrap_angle`'s doc for
    the fmod-vs-`%` note; `torch.fmod` matches C `fmod` sign semantics like
    `math.fmod` does."""
    two_pi = math.pi * 2
    wrapped = torch.fmod(torch.fmod(angle + math.pi, two_pi) + two_pi, two_pi) - math.pi
    return torch.where(wrapped == -math.pi, torch.full_like(wrapped, math.pi), wrapped)


def _overlaps(ax: float, az: float, a_radius: float, bx: float, bz: float, b_radius: float) -> bool:
    return math.hypot(ax - bx, az - bz) <= a_radius + b_radius


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
    """Port of `placeWithoutOverlap`. `blockers` is `(x, z, radius)` triples.
    The one genuinely sequential, data-dependent piece of this module (see
    the module doc); used per-item on CPU both at reset and, inside
    `step_world_batched`, as a masked per-item fallback for food respawn."""
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
    """One simulation instance. Mirrors `WorldState` (`src/lib/arena/types.ts`).
    Used for reset construction (`create_world_item`) and to reconstruct a
    single instance from serialized JSON (`world_item_from_dict`); stacked
    into a `WorldBatch` by `world_batch_from_items` for actual stepping."""

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
    """`len(seeds)` independent reset instances, per-item on CPU (rejection
    sampling; see the module doc). Feed the result to `world_batch_from_items`
    to get the dense `WorldBatch` `step_world_batched` steps."""
    return [create_world_item(seed, config) for seed in seeds]


def world_item_from_dict(data: Mapping, config: ArenaConfig = ARENA_CONFIG) -> WorldItem:
    """Reconstruct a `WorldItem` from a `SerializedWorld`/`SerializedWorldTick`
    dict (`scripts/training/export-traces.ts`'s `serializeWorld`/
    `serializeWorldTick`), e.g. a golden trace's `initialWorld` or a
    `--include-world` trace's `worldAfter[i]`. Used for teacher-forcing:
    resets this port's state to the recorded ground truth before each tick.

    `previousPosition`/`previousHeading` are not present in the serialized
    form (and are not needed): `step_world_batched` never reads a batch
    item's previous position/heading across ticks (see `WorldBatch`'s doc),
    exactly like `stepWorld` overwrites both unconditionally at the start of
    every tick before reading them (see `SerializedWorld`'s doc comment in
    export-traces.ts), so seeding them to the current value here is correct,
    not a placeholder.

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


@dataclass
class WorldBatch:
    """Dense `[B, ...]` batched world state, float64, on `device`. Stepped by
    `step_world_batched`. Built from a list of `WorldItem` (`world_batch_from_items`)
    or directly from seeds (`create_world_batch`).

    Shapes: `agent_*` tensors have a size-2 axis in `AGENT_IDS` order
    (`left`, `right`); `food_*`/`hazard_*` tensors have a `food_count`/
    `hazard_count`-sized axis in `create_world_item`'s fixed construction
    order. `agent_radius`/`food_radius`/`hazard_radius` are plain floats, not
    tensors: every item shares the same arena config, so per-entity radius is
    always the same scalar across the whole batch (`AgentState.radius` etc.
    are always `config.*_radius`).

    No `previous_position`/`previous_heading` fields: nothing downstream
    needs them (see `world_item_from_dict`'s doc comment) and
    `step_world_batched` computes distance-travelled directly from the
    incoming vs. outgoing position each call, so there is nothing to carry
    across ticks.
    """

    seed: torch.Tensor  # [B] int64
    rng_state: torch.Tensor  # [B] int64, holds a uint32 value
    tick: torch.Tensor  # [B] int64
    time_seconds: torch.Tensor  # [B] float64
    agent_position: torch.Tensor  # [B, 2, 2] float64 (agent, xz)
    agent_velocity: torch.Tensor  # [B, 2, 2] float64
    agent_heading: torch.Tensor  # [B, 2] float64
    agent_active_hazard: torch.Tensor  # [B, 2, hazard_count] bool
    agent_food_pickups: torch.Tensor  # [B, 2] int64
    agent_hazard_contacts: torch.Tensor  # [B, 2] int64
    agent_distance_travelled: torch.Tensor  # [B, 2] float64
    agent_movement_score: torch.Tensor  # [B, 2] float64
    food_position: torch.Tensor  # [B, food_count, 2] float64
    food_respawns: torch.Tensor  # [B, food_count] int64
    hazard_position: torch.Tensor  # [B, hazard_count, 2] float64
    hazard_velocity: torch.Tensor  # [B, hazard_count, 2] float64
    agent_radius: float
    food_radius: float
    hazard_radius: float

    @property
    def batch_size(self) -> int:
        return self.agent_position.shape[0]

    @property
    def device(self) -> torch.device:
        return self.agent_position.device


def world_batch_from_items(
    items: Sequence[WorldItem], device: str | torch.device = "cpu", config: ArenaConfig = ARENA_CONFIG
) -> WorldBatch:
    """Stack `B` per-item `WorldItem`s (from `create_world`/`create_world_item`
    or `world_item_from_dict`) into a dense `WorldBatch` on `device`. All
    items must share the same `food_count`/`hazard_count` (true for every
    item built from the same `config`)."""
    if len(items) == 0:
        raise ValueError("world_batch_from_items requires at least one item")
    device = torch.device(device)

    def agent_tensor(select, dtype):
        return torch.tensor([[select(item.agents[a]) for a in range(2)] for item in items], dtype=dtype, device=device)

    agent_position = agent_tensor(lambda a: list(a.position), torch.float64)
    agent_velocity = agent_tensor(lambda a: list(a.velocity), torch.float64)
    agent_heading = agent_tensor(lambda a: a.heading, torch.float64)
    agent_active_hazard = agent_tensor(lambda a: list(a.active_hazard), torch.bool)
    agent_food_pickups = agent_tensor(lambda a: a.score.food_pickups, torch.int64)
    agent_hazard_contacts = agent_tensor(lambda a: a.score.hazard_contacts, torch.int64)
    agent_distance_travelled = agent_tensor(lambda a: a.score.distance_travelled, torch.float64)
    agent_movement_score = agent_tensor(lambda a: a.score.movement_score, torch.float64)

    food_position = torch.tensor(
        [[list(f.position) for f in item.foods] for item in items], dtype=torch.float64, device=device
    )
    food_respawns = torch.tensor(
        [[f.respawns for f in item.foods] for item in items], dtype=torch.int64, device=device
    )
    hazard_position = torch.tensor(
        [[list(h.position) for h in item.hazards] for item in items], dtype=torch.float64, device=device
    )
    hazard_velocity = torch.tensor(
        [[list(h.velocity) for h in item.hazards] for item in items], dtype=torch.float64, device=device
    )

    return WorldBatch(
        seed=torch.tensor([item.seed for item in items], dtype=torch.int64, device=device),
        rng_state=torch.tensor([item.rng_state for item in items], dtype=torch.int64, device=device),
        tick=torch.tensor([item.tick for item in items], dtype=torch.int64, device=device),
        time_seconds=torch.tensor([item.time_seconds for item in items], dtype=torch.float64, device=device),
        agent_position=agent_position,
        agent_velocity=agent_velocity,
        agent_heading=agent_heading,
        agent_active_hazard=agent_active_hazard,
        agent_food_pickups=agent_food_pickups,
        agent_hazard_contacts=agent_hazard_contacts,
        agent_distance_travelled=agent_distance_travelled,
        agent_movement_score=agent_movement_score,
        food_position=food_position,
        food_respawns=food_respawns,
        hazard_position=hazard_position,
        hazard_velocity=hazard_velocity,
        agent_radius=config.agent_radius,
        food_radius=config.food_radius,
        hazard_radius=config.hazard_radius,
    )


def create_world_batch(
    seeds: Sequence[int], device: str | torch.device = "cpu", config: ArenaConfig = ARENA_CONFIG
) -> WorldBatch:
    """`create_world(seeds)` stacked into one `WorldBatch` on `device`."""
    return world_batch_from_items(create_world(seeds, config), device, config)


def world_batch_from_dicts(
    data: Sequence[Mapping], device: str | torch.device = "cpu", config: ArenaConfig = ARENA_CONFIG
) -> WorldBatch:
    """`world_item_from_dict` for every row, stacked into one `WorldBatch`."""
    return world_batch_from_items([world_item_from_dict(d, config) for d in data], device, config)


def world_batch_to_items(state: WorldBatch) -> list[WorldItem]:
    """Inverse of `world_batch_from_items`: one `WorldItem` per batch row, on
    CPU. Used by tests to compare `step_world_batched` against the per-item
    reference oracle (`training/tests/reference_world.py`) row by row."""
    agent_position = state.agent_position.detach().cpu().tolist()
    agent_velocity = state.agent_velocity.detach().cpu().tolist()
    agent_heading = state.agent_heading.detach().cpu().tolist()
    agent_active_hazard = state.agent_active_hazard.detach().cpu().tolist()
    agent_food_pickups = state.agent_food_pickups.detach().cpu().tolist()
    agent_hazard_contacts = state.agent_hazard_contacts.detach().cpu().tolist()
    agent_distance_travelled = state.agent_distance_travelled.detach().cpu().tolist()
    agent_movement_score = state.agent_movement_score.detach().cpu().tolist()
    food_position = state.food_position.detach().cpu().tolist()
    food_respawns = state.food_respawns.detach().cpu().tolist()
    hazard_position = state.hazard_position.detach().cpu().tolist()
    hazard_velocity = state.hazard_velocity.detach().cpu().tolist()
    seed = state.seed.detach().cpu().tolist()
    rng_state = state.rng_state.detach().cpu().tolist()
    tick = state.tick.detach().cpu().tolist()
    time_seconds = state.time_seconds.detach().cpu().tolist()

    items: list[WorldItem] = []
    for b in range(state.batch_size):
        agents = [
            AgentState(
                position=tuple(agent_position[b][a]),
                previous_position=tuple(agent_position[b][a]),
                velocity=tuple(agent_velocity[b][a]),
                heading=agent_heading[b][a],
                previous_heading=agent_heading[b][a],
                radius=state.agent_radius,
                active_hazard=list(agent_active_hazard[b][a]),
                score=AgentScore(
                    food_pickups=agent_food_pickups[b][a],
                    hazard_contacts=agent_hazard_contacts[b][a],
                    distance_travelled=agent_distance_travelled[b][a],
                    movement_score=agent_movement_score[b][a],
                ),
            )
            for a in range(2)
        ]
        foods = [
            FoodState(position=tuple(food_position[b][f]), radius=state.food_radius, respawns=food_respawns[b][f])
            for f in range(len(food_position[b]))
        ]
        hazards = [
            HazardState(
                position=tuple(hazard_position[b][h]),
                previous_position=tuple(hazard_position[b][h]),
                velocity=tuple(hazard_velocity[b][h]),
                radius=state.hazard_radius,
            )
            for h in range(len(hazard_position[b]))
        ]
        items.append(
            WorldItem(
                seed=seed[b],
                rng_state=rng_state[b],
                tick=tick[b],
                time_seconds=time_seconds[b],
                agents=agents,
                foods=foods,
                hazards=hazards,
            )
        )
    return items


def validate_world_batch(state: WorldBatch, config: ArenaConfig = ARENA_CONFIG) -> None:
    """Batched, cheap equivalent of TS `stepWorld`'s `validateStepState`
    guard (`src/lib/arena/world.ts`'s `validateStepState`): finite-value and
    clock-consistency checks across the whole batch at once via
    `torch.isfinite(...).all()`/comparison reductions, raising `ValueError`
    naming which invariant failed (not narrowed to the offending batch
    index — a second, per-item pass to find it would defeat the point of a
    cheap batched guard; callers that need to localize a failure can fall
    back to `world_batch_to_items` + per-item inspection).

    One deliberate simplification vs. TS: TS checks `world.timeSeconds`
    against `tick * fixedDeltaSeconds` with `Object.is` (bit-exact). This
    batched port allows a small float64 tolerance (`1e-9`) instead of exact
    equality, since a `WorldBatch` can be built from externally-serialized
    JSON (`world_batch_from_dicts`) where the recorded `timeSeconds` and a
    freshly recomputed `tick * fixedDeltaSeconds` are two independently
    JSON-round-tripped doubles that are not guaranteed to agree to the last
    ULP even when they represent the same value; TS's own in-process
    `Object.is` check never faces that round trip. `schemaVersion` is not
    tracked by this port (nothing here ever sees a different one), so it is
    not checked."""
    if (state.tick < 0).any():
        raise ValueError("Invalid world clock state: tick is negative for at least one batch item")

    expected_time = state.tick.to(state.time_seconds.dtype) * config.fixed_delta_seconds
    if not torch.isfinite(state.time_seconds).all() or not torch.isfinite(expected_time).all():
        raise ValueError("Invalid world clock state: time_seconds or tick * fixed_delta_seconds is not finite")
    if ((state.time_seconds - expected_time).abs() > 1e-9).any():
        raise ValueError(
            "Invalid world clock state: time_seconds does not match tick * fixed_delta_seconds "
            "for at least one batch item"
        )

    numeric_fields = (
        ("agent_position", state.agent_position),
        ("agent_velocity", state.agent_velocity),
        ("agent_heading", state.agent_heading),
        ("agent_distance_travelled", state.agent_distance_travelled),
        ("agent_movement_score", state.agent_movement_score),
        ("food_position", state.food_position),
        ("hazard_position", state.hazard_position),
        ("hazard_velocity", state.hazard_velocity),
    )
    for name, tensor in numeric_fields:
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid world state numeric value: non-finite value found in {name}")

    speed = torch.hypot(state.agent_velocity[..., 0], state.agent_velocity[..., 1])
    if (speed > config.max_speed * (1 + 1e-12)).any():
        raise ValueError("Invalid world state numeric value: agent speed exceeds max_speed")

    if (state.agent_food_pickups < 0).any() or (state.agent_hazard_contacts < 0).any():
        raise ValueError("Invalid world state numeric value: negative food_pickups or hazard_contacts count")


def step_world_batched(
    state: WorldBatch,
    actions: Mapping[str, torch.Tensor | None],
    config: ArenaConfig = ARENA_CONFIG,
    validate: bool = True,
) -> WorldBatch:
    """Batched port of `stepWorld`, one fixed 30 Hz tick for every item in
    `state` at once. `actions` maps agent id -> a `[B, 3]` `(thrust, yaw,
    brake)` tensor, or `None`/missing for the zero action (matches
    `decode_action_batch`). `state` is not mutated; a new `WorldBatch` is
    returned. `validate=True` (default, matching `stepWorld` always calling
    `validateStepState`) runs `validate_world_batch` on the input first;
    pass `False` to skip it in a perf-critical rollout loop that already
    trusts its own state (WP3) after establishing it is well-formed.

    All deterministic physics (heading/velocity/position integration, wall
    clamp, hazard movement/bounce, food/hazard contact detection, score
    updates) is dense tensor ops over the whole batch. The one exception is
    food respawn placement (`place_without_overlap`): it runs as a masked
    per-item CPU loop over only the batch items that had a pickup this tick,
    then scatters the results back — see the module doc.
    """
    if validate:
        validate_world_batch(state, config)

    device = state.device
    dtype = state.agent_position.dtype
    batch_size = state.batch_size
    dt = config.fixed_delta_seconds

    decoded = torch.stack(
        [decode_action_batch(actions.get(agent_id), batch_size, device, dtype) for agent_id in AGENT_IDS],
        dim=1,
    )  # [B, 2, 3]
    thrust, yaw, brake = decoded[..., 0], decoded[..., 1], decoded[..., 2]

    heading = wrap_angle_batched(state.agent_heading + yaw * config.turn_rate * dt)  # [B, 2]
    vx = state.agent_velocity[..., 0] + torch.sin(heading) * thrust * config.acceleration * dt
    vz = state.agent_velocity[..., 1] + torch.cos(heading) * thrust * config.acceleration * dt
    drag = (1 - (config.rolling_drag + brake * config.brake_drag) * dt).clamp(min=0.0)
    vx = vx * drag
    vz = vz * drag
    speed = torch.hypot(vx, vz)
    over_speed = speed > config.max_speed
    scale = torch.where(over_speed, config.max_speed / speed.clamp(min=1e-300), torch.ones_like(speed))
    vx = vx * scale
    vz = vz * scale

    prev_position = state.agent_position
    px = prev_position[..., 0] + vx * dt
    pz = prev_position[..., 1] + vz * dt

    agent_max_x = config.half_width - config.agent_radius
    agent_max_z = config.half_depth - config.agent_radius
    out_x = (px < -agent_max_x) | (px > agent_max_x)
    out_z = (pz < -agent_max_z) | (pz > agent_max_z)
    px = torch.where(out_x, px.clamp(-agent_max_x, agent_max_x), px)
    vx = torch.where(out_x, torch.zeros_like(vx), vx)
    pz = torch.where(out_z, pz.clamp(-agent_max_z, agent_max_z), pz)
    vz = torch.where(out_z, torch.zeros_like(vz), vz)

    new_agent_position = torch.stack([px, pz], dim=-1)  # [B, 2, 2]
    new_agent_velocity = torch.stack([vx, vz], dim=-1)

    distance = torch.hypot(
        new_agent_position[..., 0] - prev_position[..., 0], new_agent_position[..., 1] - prev_position[..., 1]
    )
    agent_distance_travelled = state.agent_distance_travelled + distance
    agent_movement_score = state.agent_movement_score + distance * config.movement_score_per_unit

    # Hazard movement + wall bounce.
    hvx0, hvz0 = state.hazard_velocity[..., 0], state.hazard_velocity[..., 1]
    hx = state.hazard_position[..., 0] + hvx0 * dt
    hz = state.hazard_position[..., 1] + hvz0 * dt
    hazard_max_x = config.half_width - config.hazard_radius
    hazard_max_z = config.half_depth - config.hazard_radius
    hout_x = (hx < -hazard_max_x) | (hx > hazard_max_x)
    hout_z = (hz < -hazard_max_z) | (hz > hazard_max_z)
    hx = torch.where(hout_x, hx.clamp(-hazard_max_x, hazard_max_x), hx)
    hvx = torch.where(hout_x, -hvx0, hvx0)
    hz = torch.where(hout_z, hz.clamp(-hazard_max_z, hazard_max_z), hz)
    hvz = torch.where(hout_z, -hvz0, hvz0)
    new_hazard_position = torch.stack([hx, hz], dim=-1)  # [B, H, 2]
    new_hazard_velocity = torch.stack([hvx, hvz], dim=-1)

    # Food contacts: for each food, the first agent (in AGENT_IDS order) that
    # overlaps it collects it — matches processContacts's `.find` (first
    # match wins, not nearest/all).
    food_position = state.food_position  # [B, F, 2]; foods don't move on their own
    food_diff = food_position.unsqueeze(2) - new_agent_position.unsqueeze(1)  # [B, F, 2(agent), 2(xz)]
    food_dist = torch.linalg.norm(food_diff, dim=-1)  # [B, F, 2]
    food_overlap = food_dist <= (config.agent_radius + config.food_radius)
    overlap_left = food_overlap[..., 0]
    overlap_right = food_overlap[..., 1] & ~overlap_left  # left has priority, matching AGENT_IDS order
    picked_mask = overlap_left | overlap_right  # [B, F]

    pickups_left = overlap_left.sum(dim=1)
    pickups_right = overlap_right.sum(dim=1)
    agent_food_pickups = torch.stack(
        [state.agent_food_pickups[:, 0] + pickups_left, state.agent_food_pickups[:, 1] + pickups_right], dim=1
    )
    agent_movement_score = torch.stack(
        [
            agent_movement_score[:, 0] + pickups_left.to(dtype) * config.food_score,
            agent_movement_score[:, 1] + pickups_right.to(dtype) * config.food_score,
        ],
        dim=1,
    )

    new_food_position = food_position.clone()
    food_respawns = state.food_respawns.clone()
    rng_state = state.rng_state.clone()

    items_needing = torch.nonzero(picked_mask.any(dim=1), as_tuple=False).flatten()
    if items_needing.numel() > 0:
        picked_mask_cpu = picked_mask.index_select(0, items_needing).cpu().tolist()
        rng_state_cpu = rng_state.index_select(0, items_needing).cpu().tolist()
        food_position_cpu = new_food_position.index_select(0, items_needing).cpu().tolist()
        food_respawns_cpu = food_respawns.index_select(0, items_needing).cpu().tolist()
        agent_position_cpu = new_agent_position.index_select(0, items_needing).cpu().tolist()
        hazard_position_cpu = new_hazard_position.index_select(0, items_needing).cpu().tolist()

        food_count = food_position.shape[1]
        hazard_count = new_hazard_position.shape[1]
        for k in range(items_needing.numel()):
            item_foods = food_position_cpu[k]
            for f in range(food_count):
                if not picked_mask_cpu[k][f]:
                    continue
                blockers: list[tuple[float, float, float]] = [
                    (agent_position_cpu[k][0][0], agent_position_cpu[k][0][1], config.agent_radius),
                    (agent_position_cpu[k][1][0], agent_position_cpu[k][1][1], config.agent_radius),
                ]
                blockers += [
                    (item_foods[j][0], item_foods[j][1], config.food_radius) for j in range(food_count) if j != f
                ]
                blockers += [
                    (hazard_position_cpu[k][h][0], hazard_position_cpu[k][h][1], config.hazard_radius)
                    for h in range(hazard_count)
                ]
                rng_state_cpu[k], new_position = place_without_overlap(
                    rng_state_cpu[k], config.food_radius, config, blockers
                )
                item_foods[f] = list(new_position)
                food_respawns_cpu[k][f] += 1

        new_food_position.index_copy_(
            0, items_needing, torch.tensor(food_position_cpu, dtype=new_food_position.dtype, device=device)
        )
        food_respawns.index_copy_(
            0, items_needing, torch.tensor(food_respawns_cpu, dtype=food_respawns.dtype, device=device)
        )
        rng_state.index_copy_(0, items_needing, torch.tensor(rng_state_cpu, dtype=rng_state.dtype, device=device))

    # Hazard contacts: activeHazardIds-equivalent bool mask, transition-edge
    # scoring (a contact only scores on the tick a hazard becomes active).
    hazard_diff = new_hazard_position.unsqueeze(1) - new_agent_position.unsqueeze(2)  # [B, 2(agent), H, 2(xz)]
    hazard_dist = torch.linalg.norm(hazard_diff, dim=-1)  # [B, 2, H]
    active_hazard = hazard_dist <= (config.agent_radius + config.hazard_radius)  # [B, 2, H]
    new_contacts = active_hazard & ~state.agent_active_hazard  # [B, 2, H]
    hazard_contacts_delta = new_contacts.sum(dim=2)  # [B, 2]
    agent_hazard_contacts = state.agent_hazard_contacts + hazard_contacts_delta
    agent_movement_score = agent_movement_score - hazard_contacts_delta.to(dtype) * config.hazard_penalty

    new_tick = state.tick + 1
    new_time_seconds = new_tick.to(dtype) * dt

    return WorldBatch(
        seed=state.seed,
        rng_state=rng_state,
        tick=new_tick,
        time_seconds=new_time_seconds,
        agent_position=new_agent_position,
        agent_velocity=new_agent_velocity,
        agent_heading=heading,
        agent_active_hazard=active_hazard,
        agent_food_pickups=agent_food_pickups,
        agent_hazard_contacts=agent_hazard_contacts,
        agent_distance_travelled=agent_distance_travelled,
        agent_movement_score=agent_movement_score,
        food_position=new_food_position,
        food_respawns=food_respawns,
        hazard_position=new_hazard_position,
        hazard_velocity=new_hazard_velocity,
        agent_radius=state.agent_radius,
        food_radius=state.food_radius,
        hazard_radius=state.hazard_radius,
    )
