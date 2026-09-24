"""`WorldBatch` construction, reset parity, `validate_world_batch`, and a
property-based cross-check of `step_world_batched` against the per-item
reference oracle (`tests/reference_world.py`).

Addresses thermo-architecture review finding #2's "zero test coverage"
observation for `create_world_item`/reset placement: every committed golden
seed file already has `initialWorld` (TS's `createWorld(seed)` output), but
no test previously compared it against this port's own reset construction
directly — `test_parity.py`'s world tests only ever *deserialized* a TS
world, never built one from a seed and checked it.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from flyarena_training.config import ARENA_CONFIG
from flyarena_training.world import (
    create_world,
    step_world_batched,
    validate_world_batch,
    world_batch_from_items,
    world_batch_to_items,
)

from reference_world import step_world_item_reference

GRAPH_ID = "trace-graph"
TRACE_SEEDS: tuple[int, ...] = (1, 2, 3, 12345)
WORLD_ABS_TOL = 1e-9


def _load_json(path: Path):
    return json.loads(path.read_text())


@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_create_world_batch_matches_initial_world_trace(seed, trace_dir):
    """`create_world([seed])` stacked into a `WorldBatch`, compared directly
    against the committed golden seed file's `initialWorld` (TS's
    `createWorld(seed)`) — the reset/placement path itself, not just a
    deserialization of it."""
    trace = _load_json(trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    expected = trace["initialWorld"]

    state = world_batch_from_items(create_world([seed]), device="cpu")

    expected_positions = torch.tensor([expected["agentPositions"]], dtype=torch.float64)
    expected_velocities = torch.tensor([expected["agentVelocities"]], dtype=torch.float64)
    expected_headings = torch.tensor([expected["agentHeadings"]], dtype=torch.float64)
    expected_food_positions = torch.tensor([expected["foodPositions"]], dtype=torch.float64)
    expected_hazard_positions = torch.tensor([expected["hazardPositions"]], dtype=torch.float64)
    expected_hazard_velocities = torch.tensor([expected["hazardVelocities"]], dtype=torch.float64)

    assert bool(((state.agent_position - expected_positions).abs() <= WORLD_ABS_TOL).all())
    assert bool(((state.agent_velocity - expected_velocities).abs() <= WORLD_ABS_TOL).all())
    assert bool(((state.agent_heading - expected_headings).abs() <= WORLD_ABS_TOL).all())
    assert bool(((state.food_position - expected_food_positions).abs() <= WORLD_ABS_TOL).all())
    assert bool(((state.hazard_position - expected_hazard_positions).abs() <= WORLD_ABS_TOL).all())
    assert bool(((state.hazard_velocity - expected_hazard_velocities).abs() <= WORLD_ABS_TOL).all())
    assert state.rng_state.item() == expected["rngState"]
    assert state.tick.item() == expected["tick"] == 0
    assert state.food_respawns.sum().item() == 0
    assert state.agent_food_pickups.sum().item() == 0
    assert state.agent_hazard_contacts.sum().item() == 0


def test_batched_step_matches_per_item_reference_across_random_actions():
    """Property-based cross-check (not TS-derived): step a 24-item batch for
    300 ticks under a randomized "drive forward, wandering steering" action
    policy (full thrust, no brake, yaw doing a bounded random walk —
    reliably drifts agents into walls and food across a 24 x 16 unit arena,
    unlike pure per-tick-independent uniform random actions, which mostly
    average out to jitter near the spawn point and rarely trigger a pickup
    or wall clamp within a reasonable tick budget) through both
    `step_world_batched` and the per-item `step_world_item_reference`, and
    asserts they agree at every tick. This is a stronger backstop than the
    golden-trace fixtures alone, which only ever cover a fixed, small set of
    seeds/ticks/actions (thermo-architecture review finding #1's "small
    per-item reference implementation ... needed as a test oracle")."""
    rng = random.Random(1290)
    seeds = list(range(1, 25))
    items = create_world(seeds)
    state = world_batch_from_items(items, device="cpu")

    yaw_left = [0.0] * len(seeds)
    yaw_right = [0.0] * len(seeds)
    for tick in range(300):
        yaw_left = [max(-1.0, min(1.0, y + rng.uniform(-0.2, 0.2))) for y in yaw_left]
        yaw_right = [max(-1.0, min(1.0, y + rng.uniform(-0.2, 0.2))) for y in yaw_right]
        left = torch.tensor([[1.0, y, 0.0] for y in yaw_left], dtype=torch.float64)
        right = torch.tensor([[1.0, y, 0.0] for y in yaw_right], dtype=torch.float64)
        state = step_world_batched(state, {"left": left, "right": right}, ARENA_CONFIG)
        items = [
            step_world_item_reference(item, {"left": left[i].tolist(), "right": right[i].tolist()})
            for i, item in enumerate(items)
        ]

    batched_items = world_batch_to_items(state)
    assert len(batched_items) == len(items)
    for index, (batched, reference) in enumerate(zip(batched_items, items)):
        for a in range(2):
            for c in range(2):
                assert abs(batched.agents[a].position[c] - reference.agents[a].position[c]) <= WORLD_ABS_TOL, (
                    f"seed index {index} agent {a}: position mismatch"
                )
                assert abs(batched.agents[a].velocity[c] - reference.agents[a].velocity[c]) <= WORLD_ABS_TOL
            assert abs(batched.agents[a].heading - reference.agents[a].heading) <= WORLD_ABS_TOL
            assert batched.agents[a].score.food_pickups == reference.agents[a].score.food_pickups, (
                f"seed index {index} agent {a}: food_pickups mismatch"
            )
            assert batched.agents[a].score.hazard_contacts == reference.agents[a].score.hazard_contacts, (
                f"seed index {index} agent {a}: hazard_contacts mismatch"
            )
        for f in range(len(batched.foods)):
            assert batched.foods[f].respawns == reference.foods[f].respawns, (
                f"seed index {index} food {f}: respawns mismatch"
            )
            for c in range(2):
                assert abs(batched.foods[f].position[c] - reference.foods[f].position[c]) <= WORLD_ABS_TOL

    # Sanity: this property test is only meaningful if the masked respawn
    # fallback and hazard contacts actually ran at least once.
    total_respawns = sum(item.foods[f].respawns for item in items for f in range(len(item.foods)))
    total_hazard_contacts = sum(agent.score.hazard_contacts for item in items for agent in item.agents)
    assert total_respawns > 0, "property test did not exercise the masked respawn fallback at all"
    assert total_hazard_contacts > 0, "property test did not exercise any hazard contacts"


def test_validate_world_batch_accepts_a_fresh_reset():
    state = world_batch_from_items(create_world([1, 2, 3]), device="cpu")
    validate_world_batch(state, ARENA_CONFIG)  # must not raise


def test_validate_world_batch_rejects_non_finite_position():
    state = world_batch_from_items(create_world([1]), device="cpu")
    state.agent_position[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="agent_position"):
        validate_world_batch(state, ARENA_CONFIG)


def test_validate_world_batch_rejects_excess_speed():
    state = world_batch_from_items(create_world([1]), device="cpu")
    state.agent_velocity[0, 0, 0] = ARENA_CONFIG.max_speed * 10
    with pytest.raises(ValueError, match="max_speed"):
        validate_world_batch(state, ARENA_CONFIG)


def test_validate_world_batch_rejects_clock_mismatch():
    state = world_batch_from_items(create_world([1]), device="cpu")
    state.time_seconds[0] = 999.0
    with pytest.raises(ValueError, match="time_seconds"):
        validate_world_batch(state, ARENA_CONFIG)


def test_step_world_batched_runs_validation_by_default():
    state = world_batch_from_items(create_world([1]), device="cpu")
    state.time_seconds[0] = 999.0
    with pytest.raises(ValueError, match="time_seconds"):
        step_world_batched(state, {})


def test_step_world_batched_can_skip_validation():
    state = world_batch_from_items(create_world([1]), device="cpu")
    state.time_seconds[0] = 999.0
    # Should not raise: validation explicitly disabled.
    step_world_batched(state, {}, validate=False)
