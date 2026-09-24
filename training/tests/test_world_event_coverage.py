"""Parity + event-coverage checks against the longer, wider-seed-coverage
trace `long_trace_dir` generates (`training/scripts/generate_long_traces.ts`).

Addresses thermo-architecture review finding #2: the committed golden
fixtures (60 ticks x 4 seeds) never trigger a food pickup, wall clamp, or
hazard contact, so the respawn/rejection-sampling code path
(`place_without_overlap`, invoked from `step_world_batched`'s masked
fallback) and the wall-clamp/hazard-contact branches of `step_world_batched`
were never exercised by any test, even indirectly. This module both (a)
re-runs the same teacher-forced parity check as
`test_parity.py::test_world_state_and_observation_parity_teacher_forced`
against a trace that actually contains these events, and (b) asserts the
recorded trace itself contains a minimum number of each event, so this
coverage can't silently regress back to zero (e.g. if `generate_long_traces.ts`'s
default seed list were edited without checking)."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from flyarena_training.config import ARENA_CONFIG
from flyarena_training.sensors import observe_batch
from flyarena_training.world import step_world_batched, world_batch_from_dicts

from conftest import LONG_TRACE_SEEDS

GRAPH_ID = "trace-graph"
WORLD_ABS_TOL = 1e-9
OBSERVATION_ABS_TOL = 1e-9

# Minimum event counts the committed default seed list
# (`generate_long_traces.ts`'s `DEFAULT_SEEDS`/`DEFAULT_TICKS`) must produce,
# summed across all seeds, for this coverage to be considered non-regressed.
# Measured actual totals when this test was written: 15 food respawns, 42
# hazard contacts, 63 wall-clamp ticks (see training/README.md) — thresholds
# below are set well under those measured totals so minor, legitimate future
# changes (e.g. a different but still-diverse seed list) don't flake this
# test, while still catching a regression back to "zero events" (the
# original bug this module exists to guard against).
MIN_FOOD_RESPAWNS = 5
MIN_HAZARD_CONTACTS = 10
MIN_WALL_CLAMP_TICKS = 5


def _load_json(path: Path):
    return json.loads(path.read_text())


def _is_wall_clamped(position: list[float]) -> bool:
    max_x = ARENA_CONFIG.half_width - ARENA_CONFIG.agent_radius
    max_z = ARENA_CONFIG.half_depth - ARENA_CONFIG.agent_radius
    return abs(position[0]) == max_x or abs(position[1]) == max_z


def test_long_trace_event_coverage_thresholds(long_trace_dir):
    """Asserts the long trace itself (not this port's replay of it) contains
    at least the minimum number of each rare event — guards against
    `generate_long_traces.ts`'s seed list silently regressing to
    low-coverage seeds."""
    total_food_respawns = 0
    total_hazard_contacts = 0
    total_wall_clamp_ticks = 0

    for seed in LONG_TRACE_SEEDS:
        trace = _load_json(long_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
        world_after = trace["worldAfter"]
        assert len(world_after) > 0
        last_tick = world_after[-1]
        total_food_respawns += sum(last_tick["foodRespawns"])
        total_hazard_contacts += sum(score[1] for score in last_tick["agentScores"])
        for tick_state in world_after:
            for position in tick_state["agentPositions"]:
                if _is_wall_clamped(position):
                    total_wall_clamp_ticks += 1

    assert total_food_respawns >= MIN_FOOD_RESPAWNS, (
        f"total food respawns {total_food_respawns} < {MIN_FOOD_RESPAWNS}; the respawn/rejection-sampling "
        "code path would go untested again"
    )
    assert total_hazard_contacts >= MIN_HAZARD_CONTACTS, f"total hazard contacts {total_hazard_contacts}"
    assert total_wall_clamp_ticks >= MIN_WALL_CLAMP_TICKS, f"total wall-clamp ticks {total_wall_clamp_ticks}"

    print(
        f"[coverage] long trace: food respawns={total_food_respawns}, "
        f"hazard contacts={total_hazard_contacts}, wall-clamp ticks={total_wall_clamp_ticks}"
    )


def test_long_trace_world_and_observation_parity_teacher_forced(long_trace_dir):
    """Same teacher-forced, fully-batched parity check as
    `test_parity.py::test_world_state_and_observation_parity_teacher_forced`,
    against the long, event-rich trace: every (seed, tick) row across all of
    `LONG_TRACE_SEEDS` is batched into one `WorldBatch` (`B = sum(ticks per
    seed)`) and stepped in a single `step_world_batched` call, which
    directly exercises the masked per-item respawn fallback against
    real TS-recorded pickups (not just the property-based cross-check in
    `test_world_batch.py`)."""
    all_prior_dicts: list = []
    all_observations: list = []
    all_actions: list = []
    all_expected_after: list = []

    for seed in LONG_TRACE_SEEDS:
        trace = _load_json(long_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
        prior_dicts = [trace["initialWorld"]] + trace["worldAfter"][:-1]
        all_prior_dicts.extend(prior_dicts)
        all_observations.extend(trace["observations"])
        all_actions.extend(trace["actions"])
        all_expected_after.extend(trace["worldAfter"])

    state = world_batch_from_dicts(all_prior_dicts, device="cpu")
    observation = observe_batch(state, "left", ARENA_CONFIG)
    expected_observation = torch.tensor(all_observations, dtype=torch.float64)
    obs_diff = (observation - expected_observation).abs()
    assert bool((obs_diff <= OBSERVATION_ABS_TOL).all()), f"observation parity failed, max abs diff {obs_diff.max().item():.3e}"

    actions = torch.tensor(all_actions, dtype=torch.float64)
    zero_actions = torch.zeros_like(actions)
    stepped = step_world_batched(state, {"left": actions, "right": zero_actions}, ARENA_CONFIG)

    expected_food_respawns = torch.tensor([w["foodRespawns"] for w in all_expected_after], dtype=torch.int64)
    assert torch.equal(stepped.food_respawns, expected_food_respawns), "food respawn count mismatch"
    expected_scores = torch.tensor([w["agentScores"] for w in all_expected_after], dtype=torch.int64)
    assert torch.equal(stepped.agent_food_pickups, expected_scores[:, :, 0]), "food pickup count mismatch"
    assert torch.equal(stepped.agent_hazard_contacts, expected_scores[:, :, 1]), "hazard contact count mismatch"

    expected_position = torch.tensor([w["agentPositions"] for w in all_expected_after], dtype=torch.float64)
    expected_food_position = torch.tensor([w["foodPositions"] for w in all_expected_after], dtype=torch.float64)
    position_diff = (stepped.agent_position - expected_position).abs()
    food_position_diff = (stepped.food_position - expected_food_position).abs()

    assert bool((position_diff <= WORLD_ABS_TOL).all()), (
        f"agent position parity failed, max abs diff {position_diff.max().item():.3e}"
    )
    assert bool((food_position_diff <= WORLD_ABS_TOL).all()), (
        f"food position parity failed (respawn placement), max abs diff {food_position_diff.max().item():.3e}"
    )

    print(
        f"[parity] long trace (batched, B={state.batch_size}): max position abs diff "
        f"{position_diff.max().item():.3e}, max food position abs diff {food_position_diff.max().item():.3e}, "
        f"max observation abs diff {obs_diff.max().item():.3e}"
    )
