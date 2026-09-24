"""Parity suite: `training/`'s PyTorch port vs the TypeScript golden traces
(`tests/fixtures/golden/`, or `FLYARENA_TRACE_DIR`). Tolerances match
`.agents/plans/trained-readout/02-gpu-port-and-parity.md`'s table exactly.

Every check here is teacher-forced (feeds the recorded TS inputs each tick,
so a mismatch cannot compound into the next tick) except
`test_free_running_world_positions`, which deliberately is not, per the
plan's "free-running" row.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flyarena_training.actions import decode_action
from flyarena_training.config import ARENA_CONFIG, assert_matches_fingerprint
from flyarena_training.graph import load_graph_json, output_neuron_indices
from flyarena_training.model import PreparedGraph, aggregate_outputs, create_model_state, run_substeps
from flyarena_training.readout import (
    broadcast_readout_weights,
    gather_output_rates,
    load_readout_weights_json,
    readout_forward,
)
from flyarena_training.sensors import observe_agent
from flyarena_training.world import step_world_item, world_item_from_dict

# Mirrors scripts/training/export-traces.ts's constants (DEFAULT_GRAPH_ID,
# TRACE_SEEDS, TRACE_SUBSTEPS, TRACE_TICKS). Not imported from TS: this is a
# Python test suite with no Node dependency for its default (non
# --include-world) checks.
GRAPH_ID = "trace-graph"
TRACE_SEEDS: tuple[int, ...] = (1, 2, 3, 12345)
TRACE_SUBSTEPS = 4

# The model/readout parity checks run on every device this host can exercise:
# CPU always, plus CUDA when available, per the plan's acceptance criteria
# ("It also passes on CUDA with the same tolerances, where CUDA is
# available."). World-state/observation/free-running checks are plain
# Python (no device concept), so they aren't parametrized here.
DEVICES: tuple[str, ...] = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

RATE_ABS_TOL = 1e-5
RATE_REL_TOL = 1e-4
OUTPUT_ABS_TOL = 1e-5
READOUT_ABS_TOL = 1e-6
DECODED_ACTION_ABS_TOL = 1e-5
WORLD_ABS_TOL = 1e-9
OBSERVATION_ABS_TOL = 1e-9
FREE_RUNNING_ABS_TOL = 1e-4


def _load_json(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def graph(trace_dir):
    """CPU copy, for tests (world-state/observation/free-running) that have
    no device concept of their own but still need graph metadata."""
    return load_graph_json(trace_dir / f"{GRAPH_ID}.json", device="cpu")


def test_config_matches_every_seed_fingerprint(trace_dir):
    """Config-drift gate. See config.py's module doc for why this parses
    the `configFingerprint` already embedded in each committed seed trace
    rather than a dedicated `arena-config.json` (which does not exist in
    this repository)."""
    for seed in TRACE_SEEDS:
        trace = _load_json(trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
        assert_matches_fingerprint(trace["configFingerprint"])


def test_precision_settings_applied():
    """Guards against an import-order regression silently re-enabling TF32
    (see `flyarena_training/__init__.py`)."""
    import flyarena_training  # noqa: F401  (import applies the settings)

    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.are_deterministic_algorithms_enabled() is True


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_rate_and_output_parity_teacher_forced(seed, device, trace_dir):
    """Feeds each tick's recorded observation with the *previous* tick's
    recorded rate (zero at tick 0) as the model's starting state — "recorded
    prior rates + recorded observation" in the plan's tolerance table — then
    compares the resulting rates/aggregated outputs/decoded action against
    what TS recorded for that same tick. All `ticks` rows of one seed are
    batched together (`B = ticks`), since teacher forcing makes every row
    independent of the others. Runs on every device in `DEVICES` (CPU
    always, CUDA when available) with the same tolerances.
    """
    trace = _load_json(trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    assert trace["substeps"] == TRACE_SUBSTEPS
    ticks = trace["ticks"]
    graph = load_graph_json(trace_dir / f"{GRAPH_ID}.json", device=device)
    neuron_count = graph.metadata.neuron_count

    prepared = PreparedGraph(graph, device)

    rates_after = trace["ratesAfter"]
    prior_rates = [[0.0] * neuron_count] + rates_after[:-1]
    observation_tensor = torch.tensor(trace["observations"], dtype=torch.float32, device=device)

    state = create_model_state(graph, batch_size=ticks, device=device)
    state.rate = torch.tensor(prior_rates, dtype=torch.float32, device=device)
    outputs = run_substeps(prepared, state, observation_tensor, TRACE_SUBSTEPS)

    expected_rate = torch.tensor(rates_after, dtype=torch.float64)
    actual_rate = state.rate.double().cpu()
    rate_diff = (actual_rate - expected_rate).abs()
    rate_ok = (rate_diff <= RATE_ABS_TOL) | (rate_diff <= RATE_REL_TOL * expected_rate.abs())
    assert bool(rate_ok.all()), (
        f"seed={seed} device={device}: rate parity failed, max abs diff {rate_diff.max().item():.3e} "
        f"at flat index {int(rate_diff.argmax())}"
    )

    expected_outputs = torch.tensor(trace["outputs"], dtype=torch.float64)
    output_diff = (outputs.double().cpu() - expected_outputs).abs()
    assert bool((output_diff <= OUTPUT_ABS_TOL).all()), (
        f"seed={seed} device={device}: aggregated-output parity failed, max abs diff "
        f"{output_diff.max().item():.3e}"
    )

    expected_actions = trace["actions"]
    for tick, output_row in enumerate(outputs.cpu().tolist()):
        decoded = decode_action(output_row)
        for actual, expected in zip(decoded, expected_actions[tick]):
            assert abs(actual - expected) <= DECODED_ACTION_ABS_TOL, (
                f"seed={seed} device={device} tick={tick}: decoded action mismatch {actual} vs {expected}"
            )

    print(
        f"[parity] seed={seed} device={device}: max rate abs diff {rate_diff.max().item():.3e}, "
        f"max output abs diff {output_diff.max().item():.3e}"
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_aggregated_output_parity_from_recorded_rates(seed, device, trace_dir):
    """Isolates the plan's "Aggregated outputs | recorded rates" tolerance
    row from rate-stepping: feeds `aggregate_outputs` the *recorded*
    `ratesAfter` directly (never our own computed rates), so this fails on
    an `aggregate_outputs`/`output_weight`/`output_population_index` bug
    even if `step_model` were bug-for-bug compatible with it (the more
    general `test_rate_and_output_parity_teacher_forced` above would not
    distinguish the two)."""
    trace = _load_json(trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    graph = load_graph_json(trace_dir / f"{GRAPH_ID}.json", device=device)
    prepared = PreparedGraph(graph, device)

    state = create_model_state(graph, batch_size=trace["ticks"], device=device)
    state.rate = torch.tensor(trace["ratesAfter"], dtype=torch.float32, device=device)
    outputs = aggregate_outputs(prepared, state)

    expected_outputs = torch.tensor(trace["outputs"], dtype=torch.float64)
    diff = (outputs.double().cpu() - expected_outputs).abs()
    max_abs = diff.max().item()
    assert max_abs <= OUTPUT_ABS_TOL, (
        f"seed={seed} device={device}: aggregated-output-from-recorded-rates parity failed, "
        f"max abs diff {max_abs:.3e}"
    )
    print(f"[parity] aggregated outputs (recorded rates) seed={seed} device={device}: max abs diff {max_abs:.3e}")


@pytest.mark.parametrize("device", DEVICES)
def test_readout_parity_teacher_forced(device, trace_dir):
    """readoutForward parity: recorded rates + recorded weights, batched
    across every tick of the committed readout case's designated seed. Runs
    on every device in `DEVICES`."""
    readout_case = _load_json(trace_dir / f"{GRAPH_ID}-readout.json")
    seed = readout_case["seed"]
    trace = _load_json(trace_dir / f"{GRAPH_ID}-seed-{seed}.json")

    graph = load_graph_json(trace_dir / f"{GRAPH_ID}.json", device=device)
    weights = load_readout_weights_json(trace_dir / f"{GRAPH_ID}-readout.json", device)
    indices = output_neuron_indices(graph)

    rates = torch.tensor(trace["ratesAfter"], dtype=torch.float32, device=device)  # [T, N]
    gathered = gather_output_rates(graph, rates, indices)  # [T, D]
    batched_weights = broadcast_readout_weights(weights, rates.shape[0])
    out = readout_forward(batched_weights, gathered)  # [T, 3]

    expected = torch.tensor(readout_case["outputs"], dtype=torch.float64)
    diff = (out.double().cpu() - expected).abs()
    max_abs = diff.max().item()
    assert max_abs <= READOUT_ABS_TOL, f"device={device}: readout parity failed, max abs diff {max_abs:.3e}"
    print(f"[parity] readout (seed={seed}, device={device}): max abs diff {max_abs:.3e}")


@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_world_state_and_observation_parity_teacher_forced(seed, include_world_trace_dir):
    """Requires the `--include-world` trace (see conftest.py). Each tick:
    observe from the *recorded* prior world state, step with the *recorded*
    decoded action, then compare the result (and food-respawn/hazard-contact
    event counters, checked exactly) against the recorded post-step state
    before resetting to it for the next tick."""
    trace = _load_json(include_world_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    ticks = trace["ticks"]

    prior = world_item_from_dict(trace["initialWorld"])
    max_world_abs = 0.0
    max_obs_abs = 0.0
    for tick in range(ticks):
        observation = observe_agent(prior, "left", ARENA_CONFIG)
        expected_observation = trace["observations"][tick]
        for actual, expected in zip(observation, expected_observation):
            diff = abs(actual - expected)
            max_obs_abs = max(max_obs_abs, diff)
            assert diff <= OBSERVATION_ABS_TOL, f"seed={seed} tick={tick}: observation mismatch"

        action = trace["actions"][tick]
        stepped = step_world_item(prior, {"left": action, "right": [0.0, 0.0, 0.0]})
        expected_after = trace["worldAfter"][tick]

        # Exact event checks.
        for i, food in enumerate(stepped.foods):
            assert food.respawns == expected_after["foodRespawns"][i], (
                f"seed={seed} tick={tick} food {i}: respawn count {food.respawns} != "
                f"{expected_after['foodRespawns'][i]}"
            )
        for i, agent in enumerate(stepped.agents):
            assert agent.score.food_pickups == expected_after["agentScores"][i][0]
            assert agent.score.hazard_contacts == expected_after["agentScores"][i][1]

        # World-state numeric parity.
        for i, agent in enumerate(stepped.agents):
            expected_position = expected_after["agentPositions"][i]
            for actual, expected in zip(agent.position, expected_position):
                diff = abs(actual - expected)
                max_world_abs = max(max_world_abs, diff)
                assert diff <= WORLD_ABS_TOL, f"seed={seed} tick={tick} agent {i}: position mismatch"
            for actual, expected in zip(agent.velocity, expected_after["agentVelocities"][i]):
                assert abs(actual - expected) <= WORLD_ABS_TOL
            assert abs(agent.heading - expected_after["agentHeadings"][i]) <= WORLD_ABS_TOL
        for i, food in enumerate(stepped.foods):
            for actual, expected in zip(food.position, expected_after["foodPositions"][i]):
                diff = abs(actual - expected)
                max_world_abs = max(max_world_abs, diff)
                assert diff <= WORLD_ABS_TOL, f"seed={seed} tick={tick} food {i}: position mismatch"
        for i, hazard in enumerate(stepped.hazards):
            for actual, expected in zip(hazard.position, expected_after["hazardPositions"][i]):
                assert abs(actual - expected) <= WORLD_ABS_TOL

        prior = world_item_from_dict(expected_after)  # teacher-force for the next tick

    print(
        f"[parity] world/observation seed={seed}: max world abs diff {max_world_abs:.3e}, "
        f"max observation abs diff {max_obs_abs:.3e}"
    )


@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_free_running_world_positions(seed, include_world_trace_dir):
    """No teacher forcing: replays `initialWorld` forward through our own
    `step_world_item` for all committed ticks using the recorded decoded
    actions, without resetting to the recorded world state each tick —
    catches systematic port errors (e.g. an operation-order slip) that
    per-tick teacher forcing could mask."""
    trace = _load_json(include_world_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    ticks = trace["ticks"]

    world = world_item_from_dict(trace["initialWorld"])
    max_abs = 0.0
    for tick in range(ticks):
        action = trace["actions"][tick]
        world = step_world_item(world, {"left": action, "right": [0.0, 0.0, 0.0]})
        expected_after = trace["worldAfter"][tick]
        for i, agent in enumerate(world.agents):
            for actual, expected in zip(agent.position, expected_after["agentPositions"][i]):
                diff = abs(actual - expected)
                max_abs = max(max_abs, diff)
                assert diff <= FREE_RUNNING_ABS_TOL, (
                    f"seed={seed} tick={tick} agent {i}: free-running position drift {diff:.3e}"
                )

    print(f"[parity] free-running seed={seed}: max position abs diff over {ticks} ticks {max_abs:.3e}")
