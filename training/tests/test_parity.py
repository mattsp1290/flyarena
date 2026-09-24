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
from flyarena_training.sensors import observe_batch
from flyarena_training.world import step_world_batched, world_batch_from_dicts, world_batch_from_items, world_item_from_dict

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
    """Requires the `--include-world` trace (see conftest.py). Teacher-forced
    and fully batched: every tick's recorded prior world state becomes one
    row of a `B = ticks` `WorldBatch` (`world_batch_from_dicts`), observed
    (`observe_batch`) and stepped (`step_world_batched`) all at once — each
    row is independent under teacher forcing, so this is a real exercise of
    the batched port across many rows in a single call, not `B = 1` in a
    Python loop. Compares the result (and food-respawn/food-pickup/hazard-
    contact event counters, checked exactly) against the recorded post-step
    state for every tick."""
    trace = _load_json(include_world_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    ticks = trace["ticks"]

    prior_dicts = [trace["initialWorld"]] + trace["worldAfter"][:-1]
    state = world_batch_from_dicts(prior_dicts, device="cpu")

    observation = observe_batch(state, "left", ARENA_CONFIG)
    expected_observation = torch.tensor(trace["observations"], dtype=torch.float64)
    obs_diff = (observation - expected_observation).abs()
    assert bool((obs_diff <= OBSERVATION_ABS_TOL).all()), (
        f"seed={seed}: observation parity failed, max abs diff {obs_diff.max().item():.3e}"
    )

    actions = torch.tensor(trace["actions"], dtype=torch.float64)
    zero_actions = torch.zeros_like(actions)
    stepped = step_world_batched(state, {"left": actions, "right": zero_actions}, ARENA_CONFIG)

    expected_food_respawns = torch.tensor([w["foodRespawns"] for w in trace["worldAfter"]], dtype=torch.int64)
    assert torch.equal(stepped.food_respawns, expected_food_respawns), f"seed={seed}: food respawn count mismatch"
    expected_scores = torch.tensor([w["agentScores"] for w in trace["worldAfter"]], dtype=torch.int64)
    assert torch.equal(stepped.agent_food_pickups, expected_scores[:, :, 0]), f"seed={seed}: food pickup count mismatch"
    assert torch.equal(
        stepped.agent_hazard_contacts, expected_scores[:, :, 1]
    ), f"seed={seed}: hazard contact count mismatch"

    expected_position = torch.tensor([w["agentPositions"] for w in trace["worldAfter"]], dtype=torch.float64)
    expected_velocity = torch.tensor([w["agentVelocities"] for w in trace["worldAfter"]], dtype=torch.float64)
    expected_heading = torch.tensor([w["agentHeadings"] for w in trace["worldAfter"]], dtype=torch.float64)
    expected_food_position = torch.tensor([w["foodPositions"] for w in trace["worldAfter"]], dtype=torch.float64)
    expected_hazard_position = torch.tensor([w["hazardPositions"] for w in trace["worldAfter"]], dtype=torch.float64)

    position_diff = (stepped.agent_position - expected_position).abs()
    velocity_diff = (stepped.agent_velocity - expected_velocity).abs()
    heading_diff = (stepped.agent_heading - expected_heading).abs()
    food_position_diff = (stepped.food_position - expected_food_position).abs()
    hazard_position_diff = (stepped.hazard_position - expected_hazard_position).abs()

    max_world_abs = max(
        position_diff.max().item(),
        velocity_diff.max().item(),
        heading_diff.max().item(),
        food_position_diff.max().item(),
        hazard_position_diff.max().item(),
    )
    assert bool((position_diff <= WORLD_ABS_TOL).all()), f"seed={seed}: agent position parity failed"
    assert bool((velocity_diff <= WORLD_ABS_TOL).all()), f"seed={seed}: agent velocity parity failed"
    assert bool((heading_diff <= WORLD_ABS_TOL).all()), f"seed={seed}: agent heading parity failed"
    assert bool((food_position_diff <= WORLD_ABS_TOL).all()), f"seed={seed}: food position parity failed"
    assert bool((hazard_position_diff <= WORLD_ABS_TOL).all()), f"seed={seed}: hazard position parity failed"

    print(
        f"[parity] world/observation (batched, B={ticks}) seed={seed}: max world abs diff "
        f"{max_world_abs:.3e}, max observation abs diff {obs_diff.max().item():.3e}"
    )


@pytest.mark.parametrize("seed", TRACE_SEEDS)
def test_free_running_world_positions(seed, include_world_trace_dir):
    """No teacher forcing: replays `initialWorld` forward through
    `step_world_batched` (`B = 1`) for all committed ticks using the
    recorded decoded actions, without resetting to the recorded world state
    each tick — catches systematic port errors (e.g. an operation-order
    slip) that per-tick teacher forcing could mask. See
    `test_free_running_world_positions_batched_across_seeds` below for the
    same check batched across all seeds at once (`B = len(TRACE_SEEDS)`)."""
    trace = _load_json(include_world_trace_dir / f"{GRAPH_ID}-seed-{seed}.json")
    ticks = trace["ticks"]

    state = world_batch_from_items([world_item_from_dict(trace["initialWorld"])], device="cpu")
    max_abs = 0.0
    for tick in range(ticks):
        action = torch.tensor([trace["actions"][tick]], dtype=torch.float64)
        zero_action = torch.zeros_like(action)
        state = step_world_batched(state, {"left": action, "right": zero_action}, ARENA_CONFIG)
        expected_after = trace["worldAfter"][tick]
        expected_position = torch.tensor([expected_after["agentPositions"]], dtype=torch.float64)
        diff = (state.agent_position - expected_position).abs()
        max_abs = max(max_abs, diff.max().item())
        assert bool((diff <= FREE_RUNNING_ABS_TOL).all()), (
            f"seed={seed} tick={tick}: free-running position drift {diff.max().item():.3e}"
        )

    print(f"[parity] free-running seed={seed}: max position abs diff over {ticks} ticks {max_abs:.3e}")


def test_free_running_world_positions_batched_across_seeds(include_world_trace_dir):
    """Same check as `test_free_running_world_positions`, but batched across
    every `TRACE_SEEDS` row at once (`B = len(TRACE_SEEDS)`, sequential over
    ticks) instead of one `step_world_batched(B=1)` call per seed — a direct
    exercise of `step_world_batched` doing independent, unrelated work for
    different batch rows simultaneously."""
    traces = [_load_json(include_world_trace_dir / f"{GRAPH_ID}-seed-{seed}.json") for seed in TRACE_SEEDS]
    ticks = traces[0]["ticks"]
    assert all(trace["ticks"] == ticks for trace in traces)

    state = world_batch_from_items([world_item_from_dict(trace["initialWorld"]) for trace in traces], device="cpu")
    max_abs = 0.0
    for tick in range(ticks):
        action = torch.tensor([trace["actions"][tick] for trace in traces], dtype=torch.float64)
        zero_action = torch.zeros_like(action)
        state = step_world_batched(state, {"left": action, "right": zero_action}, ARENA_CONFIG)
        expected_position = torch.tensor(
            [trace["worldAfter"][tick]["agentPositions"] for trace in traces], dtype=torch.float64
        )
        diff = (state.agent_position - expected_position).abs()
        max_abs = max(max_abs, diff.max().item())
        assert bool((diff <= FREE_RUNNING_ABS_TOL).all()), (
            f"tick={tick}: free-running position drift {diff.max().item():.3e} (batched across seeds)"
        )

    print(f"[parity] free-running (batched across {len(TRACE_SEEDS)} seeds): max position abs diff {max_abs:.3e}")
