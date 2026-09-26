"""Task-generality WP1 (`.agents/plans/task-generality/01-task-plumbing.md`):
cross-language parity between `training/src/flyarena_training/config.py`'s
`ARENA_TASKS`/`ARENA_TASK_FINGERPRINTS` and TS's `src/lib/arena/tasks.ts`
(`ARENA_TASKS`, `resolveArenaTask`), via the committed
`tests/fixtures/golden/tasks.json` (TS-computed fingerprints) and each
non-default task's committed golden trace
(`tests/fixtures/golden/tasks/<id>/`).

Mirrors `test_parity.py`'s own teacher-forced rate/output/action parity
checks, at the reduced tick count and single seed the per-task fixtures
were exported at (`scripts/training/export-traces.ts`'s `TASK_TRACE_TICKS`/
`TASK_TRACE_SEED`), with the same tolerances.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from flyarena_training.actions import decode_action
from flyarena_training.config import (
    ARENA_TASK_FINGERPRINTS,
    ARENA_TASKS,
    TS_FIELD_ORDER,
    assert_matches_fingerprint,
    parse_fingerprint,
    resolve_arena_task,
    resolve_arena_task_fingerprint,
)
from flyarena_training.graph import load_graph_json
from flyarena_training.model import PreparedGraph, create_model_state, run_substeps

# training/tests/test_tasks.py -> training/tests -> training -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_JSON_PATH = REPO_ROOT / "tests" / "fixtures" / "golden" / "tasks.json"
TASK_GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "golden" / "tasks"

GRAPH_ID = "trace-graph"
NON_DEFAULT_TASK_IDS: tuple[str, ...] = tuple(sorted(id for id in ARENA_TASKS if id != "default"))

RATE_ABS_TOL = 1e-5
RATE_REL_TOL = 1e-4
OUTPUT_ABS_TOL = 1e-5
DECODED_ACTION_ABS_TOL = 1e-5


def _load_json(path: Path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def ts_task_fingerprints() -> dict[str, str]:
    return _load_json(TASKS_JSON_PATH)


def test_every_task_id_present_in_committed_fingerprints(ts_task_fingerprints):
    assert set(ARENA_TASKS) == set(ts_task_fingerprints)
    assert set(ARENA_TASK_FINGERPRINTS) == set(ts_task_fingerprints)


@pytest.mark.parametrize("task_id", sorted(ARENA_TASKS))
def test_arena_tasks_match_ts_fingerprint(task_id, ts_task_fingerprints):
    """Each Python `ARENA_TASKS` entry matches `tasks.json` through
    `parse_fingerprint`/`TS_FIELD_ORDER` — the same numeric-only comparison
    `assert_matches_fingerprint` already runs for the default task alone,
    generalized here to every task id."""
    fingerprint = ts_task_fingerprints[task_id]
    parsed = parse_fingerprint(fingerprint)
    config = ARENA_TASKS[task_id]
    for ts_name, py_name in TS_FIELD_ORDER:
        expected = parsed[ts_name]
        actual = float(getattr(config, py_name))
        assert abs(actual - expected) <= 1e-12, (
            f"{task_id}.{py_name} = {actual} does not match TS {task_id}.{ts_name} = {expected}"
        )


def test_default_task_matches_ts_fingerprint_via_assert_matches_fingerprint(ts_task_fingerprints):
    # Re-proves the existing default-only drift check still agrees with the
    # per-task generalization above, using the committed tasks.json's own
    # "default" entry rather than a golden trace's embedded configFingerprint
    # (test_parity.py's test_config_matches_every_seed_fingerprint already
    # covers that source independently).
    assert_matches_fingerprint(ts_task_fingerprints["default"])


@pytest.mark.parametrize("task_id", sorted(ARENA_TASKS))
def test_arena_task_fingerprints_are_hand_copied_correctly(task_id, ts_task_fingerprints):
    """`ARENA_TASK_FINGERPRINTS`'s hand-copied literals must equal the
    committed `tasks.json` byte-for-byte (never merely numerically) — see
    that dict's own doc comment for why cli.py copies these strings verbatim
    instead of reformatting a Python float."""
    assert ARENA_TASK_FINGERPRINTS[task_id] == ts_task_fingerprints[task_id]
    assert resolve_arena_task_fingerprint(task_id) == ts_task_fingerprints[task_id]


def test_resolve_arena_task_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown arena task"):
        resolve_arena_task("not-a-real-task")


def test_resolve_arena_task_fingerprint_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown arena task"):
        resolve_arena_task_fingerprint("not-a-real-task")


# ---------------------------------------------------------------------------
# Per-task golden trace parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", NON_DEFAULT_TASK_IDS)
def test_task_trace_config_fingerprint(task_id, ts_task_fingerprints):
    """The committed per-task trace's own `configFingerprint` (recorded by
    `export-traces.ts` from the *actual* `createWorld` call it made) equals
    the task's fingerprint in `tasks.json` — proves the exporter really did
    resolve and pass the requested task's config through, not silently fall
    back to `ARENA_CONFIG`."""
    trace_dir = TASK_GOLDEN_DIR / task_id
    files = sorted(trace_dir.glob(f"{GRAPH_ID}-seed-*.json"))
    assert len(files) == 1, f"{task_id}: expected exactly one committed seed trace, found {len(files)}"
    trace = _load_json(files[0])
    assert trace["configFingerprint"] == ts_task_fingerprints[task_id]


@pytest.mark.parametrize("task_id", NON_DEFAULT_TASK_IDS)
def test_task_trace_rate_and_output_parity_teacher_forced(task_id):
    """Same teacher-forced rate/aggregated-output/decoded-action parity check
    as `test_parity.py`'s `test_rate_and_output_parity_teacher_forced`, run
    against this task's own committed (single-seed, reduced-tick) golden
    trace, at the same tolerances — arena tasks are config variants only, so
    the connectome model itself is not expected to behave any differently
    per task; this proves it doesn't."""
    trace_dir = TASK_GOLDEN_DIR / task_id
    files = sorted(trace_dir.glob(f"{GRAPH_ID}-seed-*.json"))
    assert len(files) == 1
    trace = _load_json(files[0])

    graph = load_graph_json(trace_dir / f"{GRAPH_ID}.json", device="cpu")
    neuron_count = graph.metadata.neuron_count
    prepared = PreparedGraph(graph, "cpu")

    ticks = trace["ticks"]
    substeps = trace["substeps"]
    rates_after = trace["ratesAfter"]
    prior_rates = [[0.0] * neuron_count] + rates_after[:-1]
    observation_tensor = torch.tensor(trace["observations"], dtype=torch.float32, device="cpu")

    state = create_model_state(graph, batch_size=ticks, device="cpu")
    state.rate = torch.tensor(prior_rates, dtype=torch.float32, device="cpu")
    outputs = run_substeps(prepared, state, observation_tensor, substeps)

    expected_rate = torch.tensor(rates_after, dtype=torch.float64)
    actual_rate = state.rate.double().cpu()
    rate_diff = (actual_rate - expected_rate).abs()
    rate_ok = (rate_diff <= RATE_ABS_TOL) | (rate_diff <= RATE_REL_TOL * expected_rate.abs())
    assert bool(rate_ok.all()), f"{task_id}: rate parity failed, max abs diff {rate_diff.max().item():.3e}"

    expected_outputs = torch.tensor(trace["outputs"], dtype=torch.float64)
    output_diff = (outputs.double().cpu() - expected_outputs).abs()
    assert bool((output_diff <= OUTPUT_ABS_TOL).all()), (
        f"{task_id}: aggregated-output parity failed, max abs diff {output_diff.max().item():.3e}"
    )

    expected_actions = trace["actions"]
    for tick, output_row in enumerate(outputs.cpu().tolist()):
        decoded = decode_action(output_row)
        for actual, expected in zip(decoded, expected_actions[tick]):
            assert abs(actual - expected) <= DECODED_ACTION_ABS_TOL, (
                f"{task_id} tick={tick}: decoded action mismatch {actual} vs {expected}"
            )
